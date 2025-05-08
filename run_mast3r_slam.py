# master_slam/run_mast3r_slam.py
# This script integrates MASt3R-SLAM with RabbitMQ for input and output.
# Prompts:
# - User request to implement MASt3R-SLAM in a Dockerfile and integrate with RabbitMQ.
# - Consume RGB images from VIDEO_FRAMES_EXCHANGE.
# - Output pose, keyframes, point clouds, dense map to separate RabbitMQ exchanges.
# - Use camera intrinsics from MASt3R-SLAM's config/intrinsics.yaml.
# - Based on the structure of existing slam/aaa/run_rgbd.py and MASt3R-SLAM's main.py.

import os
import sys
import json
import time
import cv2
import numpy as np
import pika
import argparse
import base64
import yaml
import threading
from PIL import Image
import io
import random # For dummy data, to be removed

import torch
import lietorch # From MASt3R-SLAM dependencies
import torch.multiprocessing as mp # Added for backend process

# MASt3R-SLAM specific imports
from omegaconf import OmegaConf, open_dict
from mast3r_slam.config import (
    load_config as mast3r_load_config_internal,
    config as mast3r_global_config, # Access the global config
    set_global_config as mast3r_set_global_config,
    update_config as mast3r_update_config
)
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import Mode as MASt3RMode, create_frame, SharedKeyframes
from mast3r_slam.mast3r_utils import load_mast3r, load_retriever
from mast3r_slam.tracker import FrameTracker
import mast3r_slam.evaluate as mast3r_eval # For saving mesh
from mast3r_slam.global_opt import FactorGraph # Needed for run_backend
from mast3r_slam.mast3r_utils import load_retriever # Needed for run_backend

# PROMETHEUS (Optional - can be added later if needed, similar to run_rgbd.py)
# from prometheus_client import start_http_server, Counter, Histogram, Gauge

# --- RabbitMQ Configuration ---
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
VIDEO_FRAMES_EXCHANGE = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")

# Output Exchanges (new for MASt3R-SLAM)
MAST3R_SLAM_POSE_EXCHANGE = os.getenv("MAST3R_SLAM_POSE_EXCHANGE", "mast3r_slam_pose_exchange")
MAST3R_SLAM_KEYFRAME_EXCHANGE = os.getenv("MAST3R_SLAM_KEYFRAME_EXCHANGE", "mast3r_slam_keyframe_exchange")
MAST3R_SLAM_POINTCLOUD_EXCHANGE = os.getenv("MAST3R_SLAM_POINTCLOUD_EXCHANGE", "mast3r_slam_pointcloud_exchange")
MAST3R_SLAM_DENSE_MAP_EXCHANGE = os.getenv("MAST3R_SLAM_DENSE_MAP_EXCHANGE", "mast3r_slam_dense_map_exchange")
RESTART_EXCHANGE = os.getenv("RESTART_EXCHANGE", "restart_exchange")

# Define a directory for saving maps/meshes if that functionality is used
SAVE_MAP_DIR = "/maps/mast3r_slam" # This path should be a mounted volume in Docker

# --- Top-level functions for the backend process (adapted from MASt3R-SLAM's main.py) ---

def relocalization_for_backend(current_frame, keyframes_shared, factor_graph_instance, retrieval_db_instance, global_cfg):
    # Adapted from relocalization in MASt3R-SLAM main.py
    # Uses passed instances and global_cfg directly
    with keyframes_shared.lock:
        kf_idx = []
        retrieval_inds = retrieval_db_instance.update(
            current_frame,
            add_after_query=False,
            k=global_cfg.retrieval.k,
            min_thresh=global_cfg.retrieval.min_thresh,
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes_shared.append(current_frame) # current_frame is added to keyframes_shared here
            n_kf = len(keyframes_shared)
            kf_idx = list(kf_idx)  
            frame_idx_list = [n_kf - 1] * len(kf_idx) # This is the index of the newly added current_frame
            
            print(f"[Backend] RELOCALIZING frame (now KF {n_kf - 1}) against KFs: {kf_idx}")
            if factor_graph_instance.add_factors(
                frame_idx_list, # Source is the new KF (current_frame)
                kf_idx,         # Target are existing KFs from retrieval
                global_cfg.reloc.min_match_frac,
                is_reloc=global_cfg.reloc.strict,
            ):
                retrieval_db_instance.update(
                    current_frame,
                    add_after_query=True, # Add to db after successful reloc
                    k=global_cfg.retrieval.k,
                    min_thresh=global_cfg.retrieval.min_thresh,
                )
                print("[Backend] Success! Relocalized.")
                successful_loop_closure = True
                # Pose of the new KF (current_frame) is updated by FactorGraph solve
            else:
                keyframes_shared.pop_last() # Remove current_frame if relocalization failed
                print("[Backend] Failed to relocalize.")

        if successful_loop_closure:
            if global_cfg.use_calib:
                factor_graph_instance.solve_GN_calib()
            else:
                factor_graph_instance.solve_GN_rays()
        return successful_loop_closure

def run_slam_backend_process(passed_config, mast3r_model_shared, states_shared, keyframes_shared, K_intrinsics_tensor):
    """
    Adapted from run_backend in MASt3R-SLAM's main.py.
    This function is intended to be run as a separate process.
    """
    mast3r_set_global_config(passed_config) # Set the global config for this process
    # Re-access global config for clarity within this function scope, though it's now set globally
    global_cfg = mast3r_global_config 

    print(f"[Backend PID {os.getpid()}] Process started. use_calib: {global_cfg.use_calib}")
    device = keyframes_shared.device # Get device from shared keyframes
    
    factor_graph_instance = FactorGraph(mast3r_model_shared, keyframes_shared, K_intrinsics_tensor, device, config=global_cfg)
    retrieval_db_instance = load_retriever(mast3r_model_shared, config=global_cfg.retrieval)

    mode = states_shared.get_mode()
    while mode is not MASt3RMode.TERMINATED:
        mode = states_shared.get_mode()
        if mode == MASt3RMode.INIT or states_shared.is_paused():
            time.sleep(0.01)
            continue
        
        if mode == MASt3RMode.RELOC:
            # Relocalization task handling
            current_frame_for_reloc = states_shared.get_frame() # Gets frame added via add_reloc_task
            if current_frame_for_reloc:
                print(f"[Backend] RELOC task received for frame_idx {current_frame_for_reloc.frame_idx}")
                success = relocalization_for_backend(
                    current_frame_for_reloc, keyframes_shared, factor_graph_instance, retrieval_db_instance, global_cfg
                )
                if success:
                    states_shared.set_mode(MASt3RMode.TRACKING) # Back to tracking after successful reloc
                else:
                    # If reloc fails, tracker might still be lost. Tracker handles its state.
                    # Mode might remain RELOC or be set by tracker/main loop.
                    # For now, we assume if reloc fails, it stays in a state where tracker might try again or stay lost.
                    # The original main.py sets mode back to tracking only on success.
                    # If reloc fails, the frame added to keyframes by relocalization_for_backend is popped.
                    pass 
                states_shared.dequeue_reloc() # Remove the processed reloc task
            else:
                # This case should ideally not happen if add_reloc_task always adds a frame.
                # If it does, switch back to tracking to avoid getting stuck.
                print("[Backend] Warning: RELOC mode but no frame from states.get_frame(). Switching to TRACKING.")
                states_shared.set_mode(MASt3RMode.TRACKING)
            continue # Loop back to check mode again

        # Global optimization task handling
        kf_opt_idx = -1
        with states_shared.lock:
            if len(states_shared.global_optimizer_tasks) > 0:
                kf_opt_idx = states_shared.global_optimizer_tasks[0] # Peek at the first task
        
        if kf_opt_idx == -1:
            time.sleep(0.01) # No optimization tasks, wait a bit
            continue

        print(f"[Backend] OPTIMIZER task received for KF idx: {kf_opt_idx}")
        
        # Graph Construction for optimization task
        # k to previous consecutive keyframes (original logic)
        connected_kf_indices = []
        n_consecutive_connections = 1 # As in original MASt3R-SLAM main.py run_backend
        for j in range(min(n_consecutive_connections, kf_opt_idx)):
            connected_kf_indices.append(kf_opt_idx - 1 - j)
        
        # Retrieval for loop closure candidates for this keyframe
        # The keyframe (kf_opt_idx) is already in keyframes_shared and retrieval_db_instance
        # if it was added by tracker and then its ID was put in global_optimizer_tasks.
        # If retrieval_db_instance is updated by tracker or here, ensure consistency.
        # Original `main.py` updates retrieval DB when KF is added by tracker.
        # Here, we assume `kf_opt_idx` is a valid KF. We might need to ensure it's in the DB if not already.
        current_kf_obj = keyframes_shared[kf_opt_idx]
        retrieval_inds = retrieval_db_instance.update(
            current_kf_obj, # Use the keyframe object itself
            add_after_query=True, # Ensure it is in DB if not already (idempotent if already there)
            k=global_cfg.retrieval.k,
            min_thresh=global_cfg.retrieval.min_thresh,
        )
        connected_kf_indices += retrieval_inds

        # Debug: print loop closure candidates
        lc_candidates = set(retrieval_inds)
        lc_candidates.discard(kf_opt_idx) # Should not loop with self
        if kf_opt_idx -1 in lc_candidates: lc_candidates.discard(kf_opt_idx -1) # Often connected to prev
        if lc_candidates:
            print(f"[Backend] Database retrieval for KF {kf_opt_idx} suggests potential loop closures with: {lc_candidates}")

        connected_kf_indices = set(connected_kf_indices)  # Remove duplicates
        if kf_opt_idx in connected_kf_indices: # Should not add factors to itself
             connected_kf_indices.discard(kf_opt_idx)
        connected_kf_indices = list(connected_kf_indices)
        
        source_kf_indices = [kf_opt_idx] * len(connected_kf_indices)
        
        if connected_kf_indices: # If there are KFs to connect to
            factor_graph_instance.add_factors(
                source_kf_indices, connected_kf_indices, global_cfg.local_opt.min_match_frac
            )

        # Update shared edge data for visualization (optional, but good to keep if viz is ever added)
        with states_shared.lock:
            states_shared.edges_ii[:] = factor_graph_instance.ii.cpu().tolist()
            states_shared.edges_jj[:] = factor_graph_instance.jj.cpu().tolist()

        # Solve the graph
        if global_cfg.use_calib:
            factor_graph_instance.solve_GN_calib()
        else:
            factor_graph_instance.solve_GN_rays()
        print(f"[Backend] Solved graph for KF idx: {kf_opt_idx}")

        # Remove the processed task from the queue
        with states_shared.lock:
            if len(states_shared.global_optimizer_tasks) > 0 and states_shared.global_optimizer_tasks[0] == kf_opt_idx:
                states_shared.global_optimizer_tasks.pop(0)
            else:
                print(f"[Backend] Warning: Optimizer task for KF {kf_opt_idx} might have been removed or changed.")

    print(f"[Backend PID {os.getpid()}] Process terminating.")

class MASt3RSLAMProcessor:
    def __init__(self, config_path: str, calib_path: str = None):
        self.config_path = config_path
        self.calib_path = calib_path
        
        self.slam_system_initialized = False
        self.mast3r_model = None
        self.tracker = None
        self.keyframes = None
        self.K_torch = None
        self.intrinsics_obj = None
        self.img_h = 0
        self.img_w = 0
        self.frame_idx_counter = 0
        self.last_image_timestamp_ns = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.states = None # Will be initialized after manager
        self.backend_process = None
        self.mp_manager = None

        torch.set_grad_enabled(False)
        if self.device == "cuda:0":
             torch.backends.cuda.matmul.allow_tf32 = True # From MASt3R-SLAM main.py

        try:
            # Initialize multiprocessing manager first if needed by Shared* objects
            # MASt3R-SLAM's SharedKeyframes and SharedStates take a manager object
            mp.set_start_method("spawn", force=True) # Ensure spawn method, essential for CUDA with mp
            self.mp_manager = mp.Manager()
        except RuntimeError as e:
            # Might happen if set_start_method is called more than once, or in wrong context
            print(f"Note: mp.set_start_method warning/error: {e}. Assuming already set or not critical.")
            if not self.mp_manager: # If manager failed to init due to this
                print("Falling back to trying mp.Manager() without explicit set_start_method again here.")
                try:
                    self.mp_manager = mp.Manager()
                except Exception as e_mgr:
                    print(f"CRITICAL: Failed to initialize mp.Manager: {e_mgr}. Backend process will not start.")
                    self.mp_manager = None # Ensure it's None

        self._load_slam_system()
        self._setup_rabbitmq()

        if self.slam_system_initialized:
            print(f"MASt3R-SLAM Processor initialized. Config: {config_path}, Calib: {calib_path if calib_path else 'Using config from main YAML'}")
            if self.mp_manager and self.backend_process is None: # Start backend only if manager is up
                print("Starting MASt3R-SLAM backend process...")
                self.backend_process = mp.Process(
                    target=run_slam_backend_process,
                    args=(
                        mast3r_global_config, # Pass the global config object
                        self.mast3r_model,    # Shared model
                        self.states,          # Shared states object
                        self.keyframes,       # Shared keyframes object
                        self.K_torch          # Camera intrinsics tensor
                    )
                )
                self.backend_process.daemon = True # Ensure it exits with the main process
                self.backend_process.start()
                print(f"MASt3R-SLAM backend process started with PID: {self.backend_process.pid}")
            elif not self.mp_manager:
                 print("WARNING: Multiprocessing manager failed to initialize. Backend process not started.")
        else:
            print(f"MASt3R-SLAM Processor FAILED to initialize. Config: {config_path}, Calib: {calib_path}")

    def _load_slam_system(self):
        print("Loading MASt3R-SLAM system...")
        try:
            # Load MASt3R-SLAM's main configuration file (e.g., config/base.yaml)
            # This function also sets a global config object within MASt3R-SLAM
            mast3r_load_config_internal(self.config_path)
            
            # If a separate calibration file is provided, load it and update the global config
            # MASt3R-SLAM's main.py logic:
            # if args.calib:
            #     with open(args.calib, "r") as f:
            #         intrinsics_data = yaml.load(f, Loader=yaml.SafeLoader)
            #     config["use_calib"] = True  <- This is part of global config, might need to be set
            #     dataset.use_calibration = True <- specific to their dataloader, we adapt
            #     dataset.camera_intrinsics = Intrinsics.from_calib(...)
            
            intrinsics_data_loaded = None
            if self.calib_path and os.path.exists(self.calib_path):
                print(f"Loading intrinsics from: {self.calib_path}")
                with open(self.calib_path, "r") as f:
                    intrinsics_data_loaded = yaml.safe_load(f)
                
                # Update the global MASt3R-SLAM config if necessary
                # Example: enable 'use_calib' if the calib file implies it
                # This part is tricky as MASt3R-SLAM's main.py intertwines dataset config with global config
                with open_dict(mast3r_global_config): # Allow modification of OmegaConf
                    mast3r_global_config.use_calib = True
                    # If intrinsics_data_loaded contains specific camera parameters,
                    # MASt3R-SLAM might expect them under a certain key in its global config,
                    # or they are passed directly to objects. For now, we rely on Intrinsics.from_calib.
            else:
                print(f"Calibration file not provided or not found at '{self.calib_path}'. MASt3R-SLAM might use defaults from main config if 'use_calib' is true, or operate without explicit intrinsics if 'use_calib' is false.")
                # Ensure use_calib is consistent if no calib file
                if mast3r_global_config.get("use_calib", False):
                    print("Warning: MASt3R-SLAM config has 'use_calib=True' but no calibration file was loaded by the wrapper.")
                    # Optionally set mast3r_global_config.use_calib = False here if that's safer

            # Determine image dimensions - required for Intrinsics object
            # Option 1: From intrinsics file if available
            if intrinsics_data_loaded and 'width' in intrinsics_data_loaded and 'height' in intrinsics_data_loaded:
                self.img_w = intrinsics_data_loaded['width']
                self.img_h = intrinsics_data_loaded['height']
            # Option 2: From main config if available
            elif mast3r_global_config.dataset and 'SlamArgs' in mast3r_global_config.dataset and 'img_hw' in mast3r_global_config.dataset.SlamArgs:
                 self.img_h, self.img_w = mast3r_global_config.dataset.SlamArgs.img_hw
            else:
                # Fallback, MASt3R-SLAM might have its own defaults or fail.
                # We need this for SharedKeyframes. Let's use a common default or make it mandatory.
                print("Warning: Image width/height not found in calib file or main config. Using fallback 640x480.")
                self.img_w, self.img_h = 640, 480 # Common default, but should be accurate

            print(f"Using image dimensions: {self.img_w}x{self.img_h}")

            if mast3r_global_config.get("use_calib", False) and intrinsics_data_loaded:
                # Create Intrinsics object using data from the calib file
                # Intrinsics.from_calib(img_size_hw_tuple, calib_width, calib_height, calib_data_list)
                img_size_hw_tuple = (self.img_h, self.img_w)
                calib_list = intrinsics_data_loaded.get('calibration')
                if not calib_list:
                    raise ValueError("Calibration data list not found in intrinsics file under 'calibration' key.")
                
                self.intrinsics_obj = Intrinsics.from_calib(
                    img_size_hw_tuple,
                    intrinsics_data_loaded['width'], # width from calib file
                    intrinsics_data_loaded['height'],# height from calib file
                    calib_list
                )
                self.K_torch = self.intrinsics_obj.K_frame_torch.to(self.device, dtype=torch.float32)
                print(f"Loaded K_torch from calib file: {self.K_torch}")
            elif mast3r_global_config.get("use_calib", False) and not intrinsics_data_loaded:
                 raise ValueError("'use_calib' is true in config, but calibration file failed to load or parse.")
            else: # Not using explicit calibration file, or use_calib is false
                self.K_torch = None # MASt3R-SLAM might derive it or work without it (e.g. calib-free mode)
                print("Running without explicit K_torch from separate calib file. MASt3R-SLAM might use internal defaults or operate calibration-free.")


            # Load the MASt3R model
            self.mast3r_model = load_mast3r(config=mast3r_global_config.mast3r, device=self.device) # Pass the sub-config for mast3r
            self.mast3r_model.share_memory() # From main.py

            # Initialize SharedKeyframes. `manager=None` as we are not using mp.Manager here.
            # Height and width are needed.
            self.keyframes = SharedKeyframes(manager=self.mp_manager, h=self.img_h, w=self.img_w, device=self.device)
            if self.K_torch is not None:
                self.keyframes.set_intrinsics(self.K_torch)

            # Initialize the FrameTracker
            self.tracker = FrameTracker(self.mast3r_model, self.keyframes, self.device, config=mast3r_global_config) # Pass global config

            # Initialize SharedStates (needs mp_manager)
            self.states = SharedStates(manager=self.mp_manager, h=self.img_h, w=self.img_w, device=self.device)

            self.slam_system_initialized = True
            print("MASt3R-SLAM system components loaded successfully.")

        except Exception as e:
            print(f"Error loading MASt3R-SLAM system: {e}")
            import traceback
            traceback.print_exc()
            self.slam_system_initialized = False

    def _setup_rabbitmq(self):
        try:
            params = pika.URLParameters(RABBITMQ_URL)
            params.heartbeat = 3600 
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()

            exchanges_to_declare = [
                VIDEO_FRAMES_EXCHANGE, MAST3R_SLAM_POSE_EXCHANGE,
                MAST3R_SLAM_KEYFRAME_EXCHANGE, MAST3R_SLAM_POINTCLOUD_EXCHANGE,
                MAST3R_SLAM_DENSE_MAP_EXCHANGE, RESTART_EXCHANGE
            ]
            for ex_name in exchanges_to_declare:
                self.channel.exchange_declare(exchange=ex_name, exchange_type='fanout', durable=True)

            res_image = self.channel.queue_declare(queue='mast3r_slam_video_input', exclusive=True)
            self.image_queue_name = res_image.method.queue
            self.channel.queue_bind(exchange=VIDEO_FRAMES_EXCHANGE, queue=self.image_queue_name, routing_key='')

            res_restart = self.channel.queue_declare(queue='mast3r_slam_restart_input', durable=True)
            self.restart_queue_name = res_restart.method.queue
            self.channel.queue_bind(exchange=RESTART_EXCHANGE, queue=self.restart_queue_name, routing_key='')
            
            print(f"[*] Subscribed to {VIDEO_FRAMES_EXCHANGE} on queue {self.image_queue_name}")
            print(f"[*] Subscribed to {RESTART_EXCHANGE} on queue {self.restart_queue_name}")

        except Exception as e:
            print(f"Error setting up RabbitMQ: {e}")
            import traceback
            traceback.print_exc()
            raise

    def on_image_message(self, ch, method, properties, body):
        if not self.slam_system_initialized or not self.tracker or not self.states:
            print("SLAM system not fully initialized (tracker or states missing), cannot process frame.")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return
        try:
            start_time = time.time()
            
            frame_ts_ns = int(properties.headers.get("timestamp_ns", 0))
            if frame_ts_ns <= self.last_image_timestamp_ns and self.last_image_timestamp_ns != 0 : # Allow first frame
                print(f"Received outdated or duplicate frame: ts_ns={frame_ts_ns}, last_ts_ns={self.last_image_timestamp_ns}. Skipping.")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            self.last_image_timestamp_ns = frame_ts_ns
            current_timestamp_s = frame_ts_ns / 1e9

            try:
                image_pil = Image.open(io.BytesIO(body)).convert('RGB')
                # Check if image dimensions match configured dimensions
                if image_pil.width != self.img_w or image_pil.height != self.img_h:
                    print(f"Warning: Incoming image dimensions ({image_pil.width}x{image_pil.height}) "
                          f"do not match configured dimensions ({self.img_w}x{self.img_h}). This might cause issues.")
                    # Optionally, resize image_pil here if MASt3R-SLAM requires fixed size not handled by Intrinsics object
                    # image_pil = image_pil.resize((self.img_w, self.img_h))

                frame_rgb_np = np.array(image_pil)
            except Exception as e:
                print(f"Error decoding image: {e}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            if frame_rgb_np is None:
                print("Failed to decode image.")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            # Convert to torch tensor ( HxWxC (numpy) -> CxHxW (torch) )
            img_tensor = torch.from_numpy(frame_rgb_np).permute(2, 0, 1).to(
                self.device, non_blocking=True, dtype=torch.uint8 # MASt3R uses uint8
            )

            # Create a MASt3R-SLAM frame object
            # create_frame(rgb_tensor, depth_tensor, K_frame_torch, timestamp, frame_idx, device)
            # K_frame_torch should be for the current frame's resolution after any undistortion/rectification
            # self.K_torch is based on the intrinsics file, assuming images match that.
            current_K = self.K_torch
            if self.intrinsics_obj: # If loaded from calib file
                 # Get K for the specific image size if it was resized, or use the one from init
                 # For now, assume self.K_torch is correct for the image tensor being passed
                 current_K = self.intrinsics_obj.K_repr_torch(img_h=img_tensor.shape[1], img_w=img_tensor.shape[2]).to(self.device)


            slam_frame = create_frame(
                img_tensor, 
                None, # No depth sensor for mono
                current_K, # K for the current frame (possibly adjusted for resolution)
                current_timestamp_s, 
                self.frame_idx_counter, 
                device=self.device
            )

            # Process frame with MASt3R-SLAM tracker
            self.tracker.track_frame(slam_frame) # This updates slam_frame in-place (e.g., .T_WC_curr)
            
            tracking_lost = slam_frame.tracking_lost
            tracking_successful = not tracking_lost
            
            # Publish Pose
            current_pose_matrix_np = np.eye(4) # Default to identity
            if slam_frame.T_WC_curr is not None:
                try:
                    current_pose_matrix_np = slam_frame.T_WC_curr.matrix().cpu().numpy()
                except Exception as e:
                    print(f"Error getting pose matrix from T_WC_curr: {e}")
            
            pose_msg = {
                "timestamp_ns": frame_ts_ns,
                "pose": current_pose_matrix_np.tolist(),
                "tracking_success": tracking_successful,
                "tracking_lost_flag": tracking_lost, # Explicit flag from MASt3R
                "frame_id": self.frame_idx_counter,
                "algorithm": "MASt3R-SLAM"
            }
            self.channel.basic_publish(
                exchange=MAST3R_SLAM_POSE_EXCHANGE, routing_key='', body=json.dumps(pose_msg),
                properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
            )

            # Publish Keyframe Data if a new keyframe was made
            if slam_frame.is_keyframe:
                kf_id = slam_frame.id # This should be the index in self.keyframes
                kf_pose_matrix_np = self.keyframes.T_WC[kf_id].matrix().cpu().numpy() # Get pose from SharedKeyframes store
                
                kf_msg = {
                    "timestamp_ns": int(self.keyframes.timestamps[kf_id].item() * 1e9), # Keyframe specific timestamp
                    "keyframe_id": kf_id,
                    "internal_frame_id": slam_frame.frame_idx, # Original frame_idx_counter
                    "pose": kf_pose_matrix_np.tolist(),
                    "algorithm": "MASt3R-SLAM"
                }
                self.channel.basic_publish(
                    exchange=MAST3R_SLAM_KEYFRAME_EXCHANGE, routing_key='', body=json.dumps(kf_msg),
                    properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
                )
                print(f"Published Keyframe: id={kf_id}, original_frame_id={slam_frame.frame_idx}")

                # Publish Point Cloud for this new keyframe
                # get_dense_points gives points in world frame.
                points_xyz_world_torch = self.keyframes.get_dense_points(kf_id) # Returns Nx3 tensor or None
                if points_xyz_world_torch is not None and points_xyz_world_torch.numel() > 0:
                    points_list = points_xyz_world_torch.cpu().tolist()
                    pc_msg = {
                        "timestamp_ns": int(self.keyframes.timestamps[kf_id].item() * 1e9),
                        "keyframe_id": kf_id,
                        "format": "list_xyz_world_meters",
                        "points": points_list,
                        "count": len(points_list),
                        "algorithm": "MASt3R-SLAM"
                    }
                    self.channel.basic_publish(
                        exchange=MAST3R_SLAM_POINTCLOUD_EXCHANGE, routing_key='', body=json.dumps(pc_msg),
                        properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
                    )
                    print(f"Published Point Cloud for KF {kf_id}: {len(points_list)} points")

            # Dense Map Notification (Periodically, if keyframes exist)
            # This requires more of the backend/meshing part of MASt3R-SLAM.
            # For now, let's try calling the eval.save_mesh if there are enough keyframes.
            # This is a heavy operation and should not be done too frequently.
            if self.keyframes and len(self.keyframes) > mast3r_global_config.get("meshing_min_keyframes", 5) and \
               self.frame_idx_counter % mast3r_global_config.get("meshing_freq_frames", 30) == 0:
                
                if not os.path.exists(SAVE_MAP_DIR):
                    os.makedirs(SAVE_MAP_DIR, exist_ok=True)
                
                mesh_filename = f"mesh_kf_{len(self.keyframes)}_{frame_ts_ns}.ply"
                mesh_filepath = os.path.join(SAVE_MAP_DIR, mesh_filename)
                
                print(f"Attempting to save mesh to {mesh_filepath}...")
                try:
                    # eval.save_mesh might need a FactorGraph, which we are not running in this simplified wrapper.
                    # It also might need the 'model' which is self.mast3r_model
                    # Let's check the signature: save_mesh(keyframes, model, path, ...)
                    mast3r_eval.save_mesh(
                        self.keyframes, 
                        self.mast3r_model, 
                        mesh_filepath, 
                        local_batch_size=1, # From MASt3R-SLAM example
                        simplify_mesh=True  # From MASt3R-SLAM example
                    )
                    dense_map_info = {
                        "timestamp_ns": frame_ts_ns,
                        "type": "mesh_saved_notification",
                        "path_on_shared_volume": mesh_filepath, # Path inside Docker
                        "keyframe_count": len(self.keyframes),
                        "algorithm": "MASt3R-SLAM"
                    }
                    self.channel.basic_publish(
                        exchange=MAST3R_SLAM_DENSE_MAP_EXCHANGE, routing_key='', body=json.dumps(dense_map_info),
                        properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
                    )
                    print(f"Dense map mesh saved notification published for: {mesh_filepath}")
                except Exception as e:
                    print(f"Failed to save mesh: {e}")
                    # import traceback; traceback.print_exc() # for more detailed error

            # After tracking, update states for backend
            if slam_frame.is_keyframe:
                print(f"[Frontend] New KeyFrame ID: {slam_frame.id}. Adding to backend optimizer queue.")
                self.states.add_global_optimizer_task(slam_frame.id)
            
            # Check for relocalization trigger
            if tracking_lost and mast3r_global_config.reloc.enable and \
               self.keyframes and len(self.keyframes) > mast3r_global_config.reloc.min_keyframes:
                print(f"[Frontend] Tracking lost for frame {self.frame_idx_counter}. Adding relocalization task.")
                self.states.add_reloc_task(slam_frame) # Backend will pick this up
                # The mode will be set to RELOC by the backend when it processes this task 
                # or tracker might also set its own state. For now, let backend manage mode for RELOC.

            self.frame_idx_counter += 1
            ch.basic_ack(delivery_tag=method.delivery_tag)
            processing_time = time.time() - start_time
            # print(f"Frame {self.frame_idx_counter-1} processed in {processing_time:.4f}s. Tracking: {'OK' if tracking_successful else 'LOST'}")

        except Exception as e:
            print(f"Error processing image message: {e}")
            import traceback
            traceback.print_exc()
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    def on_restart_message(self, ch, method, properties, body):
        try:
            msg = json.loads(body)
            if msg.get("type") == "restart":
                print("Restart command received for MASt3R-SLAM.")
                if self.slam_system_initialized:
                    if self.tracker:
                        # self.tracker.reset() # MASt3R-SLAM's FrameTracker does not have an explicit public reset()
                        # Re-initialize tracker for a full reset
                        print("Re-initializing MASt3R-SLAM FrameTracker for reset...")
                        self.tracker = FrameTracker(self.mast3r_model, self.keyframes, self.device, config=mast3r_global_config)
                    
                    if self.keyframes:
                        print("Re-initializing SharedKeyframes for reset...")
                        self.keyframes = SharedKeyframes(manager=self.mp_manager, h=self.img_h, w=self.img_w, device=self.device)
                        if self.K_torch is not None:
                             self.keyframes.set_intrinsics(self.K_torch)
                        if self.tracker: # Update tracker's keyframes reference
                             self.tracker.keyframes = self.keyframes
                    
                    if self.states:
                        print("Resetting SharedStates for reset...")
                        self.states.reset() # SharedStates has a reset method

                    self.frame_idx_counter = 0
                    self.last_image_timestamp_ns = 0
                    print("MASt3R-SLAM state counters reset.")
                    # Signal backend to re-evaluate its state if needed, though clearing tasks in self.states.reset() helps
                    # The backend loop naturally picks up empty task queues.
                else:
                    print("SLAM system not initialized, attempting to reload for restart.")
                    self._load_slam_system() 
            else:
                print(f"Unknown message on restart queue: {msg}")
            
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f"Error processing restart message: {e}")
            import traceback; traceback.print_exc()
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def run_forever(self):
        if not self.slam_system_initialized:
            print("MASt3R-SLAM system failed to initialize. Consumer will not start.")
            # Optionally, implement a retry mechanism for _load_slam_system or exit.
            return

        print("Starting MASt3R-SLAM consumer...")
        self.channel.basic_consume(
            queue=self.image_queue_name,
            on_message_callback=self.on_image_message
        )
        self.channel.basic_consume(
            queue=self.restart_queue_name,
            on_message_callback=self.on_restart_message
        )

        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("MASt3R-SLAM consumer stopped by user.")
        except Exception as e:
            print(f"MASt3R-SLAM consumer failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Shutting down MASt3R-SLAM processor...")
            if self.states:
                print("Signaling backend process to terminate...")
                self.states.set_mode(MASt3RMode.TERMINATED)
            
            if self.backend_process and self.backend_process.is_alive():
                print("Waiting for backend process to join...")
                self.backend_process.join(timeout=10) # Wait up to 10 seconds
                if self.backend_process.is_alive():
                    print("Backend process did not terminate gracefully, attempting to kill.")
                    self.backend_process.terminate() # Force kill if it doesn't stop
                    self.backend_process.join(timeout=2) # Wait for terminate to take effect
            
            if self.mp_manager:
                print("Shutting down multiprocessing manager...")
                try:
                    # Check if manager process is alive before shutting down
                    # Accessing _process might be using internal API, but common for robust shutdown
                    if hasattr(self.mp_manager, '_process') and self.mp_manager._process.is_alive():
                        self.mp_manager.shutdown()
                    else:
                        print("Multiprocessing manager process already stopped or not applicable.")
                except Exception as e_mgr_shutdown:
                    print(f"Error shutting down multiprocessing manager: {e_mgr_shutdown}")

            if self.connection and self.connection.is_open:
                self.connection.close()
            print("MASt3R-SLAM resources released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MASt3R-SLAM RabbitMQ Processor")
    parser.add_argument(
        "--config", type=str, default="config/base.yaml",
        help="Path to MASt3R-SLAM config file (e.g., /app/config/base.yaml in Docker)."
    )
    parser.add_argument(
        "--calib", type=str, default="config/intrinsics.yaml",
        help="Path to camera intrinsics YAML (e.g., /app/config/intrinsics.yaml in Docker)."
    )
    args = parser.parse_args()

    # Basic check for config files existence (paths are relative to WORKDIR /app in Docker)
    if not os.path.exists(args.config):
        print(f"CRITICAL: MASt3R-SLAM main config file '{args.config}' not found. Exiting.")
        sys.exit(1)
    if args.calib and not os.path.exists(args.calib):
        # MASt3R-SLAM might be configured to run without a separate calib file if use_calib=false
        print(f"WARNING: Specified calibration file '{args.calib}' not found. MASt3R-SLAM will proceed based on 'use_calib' in main config.")


    processor = MASt3RSLAMProcessor(config_path=args.config, calib_path=args.calib)
    processor.run_forever() 