# Dockerfile for MASt3R-SLAM
# Prompts:
# - User request to implement MASt3R-SLAM in a Dockerfile within @master_slam.
# - Use CUDA 12.4, NVIDIA Docker base image.
# - Adapt for RabbitMQ integration (details in wrapper script later).

# Use NVIDIA CUDA 12.4 base image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    unzip \
    # Python build dependencies (though we use conda)
    python3-dev \
    # MASt3R-SLAM and its dependencies might need these
    libgl1-mesa-dev \
    libglew-dev \
    libeigen3-dev \
    # OpenCV dependencies often include these
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev \
    # For X11 forwarding if GUI is ever used
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda for Python 3.11
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py311_23.10.0-1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    $CONDA_DIR/bin/conda init bash

# Make conda available in subsequent RUN instructions
SHELL ["/bin/bash", "-c"]
ENV PATH $CONDA_DIR/bin:$PATH

# Create a conda environment for MASt3R-SLAM
RUN conda create -n mast3r-slam python=3.11 -y
ENV CONDA_DEFAULT_ENV mast3r-slam
ENV CONDA_PREFIX $CONDA_DIR/envs/mast3r-slam
ENV PATH $CONDA_PREFIX/bin:$PATH
RUN echo "conda activate mast3r-slam" >> ~/.bashrc

# Activate the conda environment for subsequent RUN, CMD, ENTRYPOINT instructions
RUN conda install -n mast3r-slam pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# Note: Pytorch 2.5.1 specific build for CUDA 12.4 might not be directly on conda main channels yet.
# The closest Pytorch 2.3.1 is available for CUDA 12.1, which is often compatible.
# If Pytorch 2.5.1 for CUDA 12.4 becomes available on conda, update above.
# Or, we might need to install via pip if a wheel is available.
# Let's try with Pytorch 2.3.1 + CUDA 12.1 from conda first as per general availability.
# The user specified pytorch 2.5.1, will adjust to pip if direct conda install fails or is not specific enough.
# Re-evaluating: The README specifically mentions 2.5.1. Let's try to achieve that.
# The command `conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia`
# is for CUDA 12.4. Let's use that directly.
RUN conda install -n mast3r-slam pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Set working directory
WORKDIR /app

# Copy the MASt3R-SLAM code (which is a submodule in the build context)
COPY . .

# Initialize and update submodules within the MASt3R-SLAM directory
# The Docker context is master_slam, and MASt3R-SLAM is at its root.
# This line is removed as it causes issues if master_slam is a submodule itself
# and submodules are assumed to be pre-populated on the host.
# RUN git submodule update --init --recursive

# Install MASt3R-SLAM Python dependencies
# Ensure pip is from the conda env
RUN pip install --no-cache-dir -e thirdparty/mast3r
RUN pip install --no-cache-dir -e thirdparty/in3d
RUN pip install --no-cache-dir --no-build-isolation -e .
RUN pip install --no-cache-dir torchcodec==0.1

# Install other Python dependencies for the wrapper script
RUN pip install --no-cache-dir pika opencv-python-headless PyYAML numpy Pillow

# Download MASt3R checkpoints
RUN mkdir -p checkpoints && \
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/ && \
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/ && \
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/

# User setup ( mimicking existing Dockerfile structure )
ARG USERNAME=docker_user
ARG UID=1000
ARG GID=1000
ARG HOME=/home/$USERNAME

RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Grant permissions to the new user for the app directory
RUN chown -R $USERNAME:$GID /app
RUN chown -R $USERNAME:$GID $CONDA_DIR

USER $USERNAME
WORKDIR /app

# Set environment variables for the user
ENV PYTHONUNBUFFERED=1

# CMD will be replaced by the actual wrapper script later
CMD ["python3", "run_mast3r_slam.py"] 