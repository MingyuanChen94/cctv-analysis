# Save as scripts/setup_gpu.sh
#!/bin/bash

echo "=== GPU Setup Script ==="

# Check if NVIDIA GPU is present
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU driver not found!"
    echo "Please install NVIDIA drivers from: https://www.nvidia.com/download/index.aspx"
    exit 1
fi

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "CUDA toolkit not found!"
    echo "Please install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Create new conda environment
echo "Creating new conda environment with GPU support..."
conda create -n cctv_gpu python=3.8 -y
conda activate cctv_gpu

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other required packages
echo "Installing other required packages..."
pip install -r requirements.txt

echo "Setup complete! Please run the GPU check script to verify installation."