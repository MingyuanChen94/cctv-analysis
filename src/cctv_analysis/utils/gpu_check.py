# Save this as src/cctv_analysis/utils/gpu_check.py
import subprocess
import sys
import torch

def check_gpu_status():
    """
    Comprehensive GPU status check
    Returns: Dict with diagnostic information
    """
    status = {
        "gpu_found": False,
        "cuda_available": False,
        "gpu_info": None,
        "cuda_version": None,
        "torch_cuda_version": None,
        "pytorch_built_with_cuda": False,
        "recommendations": []
    }
    
    # Check if nvidia-smi is available
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
        status["gpu_found"] = True
        status["gpu_info"] = nvidia_smi.decode()
    except:
        status["recommendations"].append(
            "NVIDIA GPU driver not found. Please install NVIDIA drivers from: "
            "https://www.nvidia.com/download/index.aspx"
        )
    
    # Check CUDA availability in PyTorch
    status["cuda_available"] = torch.cuda.is_available()
    
    # Check CUDA version if available
    try:
        nvcc_output = subprocess.check_output("nvcc --version", shell=True)
        for line in nvcc_output.decode().split('\n'):
            if "release" in line:
                status["cuda_version"] = line
                break
    except:
        status["recommendations"].append(
            "CUDA toolkit not found. Please install CUDA toolkit from: "
            "https://developer.nvidia.com/cuda-downloads"
        )
    
    # Check PyTorch CUDA capabilities
    status["torch_cuda_version"] = torch.version.cuda
    status["pytorch_built_with_cuda"] = torch.cuda.is_available()
    
    if not status["pytorch_built_with_cuda"]:
        status["recommendations"].append(
            "PyTorch is not built with CUDA. Please reinstall PyTorch with CUDA support:\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        )
    
    return status

def print_gpu_status():
    """Print formatted GPU status information"""
    status = check_gpu_status()
    
    print("\n=== GPU Status Check ===")
    print(f"\nGPU Found: {status['gpu_found']}")
    print(f"CUDA Available: {status['cuda_available']}")
    
    if status['gpu_info']:
        print("\nGPU Information:")
        print(status['gpu_info'])
    
    if status['cuda_version']:
        print("\nCUDA Version:")
        print(status['cuda_version'])
    
    print(f"\nPyTorch CUDA Version: {status['torch_cuda_version']}")
    print(f"PyTorch Built with CUDA: {status['pytorch_built_with_cuda']}")
    
    if status['recommendations']:
        print("\nRecommendations to enable GPU:")
        for i, rec in enumerate(status['recommendations'], 1):
            print(f"{i}. {rec}")
    
    return status

if __name__ == "__main__":
    print_gpu_status()
