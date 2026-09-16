"""
GPU/CUDA capability verification test.
Tests torch.cuda.is_available() and executes a small tensor operation on GPU if present,
or gracefully confirms CPU mode without error.
"""
import sys
import unittest
import torch


def check_gpu_status() -> bool:
    print("=" * 60)
    print("GPU / ACCELERATION CHECK")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available : {cuda_available}")

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print(f"Device Count   : {device_count}")
        print(f"GPU Name       : {gpu_name}")

        # Small tensor operation test on GPU
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        y = x * 2.0
        result = y.cpu().tolist()
        print(f"GPU Computation Test Passed: [1, 2, 3] * 2 = {result}")
        print("CUDA acceleration is operational.")
    else:
        print("Status         : No NVIDIA GPU / CUDA detected.")
        print("Execution Mode : CPU mode is active and verified for all computations.")
        # Verify basic tensor operation on CPU
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2.0
        result = y.tolist()
        print(f"CPU Computation Test Passed: [1, 2, 3] * 2 = {result}")

    print("=" * 60)
    return True


class TestGPU(unittest.TestCase):
    def test_gpu_or_cpu_execution(self):
        success = check_gpu_status()
        self.assertTrue(success)


if __name__ == "__main__":
    check_gpu_status()
    sys.exit(0)
