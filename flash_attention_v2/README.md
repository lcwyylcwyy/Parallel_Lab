# Flash Attention V2 Implementation

This directory contains a CUDA-based implementation of Flash Attention V2 using PyTorch.

## Building and Running

The code uses PyTorch's JIT compilation to build CUDA kernels on-the-fly. To run the benchmark:

```bash
python bench_test.py
```

## CUDA Compatibility Notes

### EmptyKernel Compilation Error

If you encounter the following error when compiling with CUDA 12.0 or later:

```
/usr/local/cuda/targets/x86_64-linux/include/cub/util_device.cuh:299:35: error: address of overloaded function 'EmptyKernel' does not match required type 'void ()'
```

This is due to a compatibility issue between the CUB library (part of CUDA Toolkit 12.x and later) and C++17/C++20 standards. The `EmptyKernel` template function in CUB has issues with function pointer type deduction in newer C++ standards when the compiler defaults to C++17 or later.

**Solution**: The code has been updated to use C++14 standard by adding `-std=c++14` to the CUDA compilation flags in `bench_test.py`. This ensures compatibility with CUDA 12.x and later versions by avoiding the C++17/C++20 standard that triggers the issue.

## Requirements

- PyTorch >= 2.5.1
- CUDA Toolkit 12.0 or later (the fix is specifically for CUDA 12.x compatibility)
- A CUDA-capable GPU (compute capability 8.0 or higher for Ampere architecture)

See `requirements.txt` for detailed Python dependencies.
