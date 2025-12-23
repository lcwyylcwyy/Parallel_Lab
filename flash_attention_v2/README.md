# Flash Attention V2 Implementation

This directory contains a CUDA-based implementation of Flash Attention V2 using PyTorch.

## Building and Running

The code uses PyTorch's JIT compilation to build CUDA kernels on-the-fly. To run the benchmark:

```bash
python bench_test.py
```

## CUDA Compatibility Notes

### EmptyKernel Compilation Error

If you encounter the following error when compiling with CUDA 12.x:

```
/usr/local/cuda/targets/x86_64-linux/include/cub/util_device.cuh:299:35: error: address of overloaded function 'EmptyKernel' does not match required type 'void ()'
```

This is due to a compatibility issue between the CUB library (part of CUDA Toolkit) and C++17/C++20 standards. The `EmptyKernel` template function in CUB has issues with function pointer type deduction in newer C++ standards.

**Solution**: The code has been updated to use C++14 standard by adding `-std=c++14` to the CUDA compilation flags in `bench_test.py`. This ensures compatibility with CUDA 12.x and the CUB library.

## Requirements

- PyTorch >= 2.5.1
- CUDA Toolkit (tested with CUDA 12.x)
- A CUDA-capable GPU

See `requirements.txt` for detailed Python dependencies.
