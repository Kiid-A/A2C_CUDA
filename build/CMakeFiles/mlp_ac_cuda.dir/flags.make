# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

# compile CUDA with /usr/local/cuda-12.8/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_DEFINES = -Dmlp_ac_cuda_EXPORTS

CUDA_INCLUDES = --options-file CMakeFiles/mlp_ac_cuda.dir/includes_CUDA.rsp

CUDA_FLAGS = -std=c++14 "--generate-code=arch=compute_86,code=[compute_86,sm_86]" -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden

CXX_DEFINES = -Dmlp_ac_cuda_EXPORTS

CXX_INCLUDES = -isystem /usr/include/python3.10 -isystem /usr/local/cuda-12.8/targets/x86_64-linux/include

CXX_FLAGS = -std=gnu++14 -fPIC -fvisibility=hidden -flto -fno-fat-lto-objects

