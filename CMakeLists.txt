cmake_minimum_required(VERSION 3.17)
project(MLP_AC_PYBIND LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDAToolkit_ROOT "/usr/local/cuda-12.8")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
set(Python3_EXECUTABLE /usr/bin/python3.10)

set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(mlp_ac_cuda_obj OBJECT src/lib/mlp_ac.cu)
set_target_properties(mlp_ac_cuda_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(mlp_ac_cuda src/lib/mlp_ac_pybind.cpp $<TARGET_OBJECTS:mlp_ac_cuda_obj>)
target_link_libraries(mlp_ac_cuda PRIVATE CUDA::cudart)