#!/bin/zsh
rm -rf build
mkdir build && cd build
cmake .. \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCUDAToolkit_ROOT=/opt/cuda \
    -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
    -DPython3_ROOT_DIR=/usr/lib/python3.13 \
&& make -j 12
cd ..
rm ./src/mlp_ac_cuda.cpython-313-x86_64-linux-gnu.so
cp ./build/mlp_ac_cuda.cpython-313-x86_64-linux-gnu.so ./src

