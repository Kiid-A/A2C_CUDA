rm -rf build
mkdir build && cd build
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DPython3_ROOT_DIR=/usr/lib/python3.10
make
