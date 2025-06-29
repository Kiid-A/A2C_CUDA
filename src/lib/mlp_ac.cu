#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include "mlp_ac.h"
#include <cublas_v2.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

#define WARP_SIZE 32
#define BLOCK_DIM 256
#define TILE_SIZE 16

namespace cg = cooperative_groups;

// ====================== 核心模块：线性层操作 ======================
__global__ void linear_forward_kernel(
    const float *__restrict__ input, const float *__restrict__ weights,
    const float *__restrict__ bias, float *__restrict__ output, 
    int input_dim, int output_dim, int batch_size) 
{
    extern __shared__ float shared_mem[];
    float *sh_input = shared_mem;
    float *sh_weights = &shared_mem[TILE_SIZE * TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= output_dim) return;

    float sum = 0.0f;

    for (int tile = 0; tile < CEIL_DIV(input_dim, TILE_SIZE); tile++) {
        int input_col = tile * TILE_SIZE + threadIdx.x;
        int weight_row = tile * TILE_SIZE + threadIdx.y;

        if (input_col < input_dim) {
            sh_input[threadIdx.y * TILE_SIZE + threadIdx.x] = 
                input[row * input_dim + input_col];
        } else {
            sh_input[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        if (weight_row < input_dim) {
            sh_weights[threadIdx.x * TILE_SIZE + threadIdx.y] = 
                weights[col * input_dim + weight_row];
        } else {
            sh_weights[threadIdx.x * TILE_SIZE + threadIdx.y] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sh_input[threadIdx.y * TILE_SIZE + k] *
                   sh_weights[threadIdx.x * TILE_SIZE + k];
        }

        __syncthreads();
    }

    output[row * output_dim + col] = sum + bias[col];
}

void linear_forward(
    const float *d_input, const float *d_weights, const float *d_bias,
    float *d_output, int input_dim, int output_dim, int batch_size,
    cudaStream_t stream = 0)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(CEIL_DIV(output_dim, TILE_SIZE), CEIL_DIV(batch_size, TILE_SIZE));
    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    
    linear_forward_kernel<<<grid, block, shared_mem_size, stream>>>(
        d_input, d_weights, d_bias, d_output, input_dim, output_dim, batch_size);
}

// ====================== 核心模块：激活函数 ======================
__global__ void relu_forward_kernel(
    const float *__restrict__ input, 
    float *__restrict__ output, 
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

void relu_forward(
    const float *d_input, 
    float *d_output, 
    int size,
    cudaStream_t stream = 0)
{
    int blockSize = 256;
    int gridSize = CEIL_DIV(size, blockSize);
    relu_forward_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, size);
}

// ====================== 核心模块：梯度计算 ======================
__global__ void linear_weight_grad_kernel(
    const float *__restrict__ input,
    const float *__restrict__ grad_output,
    float *__restrict__ grad_weights, 
    int input_dim, int output_dim, int batch_size)
{
    extern __shared__ float sh_grad[];
    int i = blockIdx.x; // 输出维度索引
    int j = blockIdx.y; // 输入维度索引
    int tid = threadIdx.x;

    if (i >= output_dim || j >= input_dim) return;

    float sum = 0.0f;

    for (int b = tid; b < batch_size; b += blockDim.x) {
        sum += input[b * input_dim + j] * grad_output[b * output_dim + i];
    }

    sh_grad[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_grad[tid] += sh_grad[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&grad_weights[i * input_dim + j], sh_grad[0]);
    }
}

void linear_weight_grad(
    const float *d_input, const float *d_grad_output,
    float *d_grad_weights, int input_dim, int output_dim, int batch_size,
    cudaStream_t stream = 0)
{
    dim3 block(BLOCK_DIM);
    dim3 grid(output_dim, input_dim);
    size_t shared_mem_size = BLOCK_DIM * sizeof(float);
    
    linear_weight_grad_kernel<<<grid, block, shared_mem_size, stream>>>(
        d_input, d_grad_output, d_grad_weights, 
        input_dim, output_dim, batch_size);
}

__global__ void linear_bias_grad_kernel(
    const float *__restrict__ grad_output,
    float *__restrict__ grad_bias, 
    int output_dim, int batch_size)
{
    extern __shared__ float sh_grad[];
    int i = blockIdx.x;
    int tid = threadIdx.x;

    if (i >= output_dim) return;

    float sum = 0.0f;

    for (int b = tid; b < batch_size; b += blockDim.x) {
        sum += grad_output[b * output_dim + i];
    }

    sh_grad[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_grad[tid] += sh_grad[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&grad_bias[i], sh_grad[0]);
    }
}

void linear_bias_grad(
    const float *d_grad_output,
    float *d_grad_bias, 
    int output_dim, int batch_size,
    cudaStream_t stream = 0)
{
    dim3 block(BLOCK_DIM);
    dim3 grid(output_dim);
    size_t shared_mem_size = BLOCK_DIM * sizeof(float);
    
    linear_bias_grad_kernel<<<grid, block, shared_mem_size, stream>>>(
        d_grad_output, d_grad_bias, output_dim, batch_size);
}

__global__ void linear_input_grad_kernel(
    const float *__restrict__ grad_output, 
    const float *__restrict__ weights,
    float *__restrict__ grad_input, 
    int in_dim, int out_dim, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_dim)
        return;

    int b = idx / in_dim;
    int j = idx % in_dim;
    float sum = 0.0f;

    for (int i = 0; i < out_dim; ++i) {
        sum += grad_output[b * out_dim + i] * weights[i * in_dim + j];
    }

    grad_input[idx] = sum;
}

void linear_input_grad(
    const float *d_grad_output, const float *d_weights,
    float *d_grad_input, int in_dim, int out_dim, int batch_size,
    cudaStream_t stream = 0)
{
    int blockSize = 256;
    int gridSize = CEIL_DIV(batch_size * in_dim, blockSize);
    linear_input_grad_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_grad_output, d_weights, d_grad_input, in_dim, out_dim, batch_size);
}

__global__ void relu_backward_kernel(
    const float *__restrict__ grad_output, 
    const float *__restrict__ input,
    float *__restrict__ grad_input, 
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx] * (input[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

void relu_backward(
    const float *d_grad_output, const float *d_input,
    float *d_grad_input, int size,
    cudaStream_t stream = 0)
{
    int blockSize = 256;
    int gridSize = CEIL_DIV(size, blockSize);
    relu_backward_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_grad_output, d_input, d_grad_input, size);
}

// ====================== 工具函数 ======================
void zero_init(float *d_ptr, size_t size) {
    CHECK_CUDA(cudaMemset(d_ptr, 0, size * sizeof(float)));
}

// ====================== Actor-Critic 网络实现 ======================
extern "C" {

// 前向传播函数（返回中间结果）
void cuda_forward(
    const float *input, int batch_size, int input_dim, int hidden_dim,
    int actor_output_dim, int critic_output_dim,
    const float *actor_fc1_w, const float *actor_fc1_b,
    const float *actor_fc2_w, const float *actor_fc2_b,
    const float *actor_head_w, const float *actor_head_b,
    const float *critic_fc1_w, const float *critic_fc1_b,
    const float *critic_fc2_w, const float *critic_fc2_b,
    const float *critic_head_w, const float *critic_head_b,
    float *actor_output, float *critic_output,
    // 输出中间结果指针
    float **d_actor_linear_fc1, float **d_actor_linear_fc2,
    float **d_critic_linear_fc1, float **d_critic_linear_fc2) 
{
    // 分配所有 device 内存
    float *d_input;
    float *d_actor_fc1_w, *d_actor_fc1_b, *d_actor_fc2_w, *d_actor_fc2_b;
    float *d_actor_head_w, *d_actor_head_b;
    float *d_critic_fc1_w, *d_critic_fc1_b, *d_critic_fc2_w, *d_critic_fc2_b;
    float *d_critic_head_w, *d_critic_head_b;

    float *d_actor_hidden, *d_actor_fc2_output, *d_actor_out;
    float *d_critic_hidden, *d_critic_fc2_output, *d_critic_out;

    // 分配中间结果内存
    CHECK_CUDA(cudaMalloc(d_actor_linear_fc1, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(d_actor_linear_fc2, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(d_critic_linear_fc1, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(d_critic_linear_fc2, batch_size * hidden_dim * sizeof(float)));

    // 分配其他内存
    CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc1_w, input_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc1_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc2_w, hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc2_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_head_w, hidden_dim * actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_head_b, actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc1_w, input_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc1_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc2_w, hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc2_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_head_w, hidden_dim * 1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_head_b, 1 * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_actor_hidden, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc2_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_out, batch_size * actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_hidden, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc2_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_out, batch_size * critic_output_dim * sizeof(float)));

    // 拷贝参数到 device
    CHECK_CUDA(cudaMemcpy(d_input, input, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_fc1_w, actor_fc1_w, input_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_fc1_b, actor_fc1_b, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_fc2_w, actor_fc2_w, hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_fc2_b, actor_fc2_b, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_head_w, actor_head_w, hidden_dim * actor_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_head_b, actor_head_b, actor_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_fc1_w, critic_fc1_w, input_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_fc1_b, critic_fc1_b, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_fc2_w, critic_fc2_w,  hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_fc2_b, critic_fc2_b,  hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_head_w, critic_head_w,  hidden_dim * critic_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_head_b, critic_head_b, critic_output_dim * sizeof(float), cudaMemcpyHostToDevice));

    // 创建CUDA流
    cudaStream_t actor_stream, critic_stream;
    CHECK_CUDA(cudaStreamCreate(&actor_stream));
    CHECK_CUDA(cudaStreamCreate(&critic_stream));

    // Actor路径
    // FC1: Linear + ReLU
    linear_forward(
        d_input, d_actor_fc1_w, d_actor_fc1_b, 
        *d_actor_linear_fc1, input_dim, hidden_dim, batch_size, actor_stream);
    relu_forward(
        *d_actor_linear_fc1, d_actor_hidden, 
        batch_size * hidden_dim, actor_stream);

    // FC2: Linear + ReLU
    linear_forward(
        d_actor_hidden, d_actor_fc2_w, d_actor_fc2_b, 
        *d_actor_linear_fc2, hidden_dim, hidden_dim, batch_size, actor_stream);
    relu_forward(
        *d_actor_linear_fc2, d_actor_fc2_output, 
        batch_size * hidden_dim, actor_stream);

    // Head: Linear
    linear_forward(
        d_actor_fc2_output, d_actor_head_w, d_actor_head_b, 
        d_actor_out, hidden_dim, actor_output_dim, batch_size, actor_stream);

    // Critic路径
    // FC1: Linear + ReLU
    linear_forward(
        d_input, d_critic_fc1_w, d_critic_fc1_b, 
        *d_critic_linear_fc1, input_dim, hidden_dim, batch_size, critic_stream);
    relu_forward(
        *d_critic_linear_fc1, d_critic_hidden, 
        batch_size * hidden_dim, critic_stream);

    // FC2: Linear + ReLU
    linear_forward(
        d_critic_hidden, d_critic_fc2_w, d_critic_fc2_b, 
        *d_critic_linear_fc2, hidden_dim, hidden_dim, batch_size, critic_stream);
    relu_forward(
        *d_critic_linear_fc2, d_critic_fc2_output, 
        batch_size * hidden_dim, critic_stream);

    // Head: Linear
    linear_forward(
        d_critic_fc2_output, d_critic_head_w, d_critic_head_b, 
        d_critic_out, hidden_dim, critic_output_dim, batch_size, critic_stream);

    // 同步流
    CHECK_CUDA(cudaStreamSynchronize(actor_stream));
    CHECK_CUDA(cudaStreamSynchronize(critic_stream));

    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(actor_output, d_actor_out, batch_size * actor_output_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(critic_output, d_critic_out, batch_size * critic_output_dim * sizeof(float), cudaMemcpyDeviceToHost));

    // 释放临时设备内存（不释放中间结果）
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_actor_fc1_w));
    CHECK_CUDA(cudaFree(d_actor_fc1_b));
    CHECK_CUDA(cudaFree(d_actor_fc2_w));
    CHECK_CUDA(cudaFree(d_actor_fc2_b));
    CHECK_CUDA(cudaFree(d_actor_head_w));
    CHECK_CUDA(cudaFree(d_actor_head_b));
    CHECK_CUDA(cudaFree(d_critic_fc1_w));
    CHECK_CUDA(cudaFree(d_critic_fc1_b));
    CHECK_CUDA(cudaFree(d_critic_fc2_w));
    CHECK_CUDA(cudaFree(d_critic_fc2_b));
    CHECK_CUDA(cudaFree(d_critic_head_w));
    CHECK_CUDA(cudaFree(d_critic_head_b));
    CHECK_CUDA(cudaFree(d_actor_hidden));
    CHECK_CUDA(cudaFree(d_actor_fc2_output));
    CHECK_CUDA(cudaFree(d_actor_out));
    CHECK_CUDA(cudaFree(d_critic_hidden));
    CHECK_CUDA(cudaFree(d_critic_fc2_output));
    CHECK_CUDA(cudaFree(d_critic_out));

    CHECK_CUDA(cudaStreamDestroy(actor_stream));
    CHECK_CUDA(cudaStreamDestroy(critic_stream));
}

// 反向传播函数（使用预存的中间结果）
void cuda_backward(
    const float *input, int batch_size, int input_dim, int hidden_dim, 
    int actor_output_dim, int critic_output_dim,
    const float *actor_fc1_w, const float *actor_fc1_b,
    const float *actor_fc2_w, const float *actor_fc2_b,
    const float *actor_head_w, const float *actor_head_b,
    const float *critic_fc1_w, const float *critic_fc1_b,
    const float *critic_fc2_w, const float *critic_fc2_b,
    const float *critic_head_w, const float *critic_head_b,
    const float *grad_actor_output, const float *grad_critic_output,
    // 中间结果（前向传播保存）
    const float *d_actor_linear_fc1, const float *d_actor_linear_fc2,
    const float *d_critic_linear_fc1, const float *d_critic_linear_fc2,
    // 梯度输出
    float *grad_actor_fc1_w, float *grad_actor_fc1_b,
    float *grad_actor_fc2_w, float *grad_actor_fc2_b,
    float *grad_actor_head_w, float *grad_actor_head_b,
    float *grad_critic_fc1_w, float *grad_critic_fc1_b,
    float *grad_critic_fc2_w, float *grad_critic_fc2_b,
    float *grad_critic_head_w, float *grad_critic_head_b)
{
    // 1. 分配所有 device 内存
    float *d_input;
    float *d_actor_fc1_w, *d_actor_fc1_b, *d_actor_fc2_w, *d_actor_fc2_b;
    float *d_actor_head_w, *d_actor_head_b;
    float *d_critic_fc1_w, *d_critic_fc1_b, *d_critic_fc2_w, *d_critic_fc2_b;
    float *d_critic_head_w, *d_critic_head_b;

    float *d_actor_hidden, *d_actor_fc2_output;
    float *d_actor_out, *d_critic_hidden, *d_critic_fc2_output, *d_critic_out;

    float *d_grad_actor_output, *d_grad_critic_output;
    float *d_grad_actor_fc1_w, *d_grad_actor_fc1_b;
    float *d_grad_actor_fc2_w, *d_grad_actor_fc2_b;
    float *d_grad_actor_head_w, *d_grad_actor_head_b;
    float *d_grad_critic_fc1_w, *d_grad_critic_fc1_b;
    float *d_grad_critic_fc2_w, *d_grad_critic_fc2_b;
    float *d_grad_critic_head_w, *d_grad_critic_head_b;

    // 中间梯度分配
    float *d_grad_actor_fc2_output, *d_grad_actor_linear_fc2;
    float *d_grad_actor_hidden, *d_grad_actor_linear_fc1;
    float *d_grad_critic_fc2_output, *d_grad_critic_linear_fc2;
    float *d_grad_critic_hidden, *d_grad_critic_linear_fc1;

    // 分配 device 内存
    CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc1_w, input_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc1_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc2_w, hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc2_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_head_w, actor_output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_head_b, actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc1_w, input_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc1_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc2_w, hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc2_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_head_w, critic_output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_head_b, critic_output_dim * sizeof(float)));

    // 中间结果分配（使用预分配的内存）
    CHECK_CUDA(cudaMalloc(&d_actor_hidden, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc2_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_out, batch_size * actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_hidden, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc2_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_out, batch_size * critic_output_dim * sizeof(float)));

    // 梯度分配
    CHECK_CUDA(cudaMalloc(&d_grad_actor_output, batch_size * actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_output, batch_size * critic_output_dim * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_grad_actor_fc1_w, input_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_fc1_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_fc2_w, hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_fc2_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_head_w, actor_output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_head_b, actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_fc1_w, input_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_fc1_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_fc2_w, hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_fc2_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_head_w, critic_output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_head_b, critic_output_dim * sizeof(float)));

    // 中间梯度分配
    CHECK_CUDA(cudaMalloc(&d_grad_actor_fc2_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_linear_fc2, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_hidden, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_linear_fc1, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_fc2_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_linear_fc2, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_hidden, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_linear_fc1, batch_size * hidden_dim * sizeof(float)));

    // 2. 拷贝输入、参数、梯度到 device
    CHECK_CUDA(cudaMemcpy(d_input, input, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_fc1_w, actor_fc1_w, input_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_fc1_b, actor_fc1_b, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_fc2_w, actor_fc2_w, hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_fc2_b, actor_fc2_b, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_head_w, actor_head_w, hidden_dim * actor_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_head_b, actor_head_b, actor_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_fc1_w, critic_fc1_w, input_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_fc1_b, critic_fc1_b, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_fc2_w, critic_fc2_w, hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_fc2_b, critic_fc2_b, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_head_w, critic_head_w, hidden_dim * critic_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_head_b, critic_head_b, critic_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad_actor_output, grad_actor_output, batch_size * actor_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad_critic_output, grad_critic_output, batch_size * critic_output_dim * sizeof(float), cudaMemcpyHostToDevice));

    // 初始化梯度为0
    zero_init(d_grad_actor_fc1_w, input_dim * hidden_dim);
    zero_init(d_grad_actor_fc1_b, hidden_dim);
    zero_init(d_grad_actor_fc2_w, hidden_dim * hidden_dim);
    zero_init(d_grad_actor_fc2_b, hidden_dim);
    zero_init(d_grad_actor_head_w, actor_output_dim * hidden_dim);
    zero_init(d_grad_actor_head_b, actor_output_dim);
    zero_init(d_grad_critic_fc1_w, input_dim * hidden_dim);
    zero_init(d_grad_critic_fc1_b, hidden_dim);
    zero_init(d_grad_critic_fc2_w, hidden_dim * hidden_dim);
    zero_init(d_grad_critic_fc2_b, hidden_dim);
    zero_init(d_grad_critic_head_w, critic_output_dim * hidden_dim);
    zero_init(d_grad_critic_head_b, critic_output_dim);

    // 创建CUDA流
    cudaStream_t actor_stream, critic_stream;
    CHECK_CUDA(cudaStreamCreate(&actor_stream));
    CHECK_CUDA(cudaStreamCreate(&critic_stream));

    // 4. 反向传播（使用预存的中间结果）
    // Actor路径
    // Head层: 线性层反向传播
    linear_input_grad(
        d_grad_actor_output, d_actor_head_w, 
        d_grad_actor_fc2_output, hidden_dim, actor_output_dim, batch_size, actor_stream);
    linear_weight_grad(
        d_actor_fc2_output, d_grad_actor_output, 
        d_grad_actor_head_w, hidden_dim, actor_output_dim, batch_size, actor_stream);
    linear_bias_grad(
        d_grad_actor_output, 
        d_grad_actor_head_b, actor_output_dim, batch_size, actor_stream);

    // FC2层: ReLU反向传播 + 线性层反向传播
    relu_backward(
        d_grad_actor_fc2_output, d_actor_linear_fc2, 
        d_grad_actor_linear_fc2, batch_size * hidden_dim, actor_stream);
    linear_input_grad(
        d_grad_actor_linear_fc2, d_actor_fc2_w, 
        d_grad_actor_hidden, hidden_dim, hidden_dim, batch_size, actor_stream);
    linear_weight_grad(
        d_actor_hidden, d_grad_actor_linear_fc2, 
        d_grad_actor_fc2_w, hidden_dim, hidden_dim, batch_size, actor_stream);
    linear_bias_grad(
        d_grad_actor_linear_fc2, 
        d_grad_actor_fc2_b, hidden_dim, batch_size, actor_stream);

    // FC1层: ReLU反向传播 + 线性层反向传播
    relu_backward(
        d_grad_actor_hidden, d_actor_linear_fc1, 
        d_grad_actor_linear_fc1, batch_size * hidden_dim, actor_stream);
    linear_input_grad(
        d_grad_actor_linear_fc1, d_actor_fc1_w, 
        d_grad_actor_fc1_w, input_dim, hidden_dim, batch_size, actor_stream);
    linear_weight_grad(
        d_input, d_grad_actor_linear_fc1, 
        d_grad_actor_fc1_w, input_dim, hidden_dim, batch_size, actor_stream);
    linear_bias_grad(
        d_grad_actor_linear_fc1, 
        d_grad_actor_fc1_b, hidden_dim, batch_size, actor_stream);

    // Critic路径（类似Actor）
    // Head层
    linear_input_grad(
        d_grad_critic_output, d_critic_head_w, 
        d_grad_critic_fc2_output, hidden_dim, critic_output_dim, batch_size, critic_stream);
    linear_weight_grad(
        d_critic_fc2_output, d_grad_critic_output, 
        d_grad_critic_head_w, hidden_dim, critic_output_dim, batch_size, critic_stream);
    linear_bias_grad(
        d_grad_critic_output, 
        d_grad_critic_head_b, critic_output_dim, batch_size, critic_stream);

    // FC2层
    relu_backward(
        d_grad_critic_fc2_output, d_critic_linear_fc2, 
        d_grad_critic_linear_fc2, batch_size * hidden_dim, critic_stream);
    linear_input_grad(
        d_grad_critic_linear_fc2, d_critic_fc2_w, 
        d_grad_critic_hidden, hidden_dim, hidden_dim, batch_size, critic_stream);
    linear_weight_grad(
        d_critic_hidden, d_grad_critic_linear_fc2, 
        d_grad_critic_fc2_w, hidden_dim, hidden_dim, batch_size, critic_stream);
    linear_bias_grad(
        d_grad_critic_linear_fc2, 
        d_grad_critic_fc2_b, hidden_dim, batch_size, critic_stream);

    // FC1层
    relu_backward(
        d_grad_critic_hidden, d_critic_linear_fc1, 
        d_grad_critic_linear_fc1, batch_size * hidden_dim, critic_stream);
    linear_input_grad(
        d_grad_critic_linear_fc1, d_critic_fc1_w, 
        d_grad_critic_fc1_w, input_dim, hidden_dim, batch_size, critic_stream);
    linear_weight_grad(
        d_input, d_grad_critic_linear_fc1, 
        d_grad_critic_fc1_w, input_dim, hidden_dim, batch_size, critic_stream);
    linear_bias_grad(
        d_grad_critic_linear_fc1, 
        d_grad_critic_fc1_b, hidden_dim, batch_size, critic_stream);

    // 等待两个流完成
    CHECK_CUDA(cudaStreamSynchronize(actor_stream));
    CHECK_CUDA(cudaStreamSynchronize(critic_stream));

    // 5. 拷贝所有梯度回主机
    CHECK_CUDA(cudaMemcpy(grad_actor_fc1_w, d_grad_actor_fc1_w, input_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_actor_fc1_b, d_grad_actor_fc1_b, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_actor_fc2_w, d_grad_actor_fc2_w, hidden_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_actor_fc2_b, d_grad_actor_fc2_b, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_actor_head_w, d_grad_actor_head_w, actor_output_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_actor_head_b, d_grad_actor_head_b, actor_output_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_critic_fc1_w, d_grad_critic_fc1_w, input_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_critic_fc1_b, d_grad_critic_fc1_b, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_critic_fc2_w, d_grad_critic_fc2_w, hidden_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_critic_fc2_b, d_grad_critic_fc2_b, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_critic_head_w, d_grad_critic_head_w, critic_output_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_critic_head_b, d_grad_critic_head_b, critic_output_dim * sizeof(float), cudaMemcpyDeviceToHost));

    // 6. 释放所有 device 内存
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_actor_fc1_w));
    CHECK_CUDA(cudaFree(d_actor_fc1_b));
    CHECK_CUDA(cudaFree(d_actor_fc2_w));
    CHECK_CUDA(cudaFree(d_actor_fc2_b));
    CHECK_CUDA(cudaFree(d_actor_head_w));
    CHECK_CUDA(cudaFree(d_actor_head_b));
    CHECK_CUDA(cudaFree(d_critic_fc1_w));
    CHECK_CUDA(cudaFree(d_critic_fc1_b));
    CHECK_CUDA(cudaFree(d_critic_fc2_w));
    CHECK_CUDA(cudaFree(d_critic_fc2_b));
    CHECK_CUDA(cudaFree(d_critic_head_w));
    CHECK_CUDA(cudaFree(d_critic_head_b));
    CHECK_CUDA(cudaFree(d_actor_hidden));
    CHECK_CUDA(cudaFree(d_actor_fc2_output));
    CHECK_CUDA(cudaFree(d_actor_out));
    CHECK_CUDA(cudaFree(d_critic_hidden));
    CHECK_CUDA(cudaFree(d_critic_fc2_output));
    CHECK_CUDA(cudaFree(d_critic_out));
    CHECK_CUDA(cudaFree(d_grad_actor_output));
    CHECK_CUDA(cudaFree(d_grad_critic_output));
    CHECK_CUDA(cudaFree(d_grad_actor_fc1_w));
    CHECK_CUDA(cudaFree(d_grad_actor_fc1_b));
    CHECK_CUDA(cudaFree(d_grad_actor_fc2_w));
    CHECK_CUDA(cudaFree(d_grad_actor_fc2_b));
    CHECK_CUDA(cudaFree(d_grad_actor_head_w));
    CHECK_CUDA(cudaFree(d_grad_actor_head_b));
    CHECK_CUDA(cudaFree(d_grad_critic_fc1_w));
    CHECK_CUDA(cudaFree(d_grad_critic_fc1_b));
    CHECK_CUDA(cudaFree(d_grad_critic_fc2_w));
    CHECK_CUDA(cudaFree(d_grad_critic_fc2_b));
    CHECK_CUDA(cudaFree(d_grad_critic_head_w));
    CHECK_CUDA(cudaFree(d_grad_critic_head_b));
    CHECK_CUDA(cudaFree(d_grad_actor_fc2_output));
    CHECK_CUDA(cudaFree(d_grad_actor_linear_fc2));
    CHECK_CUDA(cudaFree(d_grad_actor_hidden));
    CHECK_CUDA(cudaFree(d_grad_actor_linear_fc1));
    CHECK_CUDA(cudaFree(d_grad_critic_fc2_output));
    CHECK_CUDA(cudaFree(d_grad_critic_linear_fc2));
    CHECK_CUDA(cudaFree(d_grad_critic_hidden));
    CHECK_CUDA(cudaFree(d_grad_critic_linear_fc1));

    CHECK_CUDA(cudaStreamDestroy(actor_stream));
    CHECK_CUDA(cudaStreamDestroy(critic_stream));
}

// 释放中间结果内存
void cuda_free_intermediate(float *d_actor_linear_fc1, float *d_actor_linear_fc2,
                            float *d_critic_linear_fc1, float *d_critic_linear_fc2) {
    CHECK_CUDA(cudaFree(d_actor_linear_fc1));
    CHECK_CUDA(cudaFree(d_actor_linear_fc2));
    CHECK_CUDA(cudaFree(d_critic_linear_fc1));
    CHECK_CUDA(cudaFree(d_critic_linear_fc2));
}

} // extern "C"