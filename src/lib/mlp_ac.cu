#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include "mlp_ac.h"
#include <stdio.h>

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

// namespace cg = cooperative_groups;

// 优化后的前向传播核函数 (支持混合精度)
__global__ void optimized_linear_forward_kernel(
    const float *__restrict__ input, const float *__restrict__ weights,
    const float *__restrict__ bias, float *__restrict__ output, int input_dim,
    int output_dim, int batch_size) {
  extern __shared__ float shared_mem[];
  float *sh_input = shared_mem;
  float *sh_weights = &shared_mem[TILE_SIZE * TILE_SIZE];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= batch_size || col >= output_dim) return; // 修正

  float sum = 0.0f;

  for (int tile = 0; tile < (input_dim + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    int input_col = tile * TILE_SIZE + threadIdx.x;
    int weight_row = tile * TILE_SIZE + threadIdx.y;

    // 加载输入瓦片
    if (row < batch_size && input_col < input_dim) {
      sh_input[threadIdx.y * TILE_SIZE + threadIdx.x] =
          input[row * input_dim + input_col];
    } else {
      sh_input[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    }

    // 加载权重瓦片
    if (col < output_dim && weight_row < input_dim) {
      sh_weights[threadIdx.y * TILE_SIZE + threadIdx.x] =
          weights[col * input_dim + weight_row];
    } else {
      sh_weights[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // 计算部分和
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += sh_input[threadIdx.y * TILE_SIZE + k] *
             sh_weights[k * TILE_SIZE + threadIdx.x];
    }

    __syncthreads();
  }

  if (row < batch_size && col < output_dim) {
    // 添加偏置并存储结果
    output[row * output_dim + col] = sum + bias[col];
  }
}

// 优化的激活前向传播 (ReLU)
__global__ void fc_relu_forward_kernel(const float *__restrict__ input,
                                       const float *__restrict__ weights,
                                       const float *__restrict__ bias,
                                       float *__restrict__ output,
                                       int input_dim, int output_dim,
                                       int batch_size) {

  // 与优化后的线性前向传播相同，但最后应用ReLU
  extern __shared__ float shared_mem[];
  float *sh_input = shared_mem;
  float *sh_weights = &shared_mem[TILE_SIZE * TILE_SIZE];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= output_dim || col >= input_dim) return;

  float sum = 0.0f;

  for (int tile = 0; tile < (input_dim + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    int input_col = tile * TILE_SIZE + threadIdx.x;
    int weight_row = tile * TILE_SIZE + threadIdx.y;

    if (row < batch_size && input_col < input_dim) {
      sh_input[threadIdx.y * TILE_SIZE + threadIdx.x] =
          input[row * input_dim + input_col];
    } else {
      sh_input[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    }

    if (col < output_dim && weight_row < input_dim) {
      sh_weights[threadIdx.y * TILE_SIZE + threadIdx.x] =
          weights[col * input_dim + weight_row];
    } else {
      sh_weights[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; k++) {
      sum += sh_input[threadIdx.y * TILE_SIZE + k] *
             sh_weights[k * TILE_SIZE + threadIdx.x];
    }

    __syncthreads();
  }

  if (row < batch_size && col < output_dim) {
    output[row * output_dim + col] = fmaxf(sum + bias[col], 0.0f);
  }
}

// 优化的权重梯度计算 (使用共享内存归约)
__global__ void
optimized_weight_grad_kernel(const float *__restrict__ input,
                             const float *__restrict__ grad_output,
                             float *__restrict__ grad_weights, int input_dim,
                             int output_dim, int batch_size) {

  extern __shared__ float shared_mem[];
  float *sh_grad = shared_mem;

  int i = blockIdx.x; // 输出维度索引
  int j = blockIdx.y; // 输入维度索引
  int tid = threadIdx.x;

  if (i >= output_dim || j >= input_dim) return;

  float sum = 0.0f;

  // 每个线程处理batch_size/BLOCK_DIM个样本
  for (int b = tid; b < batch_size; b += blockDim.x) {
    sum += input[b * input_dim + j] * grad_output[b * output_dim + i];
  }

  sh_grad[tid] = sum;
  __syncthreads();

  // 块内归约
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sh_grad[tid] += sh_grad[tid + stride];
    }
    __syncthreads();
  }

  // 使用原子操作更新全局内存
  if (tid == 0) {
    atomicAdd(&grad_weights[i * input_dim + j], sh_grad[0]);
  }
}

// 优化的偏置梯度计算
__global__ void
optimized_bias_grad_kernel(const float *__restrict__ grad_output,
                           float *__restrict__ grad_bias, int output_dim,
                           int batch_size) {

  extern __shared__ float shared_mem[];
  float *sh_grad = shared_mem;

  int i = blockIdx.x; // 输出维度索引
  int tid = threadIdx.x;

  if (i >= output_dim) return;

  float sum = 0.0f;

  // 每个线程处理batch_size/BLOCK_DIM个样本
  for (int b = tid; b < batch_size; b += blockDim.x) {
    sum += grad_output[b * output_dim + i];
  }

  sh_grad[tid] = sum;
  __syncthreads();

  // 块内归约
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sh_grad[tid] += sh_grad[tid + stride];
    }
    __syncthreads();
  }

  // 使用原子操作更新全局内存
  if (tid == 0) {
    atomicAdd(&grad_bias[i], sh_grad[0]);
  }
}

// 计算隐藏层梯度 (用于反向传播)
__global__ void compute_hidden_grad_kernel(
    const float *__restrict__ grad_output, const float *__restrict__ weights,
    float *__restrict__ grad_hidden, int in_dim, int out_dim, int batch_size) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * in_dim)
    return;

  int b = idx / in_dim;
  int j = idx % in_dim;
  float sum = 0.0f;

  for (int i = 0; i < out_dim; ++i) {
    sum += grad_output[b * out_dim + i] * weights[i * in_dim + j];
  }

  grad_hidden[idx] = sum;
}

// 计算带激活的隐藏层梯度 (ReLU)
__global__ void compute_relu_hidden_grad_kernel(
    const float *__restrict__ grad_output, const float *__restrict__ weights,
    const float *__restrict__ hidden, // 前向传播的隐藏层输出
    float *__restrict__ grad_hidden, int in_dim, int out_dim, int batch_size) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * in_dim) return;

  int b = idx / in_dim;
  int j = idx % in_dim;
  float sum = 0.0f;

  // 计算权重梯度部分
  for (int i = 0; i < out_dim; ++i) {
    sum += grad_output[b * out_dim + i] * weights[i * in_dim + j];
  }

  // 应用ReLU导数 (x > 0 ? 1 : 0)
  float relu_derivative = hidden[b * in_dim + j] > 0.0f ? 1.0f : 0.0f;

  grad_hidden[idx] = sum * relu_derivative;
}


extern "C" {
void cuda_forward(const float *input, int batch_size, int input_dim, int hidden_dim,
                  int actor_output_dim, int critic_output_dim,
                  const float *actor_fc1_w, const float *actor_fc1_b,
                  const float *actor_fc2_w, const float *actor_fc2_b,
                  const float *actor_head_w, const float *actor_head_b,
                  const float *critic_fc1_w, const float *critic_fc1_b,
                  const float *critic_fc2_w, const float *critic_fc2_b,
                  const float *critic_head_w, const float *critic_head_b,
                  float *actor_output, float *critic_output) {
  // 分配所有 device 内存
  float *d_input;
  float *d_actor_fc1_w, *d_actor_fc1_b, *d_actor_fc2_w, *d_actor_fc2_b;
  float *d_actor_head_w, *d_actor_head_b;
  float *d_critic_fc1_w, *d_critic_fc1_b, *d_critic_fc2_w, *d_critic_fc2_b;
  float *d_critic_head_w, *d_critic_head_b;

  float *d_actor_hidden, *d_actor_fc2_output, *d_actor_out;
  float *d_critic_hidden, *d_critic_fc2_output, *d_critic_out;

  // 分配 device 内存
  CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_actor_fc1_w, input_dim * hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_actor_fc1_b, hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_actor_fc2_w, actor_output_dim * hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_actor_fc2_b, actor_output_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_actor_head_w, hidden_dim * actor_output_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_actor_head_b, actor_output_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_critic_fc1_w, input_dim * hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_critic_fc1_b, hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_critic_fc2_w, hidden_dim * hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_critic_fc2_b, hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_critic_head_w, hidden_dim * 1 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_critic_head_b, 1 * sizeof(float)));

  CHECK_CUDA(cudaMalloc(&d_actor_hidden, batch_size * hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_actor_fc2_output, batch_size * actor_output_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_actor_out, batch_size * actor_output_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_critic_hidden, batch_size * hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_critic_fc2_output, batch_size * critic_output_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_critic_out, batch_size * critic_output_dim * sizeof(float)));

  // 拷贝参数到 device
  CHECK_CUDA(cudaMemcpy(d_input, input, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_actor_fc1_w, actor_fc1_w, input_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_actor_fc1_b, actor_fc1_b, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_actor_fc2_w, actor_fc2_w, actor_output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_actor_fc2_b, actor_fc2_b, actor_output_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_actor_head_w, actor_head_w, actor_output_dim * actor_output_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_actor_head_b, actor_head_b, actor_output_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_critic_fc1_w, critic_fc1_w, input_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_critic_fc1_b, critic_fc1_b, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_critic_fc2_w, critic_fc2_w, critic_output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_critic_fc2_b, critic_fc2_b, critic_output_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_critic_head_w, critic_head_w, critic_output_dim * critic_output_dim * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_critic_head_b, critic_head_b, critic_output_dim * sizeof(float), cudaMemcpyHostToDevice));

  // CUDA流和核函数配置
  cudaStream_t actor_stream, critic_stream;
  CHECK_CUDA(cudaStreamCreate(&actor_stream));
  CHECK_CUDA(cudaStreamCreate(&critic_stream));
  dim3 block(TILE_SIZE, TILE_SIZE);
  size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

  // Actor
  dim3 grid_actor_fc1((hidden_dim + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
  fc_relu_forward_kernel<<<grid_actor_fc1, block, shared_mem_size, actor_stream>>>(
      d_input, d_actor_fc1_w, d_actor_fc1_b, d_actor_hidden, input_dim, hidden_dim, batch_size);

  dim3 grid_actor_fc2((hidden_dim + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
  fc_relu_forward_kernel<<<grid_actor_fc2, block, shared_mem_size, actor_stream>>>(
      d_actor_hidden, d_actor_fc2_w, d_actor_fc2_b, d_actor_fc2_output, hidden_dim, hidden_dim, batch_size);

  optimized_linear_forward_kernel<<<grid_actor_fc2, block, shared_mem_size, actor_stream>>>(
      d_actor_fc2_output, d_actor_head_w, d_actor_head_b, d_actor_out, 
      hidden_dim,        // 正确输入维度
      actor_output_dim, 
      batch_size);

  // Critic
  dim3 grid_critic_fc1((hidden_dim + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
  fc_relu_forward_kernel<<<grid_critic_fc1, block, shared_mem_size, critic_stream>>>(
      d_input, d_critic_fc1_w, d_critic_fc1_b, d_critic_hidden, input_dim, hidden_dim, batch_size);

  dim3 grid_critic_fc2((hidden_dim + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
  fc_relu_forward_kernel<<<grid_critic_fc2, block, shared_mem_size, critic_stream>>>(
      d_critic_hidden, d_critic_fc2_w, d_critic_fc2_b, d_critic_fc2_output, 
      hidden_dim,         // 输入维度
      hidden_dim,  // 输出维度
      batch_size);

  optimized_linear_forward_kernel<<<grid_critic_fc2, block, shared_mem_size, critic_stream>>>(
      d_critic_fc2_output, d_critic_head_w, d_critic_head_b, d_critic_out, 
      hidden_dim, critic_output_dim, batch_size);

  CHECK_CUDA(cudaStreamSynchronize(actor_stream));
  CHECK_CUDA(cudaStreamSynchronize(critic_stream));

  // 拷贝结果回主机
  CHECK_CUDA(cudaMemcpy(actor_output, d_actor_out, batch_size * actor_output_dim * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(critic_output, d_critic_out, batch_size * critic_output_dim * sizeof(float), cudaMemcpyDeviceToHost));

  // 释放所有 device 内存
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

__global__ void merge_gradients_kernel(float *out, const float *a,
                                       const float *b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

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
    float *grad_actor_fc1_w, float *grad_actor_fc1_b,
    float *grad_actor_fc2_w, float *grad_actor_fc2_b,
    float *grad_actor_head_w, float *grad_actor_head_b,
    float *grad_critic_fc1_w, float *grad_critic_fc1_b,
    float *grad_critic_fc2_w, float *grad_critic_fc2_b,
    float *grad_critic_head_w, float *grad_critic_head_b
) {
    // 1. 分配所有 device 内存（forward 所需的所有中间结果）
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

    float *d_grad_actor_fc2_output, *d_grad_critic_fc2_output;
    float *d_grad_actor_hidden, *d_grad_critic_hidden;

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

    CHECK_CUDA(cudaMalloc(&d_actor_hidden, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_fc2_output, batch_size * actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actor_out, batch_size * actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_hidden, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_fc2_output, batch_size * critic_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_critic_out, batch_size * critic_output_dim * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_grad_actor_output, batch_size * actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_output, batch_size * critic_output_dim * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_grad_actor_fc1_w, input_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_fc1_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_fc2_w, actor_output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_fc2_b, actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_head_w, actor_output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_head_b, actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_fc1_w, input_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_fc1_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_fc2_w, critic_output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_fc2_b, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_head_w, critic_output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_head_b, critic_output_dim * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_grad_actor_fc2_output, batch_size * actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_fc2_output, batch_size * critic_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_actor_hidden, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_critic_hidden, batch_size * hidden_dim * sizeof(float)));

    // 2. 拷贝输入、参数、梯度到 device
    CHECK_CUDA(cudaMemcpy(d_input, input, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_fc1_w, actor_fc1_w, input_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_fc1_b, actor_fc1_b, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_fc2_w, actor_fc2_w, actor_output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_fc2_b, actor_fc2_b, actor_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_head_w, actor_head_w, actor_output_dim * actor_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_actor_head_b, actor_head_b, actor_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_fc1_w, critic_fc1_w, input_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_fc1_b, critic_fc1_b, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_fc2_w, critic_fc2_w, critic_output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_fc2_b, critic_fc2_b, critic_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_head_w, critic_head_w, critic_output_dim * critic_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_critic_head_b, critic_head_b, critic_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad_actor_output, grad_actor_output, batch_size * actor_output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad_critic_output, grad_critic_output, batch_size * critic_output_dim * sizeof(float), cudaMemcpyHostToDevice));

    // 3. forward kernel（同 forward），保存所有中间结果
    dim3 block(TILE_SIZE, TILE_SIZE);
    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    // Actor
    dim3 grid_actor_fc1((hidden_dim + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    fc_relu_forward_kernel<<<grid_actor_fc1, block, shared_mem_size>>>(
        d_input, d_actor_fc1_w, d_actor_fc1_b, d_actor_hidden, input_dim, hidden_dim, batch_size);

    dim3 grid_actor_fc2((hidden_dim + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    fc_relu_forward_kernel<<<grid_actor_fc2, block, shared_mem_size>>>(
        d_actor_hidden, d_actor_fc2_w, d_actor_fc2_b, d_actor_fc2_output, hidden_dim, hidden_dim, batch_size);

    optimized_linear_forward_kernel<<<grid_actor_fc2, block, shared_mem_size>>>(
        d_actor_fc2_output, d_actor_head_w, d_actor_head_b, d_actor_out, hidden_dim, actor_output_dim, batch_size);

    // Critic
    dim3 grid_critic_fc1((hidden_dim + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    fc_relu_forward_kernel<<<grid_critic_fc1, block, shared_mem_size>>>(
        d_input, d_critic_fc1_w, d_critic_fc1_b, d_critic_hidden, input_dim, hidden_dim, batch_size);

    dim3 grid_critic_fc2((hidden_dim + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    fc_relu_forward_kernel<<<grid_critic_fc2, block, shared_mem_size>>>(
        d_critic_hidden, d_critic_fc2_w, d_critic_fc2_b, d_critic_fc2_output, hidden_dim, hidden_dim, batch_size);

    optimized_linear_forward_kernel<<<grid_critic_fc2, block, shared_mem_size>>>(
        d_critic_fc2_output, d_critic_head_w, d_critic_head_b, d_critic_out, hidden_dim, critic_output_dim, batch_size);

    // 4. backward kernel，计算所有梯度
    // 初始化梯度为0
    CHECK_CUDA(cudaMemset(d_grad_actor_fc1_w, 0,
                          input_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_actor_fc1_b, 0, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_actor_fc2_w, 0,
                          actor_output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_actor_fc2_b, 0, actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_actor_head_w, 0,
                          hidden_dim * actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_actor_head_b, 0, actor_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_critic_fc1_w, 0,
                          input_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_critic_fc1_b, 0, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_critic_fc2_w, 0,
                          critic_output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_critic_fc2_b, 0, critic_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_critic_head_w, 0,
                          hidden_dim * critic_output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_critic_head_b, 0, critic_output_dim * sizeof(float)));

    // 创建CUDA流
    cudaStream_t actor_stream, critic_stream;
    CHECK_CUDA(cudaStreamCreate(&actor_stream));
    CHECK_CUDA(cudaStreamCreate(&critic_stream));

    // Actor路径反向传播
    // 1. Actor Head层梯度
    dim3 grid_head_grad(actor_output_dim, 1);
    dim3 block_head_grad(BLOCK_DIM);
    size_t shared_head_grad = BLOCK_DIM * sizeof(float);

    // 权重梯度
    optimized_weight_grad_kernel<<<grid_head_grad, block_head_grad,
                                  shared_head_grad, actor_stream>>>(
        d_actor_fc2_output, d_grad_actor_output, d_grad_actor_head_w,
        hidden_dim, actor_output_dim, batch_size);

    // 偏置梯度
    optimized_bias_grad_kernel<<<grid_head_grad, block_head_grad,
                                shared_head_grad, actor_stream>>>(
        d_grad_actor_output, d_grad_actor_head_b, actor_output_dim, batch_size);

    // 2. 计算Actor fc2输出梯度
    compute_hidden_grad_kernel<<<(batch_size * actor_output_dim + BLOCK_DIM - 1) /
                                    BLOCK_DIM,
                                BLOCK_DIM, 0, actor_stream>>>(
        d_grad_actor_output, d_actor_head_w,
        d_grad_actor_fc2_output, hidden_dim, 
        actor_output_dim, batch_size);

    // 3. Actor fc2层梯度
    // 权重梯度
    optimized_weight_grad_kernel<<<dim3(hidden_dim, hidden_dim), block_head_grad,
                                  shared_head_grad, actor_stream>>>(
        d_actor_hidden, d_grad_actor_fc2_output, d_grad_actor_fc2_w,
        hidden_dim, hidden_dim, batch_size);

    // 偏置梯度
    optimized_bias_grad_kernel<<<dim3(hidden_dim, 1), block_head_grad,
                                shared_head_grad, actor_stream>>>(
        d_grad_actor_fc2_output, d_grad_actor_fc2_b, hidden_dim, batch_size);

    // 4. 计算Actor隐藏层梯度 (带relu导数)
    compute_relu_hidden_grad_kernel<<<(batch_size * hidden_dim + BLOCK_DIM - 1) /
                                          BLOCK_DIM,
                                      BLOCK_DIM, 0, actor_stream>>>(
        d_grad_actor_fc2_output, d_actor_fc2_w, d_actor_hidden,
        d_grad_actor_hidden, hidden_dim, hidden_dim, batch_size);

    // 5. Actor fc1层梯度
    // 权重梯度
    optimized_weight_grad_kernel<<<dim3(input_dim, hidden_dim), block_head_grad,
                                  shared_head_grad, actor_stream>>>(
        d_input, d_grad_actor_hidden, d_grad_actor_fc1_w,
        input_dim, hidden_dim, batch_size);

    // 偏置梯度
    optimized_bias_grad_kernel<<<dim3(hidden_dim, 1), block_head_grad,
                                shared_head_grad, actor_stream>>>(
        d_grad_actor_hidden, d_grad_actor_fc1_b, hidden_dim, batch_size);

    // Critic路径反向传播 (与Actor类似)
    // 1. Critic Head层梯度
    optimized_weight_grad_kernel<<<grid_head_grad, block_head_grad,
                                  shared_head_grad, critic_stream>>>(
        d_critic_fc2_output, d_grad_critic_output, d_grad_critic_head_w,
        hidden_dim, critic_output_dim, batch_size);
    optimized_bias_grad_kernel<<<grid_head_grad, block_head_grad,
                                shared_head_grad, critic_stream>>>(
        d_grad_critic_output, d_grad_critic_head_b, critic_output_dim, batch_size);

    // 2. 计算Critic fc2输出梯度
    compute_hidden_grad_kernel<<<(batch_size * critic_output_dim + BLOCK_DIM - 1) /
                                    BLOCK_DIM,
                                BLOCK_DIM, 0, critic_stream>>>(
        d_grad_critic_output, d_critic_head_w,
        d_grad_critic_fc2_output,
        hidden_dim, critic_output_dim, batch_size);

    // 3. Critic fc2层梯度
    optimized_weight_grad_kernel<<<dim3(hidden_dim, hidden_dim), block_head_grad,
                                  shared_head_grad, critic_stream>>>(
        d_critic_hidden, d_grad_critic_fc2_output, d_grad_critic_fc2_w,
        hidden_dim, hidden_dim, batch_size);

    optimized_bias_grad_kernel<<<dim3(hidden_dim, 1), block_head_grad,
                            shared_head_grad, critic_stream>>>(
    d_grad_critic_fc2_output, d_grad_critic_fc2_b, hidden_dim, batch_size);

    // 4. 计算Critic隐藏层梯度 (带relu导数)
    compute_relu_hidden_grad_kernel<<<(batch_size * hidden_dim + BLOCK_DIM - 1) /
                                          BLOCK_DIM,
                                      BLOCK_DIM, 0, critic_stream>>>(
        d_grad_critic_fc2_output, d_critic_fc2_w, d_critic_hidden,
        d_grad_critic_hidden, hidden_dim, hidden_dim, batch_size); 

    // 5. Critic fc1层梯度
    optimized_weight_grad_kernel<<<dim3(input_dim, hidden_dim), block_head_grad,
                                  shared_head_grad, critic_stream>>>(
        d_input, d_grad_critic_hidden, d_grad_critic_fc1_w,
        input_dim, hidden_dim, batch_size);
    optimized_bias_grad_kernel<<<dim3(hidden_dim, 1), block_head_grad,
                                shared_head_grad, critic_stream>>>(
        d_grad_critic_hidden, d_grad_critic_fc1_b, hidden_dim, batch_size);

    // 等待两个流完成
    CHECK_CUDA(cudaStreamSynchronize(actor_stream));
    CHECK_CUDA(cudaStreamSynchronize(critic_stream));

    printf("Actor and Critic backward pass completed.\n");


    // 5. 拷贝所有梯度回主机
    CHECK_CUDA(cudaMemcpy(grad_actor_fc1_w, d_grad_actor_fc1_w, input_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_actor_fc1_b, d_grad_actor_fc1_b, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_actor_fc2_w, d_grad_actor_fc2_w, actor_output_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_actor_fc2_b, d_grad_actor_fc2_b, actor_output_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_actor_head_w, d_grad_actor_head_w, actor_output_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_actor_head_b, d_grad_actor_head_b, actor_output_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_critic_fc1_w, d_grad_critic_fc1_w, input_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_critic_fc1_b, d_grad_critic_fc1_b, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_critic_fc2_w, d_grad_critic_fc2_w, critic_output_dim * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_critic_fc2_b, d_grad_critic_fc2_b, critic_output_dim * sizeof(float), cudaMemcpyDeviceToHost));
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
    CHECK_CUDA(cudaFree(d_grad_critic_fc2_output));
    CHECK_CUDA(cudaFree(d_grad_actor_hidden));
    CHECK_CUDA(cudaFree(d_grad_critic_hidden));
}
} // extern "C"