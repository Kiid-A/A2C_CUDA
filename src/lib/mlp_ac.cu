#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include "mlp_ac.h"
#include <cublas_v2.h>

// 宏定义：向上取整除法
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// 宏定义：CUDA调用检查（简化错误处理）
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// 硬件相关常量
#define WARP_SIZE 32
#define BLOCK_DIM 256
#define TILE_SIZE 16  // 矩阵分块大小（优化共享内存使用）

namespace cg = cooperative_groups;

// ====================== 核心模块：线性层操作 ======================
/**
 * 线性层前向传播内核
 * 关键优化：使用共享内存+分块矩阵乘法加速计算
 * 
 * @param input     输入数据指针 [batch_size, input_dim]
 * @param weights   权重指针 [input_dim, output_dim]
 * @param bias      偏置指针 [output_dim]
 * @param output    输出指针 [batch_size, output_dim]
 * @param input_dim  输入维度
 * @param output_dim 输出维度
 * @param batch_size 批大小
 */
__global__ void linear_forward_kernel(
    const float *__restrict__ input, const float *__restrict__ weights,
    const float *__restrict__ bias, float *__restrict__ output, 
    int input_dim, int output_dim, int batch_size) 
{
    // 动态分配共享内存：输入块+权重块
    extern __shared__ float shared_mem[];
    float *sh_input = shared_mem;
    float *sh_weights = &shared_mem[TILE_SIZE * TILE_SIZE];

    // 计算当前线程处理的全局位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 批中的行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 输出维度索引

    // 边界检查
    if (row >= batch_size || col >= output_dim) return;

    float sum = 0.0f;

    // 分块矩阵乘法：每次处理一个TILE_SIZE x TILE_SIZE的子矩阵
    for (int tile = 0; tile < CEIL_DIV(input_dim, TILE_SIZE); tile++) {
        // 计算当前tile中线程对应的输入和权重位置
        int input_col = tile * TILE_SIZE + threadIdx.x;
        int weight_row = tile * TILE_SIZE + threadIdx.y;

        // 将输入数据加载到共享内存（边界填充0）
        if (input_col < input_dim) {
            sh_input[threadIdx.y * TILE_SIZE + threadIdx.x] = 
                input[row * input_dim + input_col];
        } else {
            sh_input[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        // 将权重数据加载到共享内存（边界填充0）
        if (weight_row < input_dim) {
            sh_weights[threadIdx.x * TILE_SIZE + threadIdx.y] = 
                weights[col * input_dim + weight_row];
        } else {
            sh_weights[threadIdx.x * TILE_SIZE + threadIdx.y] = 0.0f;
        }

        __syncthreads();  // 确保所有线程完成共享内存加载

        // 计算当前tile的局部乘积和
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sh_input[threadIdx.y * TILE_SIZE + k] *
                   sh_weights[threadIdx.x * TILE_SIZE + k];
        }

        __syncthreads();  // 确保所有线程完成计算
    }

    // 添加偏置并写入结果
    output[row * output_dim + col] = sum + bias[col];
}

/**
 * 线性层前向传播封装函数
 * 关键优化：分块矩阵乘法 + 共享内存使用
 */
void linear_forward(
    const float *d_input, const float *d_weights, const float *d_bias,
    float *d_output, int input_dim, int output_dim, int batch_size,
    cudaStream_t stream)
{
    // 配置网格和块维度
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(CEIL_DIV(output_dim, TILE_SIZE), CEIL_DIV(batch_size, TILE_SIZE));
    
    // 计算共享内存需求：输入块+权重块
    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    
    // 启动内核
    linear_forward_kernel<<<grid, block, shared_mem_size, stream>>>(
        d_input, d_weights, d_bias, d_output, input_dim, output_dim, batch_size);
}

// ====================== 核心模块：激活函数 ======================
/**
 * ReLU前向传播内核
 * 简单逐元素操作，高度并行化
 */
__global__ void relu_forward_kernel(
    const float *__restrict__ input, 
    float *__restrict__ output, 
    int size)
{
    // 计算全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

/**
 * ReLU前向传播封装函数
 * 优化：简单高效，每个线程处理一个元素
 */
void relu_forward(
    const float *d_input, 
    float *d_output, 
    int size,
    cudaStream_t stream)
{
    // 配置一维网格
    int blockSize = 256;
    int gridSize = CEIL_DIV(size, blockSize);
    relu_forward_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, size);
}

// ====================== 核心模块：梯度计算 ======================
/**
 * 权重梯度计算内核
 * 关键优化：使用共享内存+并行归约加速梯度求和
 */
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

    // 边界检查
    if (i >= output_dim || j >= input_dim) return;

    float sum = 0.0f;

    // 跨批次并行求和：每个线程处理部分批次数据
    for (int b = tid; b < batch_size; b += blockDim.x) {
        sum += input[b * input_dim + j] * grad_output[b * output_dim + i];
    }

    sh_grad[tid] = sum;
    __syncthreads();

    // 树状归约：在块内求和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_grad[tid] += sh_grad[tid + stride];
        }
        __syncthreads();
    }

    // 原子操作安全写入全局内存
    if (tid == 0) {
        atomicAdd(&grad_weights[i * input_dim + j], sh_grad[0]);
    }
}

/**
 * 权重梯度计算封装函数
 * 优化：二维网格覆盖所有权重参数
 */
void linear_weight_grad(
    const float *d_input, const float *d_grad_output,
    float *d_grad_weights, int input_dim, int output_dim, int batch_size,
    cudaStream_t stream)
{
    // 配置二维网格（每个权重一个线程块）
    dim3 block(BLOCK_DIM);
    dim3 grid(output_dim, input_dim);
    size_t shared_mem_size = BLOCK_DIM * sizeof(float);
    
    linear_weight_grad_kernel<<<grid, block, shared_mem_size, stream>>>(
        d_input, d_grad_output, d_grad_weights, 
        input_dim, output_dim, batch_size);
}

/**
 * 偏置梯度计算内核
 * 优化：使用共享内存+并行归约
 */
__global__ void linear_bias_grad_kernel(
    const float *__restrict__ grad_output,
    float *__restrict__ grad_bias, 
    int output_dim, int batch_size)
{
    extern __shared__ float sh_grad[];
    int i = blockIdx.x;  // 输出维度索引
    int tid = threadIdx.x;

    if (i >= output_dim) return;

    float sum = 0.0f;

    // 跨批次求和
    for (int b = tid; b < batch_size; b += blockDim.x) {
        sum += grad_output[b * output_dim + i];
    }

    sh_grad[tid] = sum;
    __syncthreads();

    // 树状归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_grad[tid] += sh_grad[tid + stride];
        }
        __syncthreads();
    }

    // 原子操作写入结果
    if (tid == 0) {
        atomicAdd(&grad_bias[i], sh_grad[0]);
    }
}

/**
 * 偏置梯度计算封装函数
 * 优化：一维网格覆盖所有偏置元素
 */
void linear_bias_grad(
    const float *d_grad_output,
    float *d_grad_bias, 
    int output_dim, int batch_size,
    cudaStream_t stream)
{
    dim3 block(BLOCK_DIM);
    dim3 grid(output_dim);
    size_t shared_mem_size = BLOCK_DIM * sizeof(float);
    
    linear_bias_grad_kernel<<<grid, block, shared_mem_size, stream>>>(
        d_grad_output, d_grad_bias, output_dim, batch_size);
}

/**
 * 输入梯度计算内核
 * 关键优化：每个线程处理一个输入元素
 */
__global__ void linear_input_grad_kernel(
    const float *__restrict__ grad_output, 
    const float *__restrict__ weights,
    float *__restrict__ grad_input, 
    int in_dim, int out_dim, int batch_size)
{
    // 计算全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_dim)
        return;

    // 解析索引
    int b = idx / in_dim;  // 批次索引
    int j = idx % in_dim;   // 输入维度索引
    
    float sum = 0.0f;

    // 计算该输入元素的梯度
    for (int i = 0; i < out_dim; ++i) {
        sum += grad_output[b * out_dim + i] * weights[i * in_dim + j];
    }

    grad_input[idx] = sum;
}

/**
 * 输入梯度计算封装函数
 * 优化：一维网格覆盖所有输入元素
 */
void linear_input_grad(
    const float *d_grad_output, const float *d_weights,
    float *d_grad_input, int in_dim, int out_dim, int batch_size,
    cudaStream_t stream)
{
    // 配置一维网格
    int blockSize = 256;
    int gridSize = CEIL_DIV(batch_size * in_dim, blockSize);
    linear_input_grad_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_grad_output, d_weights, d_grad_input, in_dim, out_dim, batch_size);
}

/**
 * ReLU反向传播内核
 * 优化：简单逐元素操作
 */
__global__ void relu_backward_kernel(
    const float *__restrict__ grad_output, 
    const float *__restrict__ input,
    float *__restrict__ grad_input, 
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // ReLU导数：输入>0时为1，否则为0
        grad_input[idx] = grad_output[idx] * (input[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

/**
 * ReLU反向传播封装函数
 */
void relu_backward(
    const float *d_grad_output, const float *d_input,
    float *d_grad_input, int size,
    cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = CEIL_DIV(size, blockSize);
    relu_backward_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_grad_output, d_input, d_grad_input, size);
}

// ====================== 工具函数 ======================
/**
 * 设备内存清零
 */
void zero_init(float *d_ptr, size_t size) {
    CHECK_CUDA(cudaMemset(d_ptr, 0, size * sizeof(float)));
}

// ====================== Actor-Critic 网络实现 ======================
extern "C" {

/**
 * 前向传播函数（返回中间结果）
 * 关键优化：
 *  1. 使用双流并行：Actor和Critic网络并行计算
 *  2. 缓存中间结果用于反向传播
 *  3. 高效内存管理：及时释放不再需要的资源
 */
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
    // 输出中间结果指针（用于反向传播）
    float **d_actor_linear_fc1, float **d_actor_linear_fc2,
    float **d_critic_linear_fc1, float **d_critic_linear_fc2) 
{
    // 1. 设备内存分配 ===========================================
    // 分配输入和参数内存
    float *d_input;
    float *d_actor_fc1_w, *d_actor_fc1_b, *d_actor_fc2_w, *d_actor_fc2_b;
    float *d_actor_head_w, *d_actor_head_b;
    float *d_critic_fc1_w, *d_critic_fc1_b, *d_critic_fc2_w, *d_critic_fc2_b;
    float *d_critic_head_w, *d_critic_head_b;

    // 分配中间结果内存
    float *d_actor_hidden, *d_actor_fc2_output, *d_actor_out;
    float *d_critic_hidden, *d_critic_fc2_output, *d_critic_out;

    // 分配并保存中间结果（用于反向传播）
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

    // 2. 数据拷贝到设备 =========================================
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

    // 4. Actor网络前向传播 ======================================
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

    // 5. Critic网络前向传播（与Actor并行）========================
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

    // 6. 同步流并获取结果 =======================================
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

/**
 * 反向传播函数（使用预存的中间结果）
 * 关键优化：
 *  1. 复用前向传播的中间结果，避免重新计算
 *  2. 双流并行：Actor和Critic网络反向传播并行
 *  3. 高效内存管理：及时分配和释放资源
 */
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

    // 5. Actor反向传播（使用前向保存的中间结果）===================
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

    // 8. 拷贝梯度回主机 ========================================
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

    // 9. 释放所有 device 内存 ==================================
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

/**
 * 释放中间结果内存
 */
void cuda_free_intermediate(float *d_actor_linear_fc1, float *d_actor_linear_fc2,
                            float *d_critic_linear_fc1, float *d_critic_linear_fc2) {
    CHECK_CUDA(cudaFree(d_actor_linear_fc1));
    CHECK_CUDA(cudaFree(d_actor_linear_fc2));
    CHECK_CUDA(cudaFree(d_critic_linear_fc1));
    CHECK_CUDA(cudaFree(d_critic_linear_fc2));
}

} // extern "C"