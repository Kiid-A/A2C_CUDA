#pragma once

#ifdef __cplusplus
extern "C" {
#endif

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
    float **d_actor_linear_fc1, float **d_actor_linear_fc2,
    float **d_critic_linear_fc1, float **d_critic_linear_fc2);

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
    const float *d_actor_linear_fc1, const float *d_actor_linear_fc2,
    const float *d_critic_linear_fc1, const float *d_critic_linear_fc2,
    float *grad_actor_fc1_w, float *grad_actor_fc1_b,
    float *grad_actor_fc2_w, float *grad_actor_fc2_b,
    float *grad_actor_head_w, float *grad_actor_head_b,
    float *grad_critic_fc1_w, float *grad_critic_fc1_b,
    float *grad_critic_fc2_w, float *grad_critic_fc2_b,
    float *grad_critic_head_w, float *grad_critic_head_b);

void cuda_free_intermediate(
    float *d_actor_linear_fc1, float *d_actor_linear_fc2,
    float *d_critic_linear_fc1, float *d_critic_linear_fc2);

#ifdef __cplusplus
}
#endif