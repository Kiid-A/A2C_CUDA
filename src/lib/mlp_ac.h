#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void cuda_forward(const float *input, int batch_size, int input_dim, int hidden_dim,
                  int actor_output_dim, int critic_output_dim,
                  const float *shared_w, const float *shared_b,
                  const float *actor_fc1_w, const float *actor_fc1_b,
                  const float *actor_fc2_w, const float *actor_fc2_b,
                  const float *actor_head_w, const float *actor_head_b,
                  const float *critic_fc1_w, const float *critic_fc1_b,
                  const float *critic_fc2_w, const float *critic_fc2_b,
                  const float *critic_head_w, const float *critic_head_b,
                  float *actor_output, float *critic_output);

void cuda_backward(const float *input, int batch_size, int input_dim, int hidden_dim,
                   int actor_output_dim, int critic_output_dim,
                   const float *shared_w, const float *shared_b,
                   const float *actor_fc1_w, const float *actor_fc1_b,
                   const float *actor_fc2_w, const float *actor_fc2_b,
                   const float *actor_head_w, const float *actor_head_b,
                   const float *critic_fc1_w, const float *critic_fc1_b,
                   const float *critic_fc2_w, const float *critic_fc2_b,
                   const float *critic_head_w, const float *critic_head_b,
                   const float *grad_actor_output, const float *grad_critic_output,
                   float *grad_shared_w, float *grad_shared_b,
                   float *grad_actor_fc1_w, float *grad_actor_fc1_b,
                   float *grad_actor_fc2_w, float *grad_actor_fc2_b,
                   float *grad_actor_head_w, float *grad_actor_head_b,
                   float *grad_critic_fc1_w, float *grad_critic_fc1_b,
                   float *grad_critic_fc2_w, float *grad_critic_fc2_b,
                   float *grad_critic_head_w, float *grad_critic_head_b);

#ifdef __cplusplus
}
#endif