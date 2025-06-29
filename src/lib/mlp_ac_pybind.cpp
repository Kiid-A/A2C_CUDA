// mlp_ac_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include "mlp_ac.h"

namespace py = pybind11;

// 正确的析构函数签名
static void capsule_cuda_free(PyObject *capsule) {
    void *ptr = PyCapsule_GetPointer(capsule, "cuda_ptr");
    if (ptr) cudaFree(ptr);
}

py::tuple mlp_forward(
    py::array_t<float> input,
    py::array_t<float> actor1_w, py::array_t<float> actor1_b,
    py::array_t<float> actor2_w, py::array_t<float> actor2_b,
    py::array_t<float> actor_head_w, py::array_t<float> actor_head_b,
    py::array_t<float> critic1_w, py::array_t<float> critic1_b,
    py::array_t<float> critic2_w, py::array_t<float> critic2_b,
    py::array_t<float> critic_head_w, py::array_t<float> critic_head_b,
    int hidden_dim, int n_actions
) {
    auto buf_input = input.request();
    int batch = buf_input.shape[0];
    int obs_dim = buf_input.shape[1];

    std::vector<float> actor_out(batch * n_actions);
    std::vector<float> critic_out(batch * 1);

    // 中间结果指针
    float *d_actor_linear_fc1 = nullptr, *d_actor_linear_fc2 = nullptr;
    float *d_critic_linear_fc1 = nullptr, *d_critic_linear_fc2 = nullptr;

    cuda_forward(
        static_cast<float*>(buf_input.ptr), batch, obs_dim, hidden_dim,
        n_actions, 1, // actor_output_dim, critic_output_dim
        static_cast<float*>(actor1_w.request().ptr),
        static_cast<float*>(actor1_b.request().ptr),
        static_cast<float*>(actor2_w.request().ptr),
        static_cast<float*>(actor2_b.request().ptr),
        static_cast<float*>(actor_head_w.request().ptr),
        static_cast<float*>(actor_head_b.request().ptr),
        static_cast<float*>(critic1_w.request().ptr),
        static_cast<float*>(critic1_b.request().ptr),
        static_cast<float*>(critic2_w.request().ptr),
        static_cast<float*>(critic2_b.request().ptr),
        static_cast<float*>(critic_head_w.request().ptr),
        static_cast<float*>(critic_head_b.request().ptr),
        actor_out.data(), critic_out.data(),
        &d_actor_linear_fc1, &d_actor_linear_fc2,
        &d_critic_linear_fc1, &d_critic_linear_fc2
    );

    // 创建Python端的输出数组
    py::array_t<float> py_actor({batch, n_actions});
    py::array_t<float> py_critic({batch, 1});
    memcpy(py_actor.mutable_data(), actor_out.data(), sizeof(float) * batch * n_actions);
    memcpy(py_critic.mutable_data(), critic_out.data(), sizeof(float) * batch * 1);

    // 用PyCapsule封装设备指针
    auto cap1 = py::capsule(d_actor_linear_fc1, "cuda_ptr", capsule_cuda_free);
    auto cap2 = py::capsule(d_actor_linear_fc2, "cuda_ptr", capsule_cuda_free);
    auto cap3 = py::capsule(d_critic_linear_fc1, "cuda_ptr", capsule_cuda_free);
    auto cap4 = py::capsule(d_critic_linear_fc2, "cuda_ptr", capsule_cuda_free);

    return py::make_tuple(py_actor, py_critic, cap1, cap2, cap3, cap4);
}

py::tuple mlp_backward(
    py::array_t<float> input,
    py::array_t<float> actor1_w, py::array_t<float> actor1_b,
    py::array_t<float> actor2_w, py::array_t<float> actor2_b,
    py::array_t<float> actor_head_w, py::array_t<float> actor_head_b,
    py::array_t<float> critic1_w, py::array_t<float> critic1_b,
    py::array_t<float> critic2_w, py::array_t<float> critic2_b,
    py::array_t<float> critic_head_w, py::array_t<float> critic_head_b,
    py::capsule cap1, py::capsule cap2, py::capsule cap3, py::capsule cap4,
    py::array_t<float> grad_actor_output,
    py::array_t<float> grad_critic_output,
    int batch_size, int input_dim, int hidden_dim, int n_actions
) {
    // 准备梯度输出数组
    std::vector<py::array_t<float>> grads;
    grads.emplace_back(py::array_t<float>({input_dim, hidden_dim})); // actor1_w
    grads.emplace_back(py::array_t<float>({hidden_dim}));            // actor1_b
    grads.emplace_back(py::array_t<float>({hidden_dim, hidden_dim}));// actor2_w
    grads.emplace_back(py::array_t<float>({hidden_dim}));            // actor2_b
    grads.emplace_back(py::array_t<float>({hidden_dim, n_actions})); // actor_head_w
    grads.emplace_back(py::array_t<float>({n_actions}));             // actor_head_b
    grads.emplace_back(py::array_t<float>({input_dim, hidden_dim})); // critic1_w
    grads.emplace_back(py::array_t<float>({hidden_dim}));            // critic1_b
    grads.emplace_back(py::array_t<float>({hidden_dim, hidden_dim}));// critic2_w
    grads.emplace_back(py::array_t<float>({hidden_dim}));            // critic2_b
    grads.emplace_back(py::array_t<float>({hidden_dim, 1}));         // critic_head_w
    grads.emplace_back(py::array_t<float>({1}));                     // critic_head_b

    // 调用CUDA反向传播
    cuda_backward(
        static_cast<float*>(input.request().ptr),
        batch_size, input_dim, hidden_dim, n_actions, 1,
        static_cast<float*>(actor1_w.request().ptr),
        static_cast<float*>(actor1_b.request().ptr),
        static_cast<float*>(actor2_w.request().ptr),
        static_cast<float*>(actor2_b.request().ptr),
        static_cast<float*>(actor_head_w.request().ptr),
        static_cast<float*>(actor_head_b.request().ptr),
        static_cast<float*>(critic1_w.request().ptr),
        static_cast<float*>(critic1_b.request().ptr),
        static_cast<float*>(critic2_w.request().ptr),
        static_cast<float*>(critic2_b.request().ptr),
        static_cast<float*>(critic_head_w.request().ptr),
        static_cast<float*>(critic_head_b.request().ptr),
        static_cast<float*>(grad_actor_output.request().ptr),
        static_cast<float*>(grad_critic_output.request().ptr),
        reinterpret_cast<float*>(cap1.get_pointer()),
        reinterpret_cast<float*>(cap2.get_pointer()),
        reinterpret_cast<float*>(cap3.get_pointer()),
        reinterpret_cast<float*>(cap4.get_pointer()),
        static_cast<float*>(grads[0].request().ptr),
        static_cast<float*>(grads[1].request().ptr),
        static_cast<float*>(grads[2].request().ptr),
        static_cast<float*>(grads[3].request().ptr),
        static_cast<float*>(grads[4].request().ptr),
        static_cast<float*>(grads[5].request().ptr),
        static_cast<float*>(grads[6].request().ptr),
        static_cast<float*>(grads[7].request().ptr),
        static_cast<float*>(grads[8].request().ptr),
        static_cast<float*>(grads[9].request().ptr),
        static_cast<float*>(grads[10].request().ptr),
        static_cast<float*>(grads[11].request().ptr)
    );
    
    return py::make_tuple(
        grads[0], grads[1], grads[2], grads[3], 
        grads[4], grads[5], grads[6], grads[7],
        grads[8], grads[9], grads[10], grads[11]
    );
}

PYBIND11_MODULE(mlp_ac_cuda, m) {
    m.def("mlp_forward", &mlp_forward, "MLP forward by CUDA");
    m.def("mlp_backward", &mlp_backward, "MLP backward by CUDA");
}