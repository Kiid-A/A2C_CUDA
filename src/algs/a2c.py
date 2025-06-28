import numpy as np

class AdamOptimizer:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params  # 模型参数（numpy数组）
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # 迭代次数
        self.m = [np.zeros_like(p) for p in params]  # 一阶动量
        self.v = [np.zeros_like(p) for p in params]  # 二阶动量

    def step(self, grads):
        """执行参数更新"""
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # 计算动量
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 参数更新
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # 处理NaN/Inf
            if not np.all(np.isfinite(update)):
                print(f"Param {i} update has NaN/Inf, skipping.")
                continue
            p -= update
            
        # 检查参数有效性
        for i, p in enumerate(self.params):
            if not np.all(np.isfinite(p)):
                print(f"Param {i} has NaN/Inf! Resetting parameter.")
                p[:] = np.random.uniform(-0.01, 0.01, size=p.shape).astype(np.float32)

    def zero_grad(self):
        """清空梯度相关状态（重置动量和迭代次数）"""
        self.t = 0
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        print("Optimizer state reset (momentum and iteration counter cleared).")

    def reset_parameters(self):
        """重置所有参数（用于重新初始化模型）"""
        for i, p in enumerate(self.params):
            p[:] = np.random.uniform(-0.01, 0.01, size=p.shape).astype(np.float32)
        self.zero_grad()  # 同时重置优化器状态

class SGDOptimizer:
    def __init__(self, params, lr=1e-3, momentum=0.9):
        """
        参数:
            params: 模型参数列表（NumPy 数组）
            lr: 学习率（默认 1e-3）
            momentum: 动量系数（默认 0.9，设为 0 则为普通 SGD）
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p) for p in params]  # 动量累积

    def step(self, grads):
        """
        执行一步参数更新
        """
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # 动量更新
            self.velocity[i] = self.momentum * self.velocity[i] + g
            # 参数更新
            p -= self.lr * self.velocity[i]

    def zero_grad(self):
        """清空动量状态"""
        self.velocity = [np.zeros_like(p) for p in self.params]

class A2C:
    def __init__(self, actor_and_critic, value_loss_coef=1., actor_loss_coef=1., entropy_coef=1e-3, learning_rate=1e-3):
        self.actor_and_critic = actor_and_critic
        self.value_loss_coef = value_loss_coef
        self.actor_loss_coef = actor_loss_coef
        self.entropy_coef = entropy_coef
        self.optimizer = AdamOptimizer(self.actor_and_critic.parameters(), lr=learning_rate)
        # self.optimizer = SGDOptimizer(self.actor_and_critic.parameters(), lr=learning_rate)

    def update(self, traj):
        print("traj finalize...")
        traj.finalize()
        observations = traj.get_observations()
        actions = traj.get_actions()
        returns = traj.get_returns()

        actor_out, critic_out = self.actor_and_critic.forward(observations)
        logits = actor_out

        # probs = softmax(actor_out)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        values = critic_out

        actLogProbs = np.log(probs[np.arange(len(actions)), actions] + 1e-8)
        distEntropy = -np.sum(probs * np.log(probs + 1e-8), axis=1, keepdims=True)

        advantages = returns.reshape(-1, 1) - values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 2e-8)

        print(advantages)
        value_loss = np.mean(np.power(advantages, 2)) * self.value_loss_coef
        action_loss = -np.mean(advantages * actLogProbs) * self.actor_loss_coef
        distEntropy_loss = -np.mean(distEntropy) * self.entropy_coef

        loss = (value_loss  + action_loss + distEntropy_loss)

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(actions)), actions] = 1
        
        # 修正后的actor梯度计算，包含策略梯度和熵正则化梯度
        policy_grad = - (advantages * (one_hot - probs)) / len(advantages) * self.actor_loss_coef
        entropy_grad = - (np.log(probs + 1e-8) + 1) / len(advantages) * self.entropy_coef
        grad_actor = policy_grad + entropy_grad
        grad_actor = - grad_actor
        
        # 修正后的critic梯度计算，增加稳定性
        grad_value = -2 * (advantages) / (len(advantages) + 1e-8) * self.value_loss_coef
        grad_value = - grad_value

        self.optimizer.zero_grad()

        grads = self.actor_and_critic.backward(
            observations, grad_actor.astype(np.float32), grad_value.astype(np.float32)
        )
        self._apply_grads(grads)

        return {
            "policy_grad": np.mean(policy_grad),
            "entropy_grad": np.mean(entropy_grad),
            "grad_actor": np.mean(grad_actor),
            "grad_value": np.mean(grad_value),

            "Value loss": float(value_loss),
            "Action loss": float(action_loss),
            "distEntropy loss": float(distEntropy_loss),
            "Loss": float(loss),
            "total_episode_reward": np.sum(traj.get_rewards()),
            "mean_episode_reward": np.mean(traj.get_rewards()),
            "num_steps": len(observations)
        }

    def _apply_grads(self, grads, max_norm=0.01):
        # 检查梯度是否有NaN/Inf
        for i, g in enumerate(grads):
            if not np.all(np.isfinite(g)):
                print(f"Gradient {i} has NaN/Inf, skipping update.")
                return
        # # 全局范数裁剪
        # total_norm = np.sqrt(sum(np.sum(g.astype(np.float64) ** 2) for g in grads))
        # if not np.isfinite(total_norm) or total_norm > 1e8:
        #     print(f"Warning: Gradient norm overflow: {total_norm}, skipping update.")
        #     return
        # clip_coef = max_norm / (total_norm + 1e-8)
        # if clip_coef < 1.0:
        #     grads = [g * clip_coef for g in grads]
        self.optimizer.step(grads)