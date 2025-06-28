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


class A2C:
    def __init__(self, actor_and_critic, value_loss_coef=1., actor_loss_coef=1., entropy_coef=0.1, learning_rate=1e-3):
        self.actor_and_critic = actor_and_critic
        self.value_loss_coef = value_loss_coef
        self.actor_loss_coef = actor_loss_coef
        self.entropy_coef = entropy_coef
        self.optimizer = AdamOptimizer(self.actor_and_critic.parameters(), lr=learning_rate)

    def update(self, traj):
        print("traj finalize...")
        traj.finalize()
        observations = traj.get_observations()
        actions = traj.get_actions()
        rewards = traj.get_rewards()
        returns = traj.get_returns()

        total_episode_reward = np.sum(rewards)
        mean_episode_reward = np.mean(rewards)
        num_steps = len(observations)

        actor_out, critic_out = self.actor_and_critic.forward(observations)
        logits = actor_out
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        values = critic_out

        actLogProbs = np.log(probs[np.arange(len(actions)), actions] + 1e-8)
        distEntropy = -np.sum(probs * np.log(probs + 1e-8), axis=1, keepdims=True)

        advantages = returns.reshape(-1, 1) - values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        value_loss = np.mean(advantages ** 2)
        action_loss = -np.mean(advantages * actLogProbs)
        distEntropy_loss = -np.mean(distEntropy)

        loss = (value_loss * self.value_loss_coef +
                action_loss * self.actor_loss_coef +
                distEntropy_loss * self.entropy_coef)

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(actions)), actions] = 1
        grad_actor = - (advantages * (one_hot - probs)) / len(advantages) * self.actor_loss_coef
        grad_value = -2 * (advantages) / len(advantages) * self.value_loss_coef

        self.optimizer.zero_grad()

        grads = self.actor_and_critic.backward(
            observations, grad_actor.astype(np.float32), grad_value.astype(np.float32)
        )
        self._apply_grads(grads)

        return {
            "Value loss": float(value_loss),
            "Action loss": float(action_loss),
            "distEntropy loss": float(distEntropy_loss),
            "Loss": float(loss),
            "total_episode_reward": float(total_episode_reward),
            "mean_episode_reward": float(mean_episode_reward),
            "num_steps": num_steps
        }

    def _apply_grads(self, grads, max_norm=0.01):
        # 检查梯度是否有NaN/Inf
        for i, g in enumerate(grads):
            if not np.all(np.isfinite(g)):
                print(f"Gradient {i} has NaN/Inf, skipping update.")
                return
        # 全局范数裁剪
        total_norm = np.sqrt(sum(np.sum(g.astype(np.float64) ** 2) for g in grads))
        if not np.isfinite(total_norm) or total_norm > 1e8:
            print(f"Warning: Gradient norm overflow: {total_norm}, skipping update.")
            return
        clip_coef = max_norm / (total_norm + 1e-8)
        if clip_coef < 1.0:
            grads = [g * clip_coef for g in grads]
        self.optimizer.step(grads)