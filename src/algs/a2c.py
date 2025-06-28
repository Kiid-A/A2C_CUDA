import numpy as np

class A2C:
    def __init__(self, actor_and_critic, value_loss_coef=1., actor_loss_coef=1., entropy_coef=0.1, learning_rate=1e-3, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8):
        self.actor_and_critic = actor_and_critic
        self.value_loss_coef = value_loss_coef
        self.actor_loss_coef = actor_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate

        # Adam参数
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.adam_t = 0

        # 参数列表（全部numpy数组引用）
        self.params = [
            self.actor_and_critic.shared_w, self.actor_and_critic.shared_b,
            self.actor_and_critic.actor1_w, self.actor_and_critic.actor1_b,
            self.actor_and_critic.actor2_w, self.actor_and_critic.actor2_b,
            self.actor_and_critic.actor_head_w, self.actor_and_critic.actor_head_b,
            self.actor_and_critic.critic1_w, self.actor_and_critic.critic1_b,
            self.actor_and_critic.critic2_w, self.actor_and_critic.critic2_b,
            self.actor_and_critic.critic_head_w, self.actor_and_critic.critic_head_b
        ]
        # Adam状态
        self.adam_m = [np.zeros_like(p) for p in self.params]
        self.adam_v = [np.zeros_like(p) for p in self.params]

    def update(self, traj):
        print("traj finalize...")
        traj.finalize()
        observations = traj.get_observations()  # numpy, shape [T, obs_dim]
        actions = traj.get_actions()            # numpy, shape [T,]
        rewards = traj.get_rewards()            # numpy, shape [T,]
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        traj.rewards_ = list(rewards)
        returns = traj.get_returns()            # numpy, shape [T,]

        total_episode_reward = np.sum(rewards)
        mean_episode_reward = np.mean(rewards)
        num_steps = len(observations)

        # 前向
        actor_out, critic_out = self.actor_and_critic.forward(observations)
        logits = actor_out
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        values = critic_out

        # 采样动作的log概率
        actLogProbs = np.log(probs[np.arange(len(actions)), actions] + 1e-8)
        distEntropy = -np.sum(probs * np.log(probs + 1e-8), axis=1, keepdims=True)

        advantages = returns.reshape(-1, 1) - values

        value_loss = np.mean(advantages ** 2)
        action_loss = -np.mean(advantages * actLogProbs)
        distEntropy_loss = -np.mean(distEntropy)

        loss = (value_loss * self.value_loss_coef +
                action_loss * self.actor_loss_coef +
                distEntropy_loss * self.entropy_coef)

        # 计算输出的梯度（手动推导A2C损失对actor/critic输出的梯度）
        grad_value = -2 * (advantages) / len(advantages) * self.value_loss_coef  # d(value_loss)/d(value)
        grad_actor = - (advantages * (1 - probs)) / len(advantages) * self.actor_loss_coef  # d(action_loss)/d(logits)

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

    def _apply_grads(self, grads, max_norm=1.0):
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        if not np.isfinite(total_norm) or total_norm > 1e6:
            print("Warning: Gradient norm overflow, skipping update.")
            return
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            grads = [g * clip_coef for g in grads]
        self.adam_t += 1
        lr = self.learning_rate
        b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_eps
        for i, (p, g) in enumerate(zip(self.params, grads)):
            m = self.adam_m[i]
            v = self.adam_v[i]
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * (g ** 2)
            m_hat = m / (1 - b1 ** self.adam_t)
            v_hat = v / (1 - b2 ** self.adam_t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            if not np.all(np.isfinite(update)):
                print(f"Param {i} update has NaN/Inf, skipping.")
                continue
            p -= update