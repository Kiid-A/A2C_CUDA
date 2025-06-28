import numpy as np
import mlp_ac_cuda

class MlpACManual:
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        def xavier(shape):
            fan_in, fan_out = shape[0], shape[1]
            limit = np.sqrt(3.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, size=shape).astype(np.float32)

        # 权重和偏置
        self.shared_w = xavier((obs_dim, hidden_dim))
        self.shared_b = np.zeros(hidden_dim, dtype=np.float32)
        self.actor1_w = xavier((hidden_dim, hidden_dim))
        self.actor1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.actor2_w = xavier((hidden_dim, hidden_dim))
        self.actor2_b = np.zeros(hidden_dim, dtype=np.float32)
        self.actor_head_w = xavier((hidden_dim, n_actions))
        self.actor_head_b = np.zeros(n_actions, dtype=np.float32)
        self.critic1_w = xavier((hidden_dim, hidden_dim))
        self.critic1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.critic2_w = xavier((hidden_dim, hidden_dim))
        self.critic2_b = np.zeros(hidden_dim, dtype=np.float32)
        self.critic_head_w = xavier((hidden_dim, 1))
        self.critic_head_b = np.zeros(1, dtype=np.float32)

    def forward(self, obs):
        actor_out, critic_out = mlp_ac_cuda.mlp_forward(
            obs.astype(np.float32),
            self.shared_w, self.shared_b,
            self.actor1_w, self.actor1_b,
            self.actor2_w, self.actor2_b,
            self.actor_head_w, self.actor_head_b,
            self.critic1_w, self.critic1_b,
            self.critic2_w, self.critic2_b,
            self.critic_head_w, self.critic_head_b,
            self.hidden_dim, self.n_actions
        )
        return actor_out, critic_out

    def backward(self, obs, grad_actor_output, grad_critic_output):
        grads = mlp_ac_cuda.mlp_backward(
            obs.astype(np.float32),
            self.shared_w, self.shared_b,
            self.actor1_w, self.actor1_b,
            self.actor2_w, self.actor2_b,
            self.actor_head_w, self.actor_head_b,
            self.critic1_w, self.critic1_b,
            self.critic2_w, self.critic2_b,
            self.critic_head_w, self.critic_head_b,
            grad_actor_output.astype(np.float32),
            grad_critic_output.astype(np.float32),
            obs.shape[0], self.obs_dim, self.hidden_dim, self.n_actions
        )
        return grads

    def act(self, obs):
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        actor_out, critic_out = self.forward(obs)
        logits = actor_out
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        action = np.array([np.random.choice(self.n_actions, p=p) for p in probs])
        value = critic_out.squeeze()
        actLogProbs = np.log(probs[np.arange(len(action)), action] + 1e-8)
        return action, value, actLogProbs