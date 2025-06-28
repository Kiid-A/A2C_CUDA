import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class MlpAC(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super(MlpAC, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.actor_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs):
        if obs.dim() != 2:
            raise RuntimeError("Input tensor must have 2 dimensions")
        input_obs_dim = obs.size(1)
        batch_size = obs.size(0)
        assert input_obs_dim == self.shared[0].in_features
        x = self.shared(obs)
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        value = self.critic_head(hidden_critic)
        logits = self.actor_head(hidden_actor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        actLogProbs = dist.log_prob(actig'g
        return action.float(), value, actLogProbs

    def act(self, obs):
        if not obs.is_floating_point():
            obs = obs.float()
        return self.forward(obs)

    def evaluate_actions(self, obs, actions):
        x = self.shared(obs)
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        value = self.critic_head(hidden_critic)
        logits = self.actor_head(hidden_actor)
        dist = Categorical(logits=logits)
        actLogProbs = dist.log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
        distEntropy = dist.entropy().unsqueeze(1)
        probs = dist.probs
        import numpy as np

class MlpACManual:
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Xavier初始化函数
        def xavier(shape):
            fan_in, fan_out = shape[0], shape[1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
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

    def relu(self, x):
        return np.maximum(x, 0)

    def forward(self, obs):
        # obs: [batch, obs_dim] numpy array
        x = obs @ self.shared_w + self.shared_b
        x = self.relu(x)

        # actor
        a = x @ self.actor1_w + self.actor1_b
        a = self.relu(a)
        a = a @ self.actor2_w + self.actor2_b
        a = self.relu(a)
        logits = a @ self.actor_head_w + self.actor_head_b  # [batch, n_actions]

        # critic
        c = x @ self.critic1_w + self.critic1_b
        c = self.relu(c)
        c = c @ self.critic2_w + self.critic2_b
        c = self.relu(c)
        value = c @ self.critic_head_w + self.critic_head_b  # [batch, 1]

        # 采样动作
        logits = logits - np.max(logits, axis=1, keepdims=True)  # 防止溢出
        probs = np.exp(logits)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        action = np.array([np.random.choice(self.n_actions, p=p) for p in probs])
        actLogProbs = np.log(probs[np.arange(len(action)), action] + 1e-8)

        return action.astype(np.float32), value, actLogProbs

    def act(self, obs):
        # obs: [batch, obs_dim] numpy array
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        return self.forward(obs)

    def evaluate_actions(self, obs, actions):
        # obs: [batch, obs_dim], actions: [batch, 1] or [batch]
        x = obs @ self.shared_w + self.shared_b
        x = self.relu(x)

        # critic
        c = x @ self.critic1_w + self.critic1_b
        c = self.relu(c)
        c = c @ self.critic2_w + self.critic2_b
        c = self.relu(c)
        value = c @ self.critic_head_w + self.critic_head_b  # [batch, 1]

        # actor
        a = x @ self.actor1_w + self.actor1_b
        a = self.relu(a)
        a = a @ self.actor2_w + self.actor2_b
        a = self.relu(a)
        logits = a @ self.actor_head_w + self.actor_head_b  # [batch, n_actions]
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        if actions.ndim == 2:
            actions = actions.squeeze(-1)
        actLogProbs = np.log(probs[np.arange(len(actions)), actions] + 1e-8)
        distEntropy = -np.sum(probs * np.log(probs + 1e-8), axis=1, keepdims=True)
        return value, actLogProbs[:, None], distEntropy,return value, actLogProbs, distEntropy, probs