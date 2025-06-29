import numpy as np
import mlp_ac_cuda

class MlpACManual:
    def __init__(self, obs_dim, n_actions, hidden_dim=64, cpu=True):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.cpu = cpu

        def xavier(shape):
            fan_in, fan_out = shape
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            return np.random.randn(*shape).astype(np.float32) * scale

        self.actor1_w = xavier((obs_dim, hidden_dim))
        self.actor1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.actor2_w = xavier((hidden_dim, hidden_dim))
        self.actor2_b = np.zeros(hidden_dim, dtype=np.float32)
        self.actor_head_w = xavier((hidden_dim, n_actions))
        self.actor_head_b = np.zeros(n_actions, dtype=np.float32)
        
        self.critic1_w = xavier((obs_dim, hidden_dim))
        self.critic1_b = np.zeros(hidden_dim, dtype=np.float32)
        self.critic2_w = xavier((hidden_dim, hidden_dim))
        self.critic2_b = np.zeros(hidden_dim, dtype=np.float32)
        self.critic_head_w = xavier((hidden_dim, 1))
        self.critic_head_b = np.zeros(1, dtype=np.float32)

        self.d_capsules = None 

        self.params = [
            self.actor1_w, self.actor1_b,
            self.actor2_w, self.actor2_b,
            self.actor_head_w, self.actor_head_b,
            self.critic1_w, self.critic1_b,
            self.critic2_w, self.critic2_b,
            self.critic_head_w, self.critic_head_b
        ]

    def parameters(self):
        return self.params

    def state_dict(self):
        return [p.copy() for p in self.params]

    def load_state_dict(self, state):
        for p, s in zip(self.params, state):
            np.copyto(p, s)

    def zero_grad(self):
        pass

    def forward(self, obs):
        if self.cpu: return self.forward_numpy(obs)
        result = mlp_ac_cuda.mlp_forward(
            obs.astype(np.float32),
            self.actor1_w, self.actor1_b,
            self.actor2_w, self.actor2_b,
            self.actor_head_w, self.actor_head_b,
            self.critic1_w, self.critic1_b,
            self.critic2_w, self.critic2_b,
            self.critic_head_w, self.critic_head_b,
            self.hidden_dim,
            self.n_actions
        )
        actor_out, critic_out, cap1, cap2, cap3, cap4 = result
        self.d_capsules = (cap1, cap2, cap3, cap4)
        return actor_out, critic_out

    def backward(self, obs, grad_actor_output, grad_critic_output):
        cap1, cap2, cap3, cap4 = self.d_capsules
        grads = mlp_ac_cuda.mlp_backward(
            obs.astype(np.float32),
            self.actor1_w, self.actor1_b,
            self.actor2_w, self.actor2_b,
            self.actor_head_w, self.actor_head_b,
            self.critic1_w, self.critic1_b,
            self.critic2_w, self.critic2_b,
            self.critic_head_w, self.critic_head_b,
            cap1, cap2, cap3, cap4,
            grad_actor_output.astype(np.float32),
            grad_critic_output.astype(np.float32),
            obs.shape[0],        # batch_size
            self.obs_dim,        # input_dim
            self.hidden_dim,     # hidden_dim
            self.n_actions,      # actor_output_dim
        )
        return grads
    
    def forward_numpy(self, obs):
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
            
        # Actor forward pass
        actor_h1 = np.maximum(0, obs @ self.actor1_w + self.actor1_b)
        actor_h2 = np.maximum(0, actor_h1 @ self.actor2_w + self.actor2_b)
        actor_out = actor_h2 @ self.actor_head_w + self.actor_head_b
        
        # Critic forward pass
        critic_h1 = np.maximum(0, obs @ self.critic1_w + self.critic1_b)
        critic_h2 = np.maximum(0, critic_h1 @ self.critic2_w + self.critic2_b)
        critic_out = critic_h2 @ self.critic_head_w + self.critic_head_b
        
        return actor_out, critic_out

    def act(self, obs):
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        actor_out, critic_out = self.forward(obs)
        print(actor_out)
        print(critic_out)
        logits = actor_out
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        action = np.array([np.random.choice(self.n_actions, p=p) for p in probs])
        value = critic_out.squeeze()
        actLogProbs = np.log(probs[np.arange(len(action)), action] + 1e-8)
        return action, value, actLogProbs
    
    def free_intermediate(self):
        if self.d_capsules is not None:
            mlp_ac_cuda.mlp_free_intermediate(*self.d_capsules)
            self.d_capsules = None

import torch.nn as nn
import torch

class MlpACTorch(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super(MlpACTorch, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化权重
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

    def act(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        with torch.no_grad():
            logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).numpy()
        value = value.squeeze().numpy()
        act_log_prob = torch.log(probs.gather(1, torch.LongTensor(action))).numpy()
        return action, value, act_log_prob