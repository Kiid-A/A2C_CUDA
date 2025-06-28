import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class MlpAC(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super(MlpAC, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Relu()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
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
        actLogProbs = dist.log_prob(action)
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
        return value, actLogProbs, distEntropy, probs