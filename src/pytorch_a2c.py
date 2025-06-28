import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MlpACPyTorch(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.shared = nn.Linear(obs_dim, hidden_dim)
        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

def pytorch_a2c_train():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")  # For simplicity, using CPU
    obs_dim, n_actions, hidden_dim = 2, 2, 128
    model = MlpACPyTorch(obs_dim, n_actions, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    gamma = 0.99
    N_episode = 1000
    episode_steps = 5

    for e in range(N_episode):
        obs = torch.from_numpy(np.random.randint(0, 10, (1, obs_dim)).astype(np.float32)).to(device)
        traj = {"obs": [], "actions": [], "rewards": [], "values": [], "log_probs": []}
        for step in range(episode_steps):
            logits, value = model(obs)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            reward = (action.item() == torch.argmax(obs[0]).item())
            traj["obs"].append(obs)
            traj["actions"].append(action)
            traj["rewards"].append(reward)
            traj["values"].append(value)
            traj["log_probs"].append(dist.log_prob(action))
            obs = torch.from_numpy(np.random.randint(0, 10, (1, obs_dim)).astype(np.float32)).to(device)
        # 计算returns
        returns = []
        R = 0
        for r in reversed(traj["rewards"]):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)
        values = torch.cat(traj["values"])
        log_probs = torch.stack(traj["log_probs"])
        advantage = returns - values.detach()
        value_loss = (advantage ** 2).mean()
        action_loss = -(advantage.detach() * log_probs).mean()
        entropy = -log_probs.mean()
        loss = value_loss + action_loss - 0.01 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Episode {e}, reward: {sum(traj['rewards'])}, loss: {loss.item():.4f}")

if __name__ == "__main__":
    pytorch_a2c_train()