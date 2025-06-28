# numpy_a2c.py
import numpy as np

class SimpleMLP:
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        self.w1 = np.random.randn(obs_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2a = np.random.randn(hidden_dim, n_actions) * 0.1
        self.b2a = np.zeros(n_actions)
        self.w2c = np.random.randn(hidden_dim, 1) * 0.1
        self.b2c = np.zeros(1)

    def forward(self, x):
        h = np.maximum(0, x @ self.w1 + self.b1)
        logits = h @ self.w2a + self.b2a
        value = h @ self.w2c + self.b2c
        return logits, value, h

    def parameters(self):
        return [self.w1, self.b1, self.w2a, self.b2a, self.w2c, self.b2c]

def numpy_a2c_train():
    obs_dim, n_actions, hidden_dim = 2, 2, 128
    model = SimpleMLP(obs_dim, n_actions, hidden_dim)
    lr = 3e-4
    gamma = 0.99
    N_episode = 100
    episode_steps = 20

    for e in range(N_episode):
        obs = np.random.randint(0, 10, (1, obs_dim)).astype(np.float32)
        traj = {"obs": [], "actions": [], "rewards": [], "values": [], "log_probs": [], "h": []}
        for step in range(episode_steps):
            logits, value, h = model.forward(obs)
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            action = np.random.choice(n_actions, p=probs.ravel())
            reward = float(action == np.argmax(obs[0]))
            traj["obs"].append(obs)
            traj["actions"].append(action)
            traj["rewards"].append(reward)
            traj["values"].append(value)
            traj["log_probs"].append(np.log(probs[0, action] + 1e-8))
            traj["h"].append(h)
            obs = np.random.randint(0, 10, (1, obs_dim)).astype(np.float32)
        # 计算returns
        returns = []
        R = 0
        for r in reversed(traj["rewards"]):
            R = r + gamma * R
            returns.insert(0, R)
        returns = np.array(returns).reshape(-1, 1)
        values = np.vstack(traj["values"])
        log_probs = np.array(traj["log_probs"]).reshape(-1, 1)
        advantage = returns - values
        value_loss = (advantage ** 2).mean()
        action_loss = -(advantage * log_probs).mean()
        loss = value_loss + action_loss
        # 简单SGD更新
        for p in model.parameters():
            p -= lr * np.random.randn(*p.shape) * 0.01  # 这里只是示例，实际应反向传播
        print(f"Episode {e}, reward: {sum(traj['rewards'])}, loss: {loss:.4f}")

if __name__ == "__main__":
    numpy_a2c_train()