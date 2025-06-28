import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class A2CTorch:
    def __init__(self, actor_and_critic, value_loss_coef=1., actor_loss_coef=1., entropy_coef=1e-3, learning_rate=1e-3):
        self.actor_and_critic = actor_and_critic
        self.value_loss_coef = value_loss_coef
        self.actor_loss_coef = actor_loss_coef
        self.entropy_coef = entropy_coef
        self.optimizer = optim.Adam(actor_and_critic.parameters(), lr=learning_rate)
        # torch max thread
        torch.set_num_threads(24)

    def update(self, traj):
        traj.finalize()
        observations = torch.FloatTensor(traj.get_observations())
        actions = torch.LongTensor(traj.get_actions()).unsqueeze(1)  # Add dimension for gather
        returns = torch.FloatTensor(traj.get_returns())
        
        # 前向传播
        logits, values = self.actor_and_critic(observations)
        probs = torch.softmax(logits, dim=-1)
        act_log_probs = torch.log(probs.gather(1, actions))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
        
        # 计算优势
        advantages = returns.unsqueeze(1) - values
        
        # 计算损失
        value_loss = advantages.pow(2).mean() * self.value_loss_coef
        action_loss = -(advantages.detach() * act_log_probs).mean() * self.actor_loss_coef
        entropy_loss = -entropy.mean() * self.entropy_coef
        loss = value_loss + action_loss + entropy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_and_critic.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            "Value loss": value_loss.item(),
            "Action loss": action_loss.item(),
            "distEntropy loss": entropy_loss.item(),
            "Loss": loss.item(),
            "total_episode_reward": np.sum(traj.get_rewards()),
            "mean_episode_reward": np.mean(traj.get_rewards()),
            "num_steps": len(observations)
        }
