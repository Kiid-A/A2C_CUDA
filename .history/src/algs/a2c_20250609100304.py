import torch
import torch.optim as optim

class A2C:
    def __init__(self, actor_and_critic, value_loss_coef=1., actor_loss_coef=1., entropy_coef=0.1, learning_rate=1e-3):
        self.actor_and_critic = actor_and_critic
        self.value_loss_coef = value_loss_coef
        self.actor_loss_coef = actor_loss_coef
        self.entropy_coef = entropy_coef
        self.original_learning_rate = learning_rate
        self.optimizer = optim.SGD(actor_and_critic.parameters(), lr=learning_rate)

    def update(self, traj):
        print("traj finalize...")
        traj.finalize()
        observations = traj.get_observations()
        actions = traj.get_actions()
        rewards = traj.get_rewards()
        returns = traj.get_returns()

        reward_dim = rewards.size(-1)
        assert reward_dim == 1
        total_episode_reward = rewards.sum().item()
        mean_episode_reward = total_episode_reward / rewards.size(0)
        num_steps = traj.current_step()
        v_logp_e_p = self.actor_and_critic.evaluate_actions(observations, actions)

        distEntropy = v_logp_e_p[2].view(1, num_steps, 1)
        values = v_logp_e_p[0].view(1, num_steps, 1)
        actLogProbs = v_logp_e_p[1].view(1, num_steps, 1)
        advantages = returns.view(1, num_steps, 1) - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * actLogProbs).mean()
        distEntropy_loss = -distEntropy.mean()

        loss = (value_loss * self.value_loss_coef + action_loss * self.actor_loss_coef + distEntropy_loss * self.entropy_coef)
        print("Calculating gradients...")
        self.optimizer.zero_grad()
        loss.backward()
        print("Updating parameters...")
        self.optimizer.step()
        return {
            "Value loss": value_loss.item(),
            "Action loss": action_loss.item(),
            "distEntropy loss": distEntropy_loss.item(),
            "Loss": loss.item(),
            "toatal_episode_reward": total_episode_reward,
            "mean_episode_reward": mean_episode_reward,
            "num_steps": num_steps
        }