import torch

class Traj:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.device = torch.device("cpu")
        self.observations_ = []
        self.actions_ = []
        self.rewards_ = []
        self.action_log_probs_ = []
        self.values_ = []
        self.dones_ = []
        self.returns_ = []
        self.gamma_ = 0.99
        self.current_step_ = 0
        self.frozen_ = False

    def remember(self, observation, action, reward, action_log_prob=None, value=None, done=None):
        if self.frozen_:
            raise RuntimeError("Cannot remember to a frozen trajectory")
        if self.current_step_ >= self.max_steps:
            raise RuntimeError("Trajectory buffer is full")
        self.observations_.append(observation.to(self.device))
        self.actions_.append(action.to(self.device))
        self.rewards_.append(reward.to(self.device))
        if action_log_prob is not None:
            self.action_log_probs_.append(action_log_prob.to(self.device))
        if value is not None:
            self.values_.append(value.to(self.device))
        if done is not None:
            self.dones_.append(done.to(self.device))
        self.current_step_ += 1

    def clear(self):
        self.observations_.clear()
        self.actions_.clear()
        self.rewards_.clear()
        self.action_log_probs_.clear()
        self.values_.clear()
        self.dones_.clear()
        self.current_step_ = 0
        self.frozen_ = False

    def current_step(self):
        return self.current_step_

    def max_steps(self):
        return self.max_steps

    def freeze(self):
        self.frozen_ = True

    def finalize(self):
        if self.frozen_:
            raise RuntimeError("Trajectory frozen. Cannot finalize.")
        self.cut_tail()
        self.compute_returns()
        self.freeze()

    def get_observations(self):
        if not self.observations_:
            return torch.tensor([])
        return torch.stack(self.observations_)

    def get_actions(self):
        if not self.actions_:
            return torch.tensor([])
        return torch.stack(self.actions_)

    def get_rewards(self):
        if not self.rewards_:
            return torch.tensor([])
        return torch.stack(self.rewards_)

    def get_action_log_probs(self):
        if not self.action_log_probs_:
            return torch.tensor([])
        return torch.stack(self.action_log_probs_)

    def get_values(self):
        if not self.values_:
            return torch.tensor([])
        return torch.stack(self.values_)

    def get_dones(self):
        if not self.dones_:
            return torch.tensor([])
        return torch.stack(self.dones_)

    def get_returns(self):
        if not self.returns_:
            return torch.tensor([])
        return torch.stack(self.returns_)

    def cut_tail(self):
        traj_length = self.current_step_
        if len(self.observations_) > traj_length:
            self.observations_ = self.observations_[:traj_length]
        if len(self.actions_) > traj_length:
            self.actions_ = self.actions_[:traj_length]
        if len(self.rewards_) > traj_length:
            self.rewards_ = self.rewards_[:traj_length]
        if len(self.action_log_probs_) > traj_length:
            self.action_log_probs_ = self.action_log_probs_[:traj_length]
        if len(self.values_) > traj_length:
            self.values_ = self.values_[:traj_length]
        if len(self.dones_) > traj_length:
            self.dones_ = self.dones_[:traj_length]

    def compute_returns(self):
        if self.current_step_ == 0:
            raise RuntimeError("Trajectory is empty. Cannot compute returns.")
        traj_len = len(self.rewards_)
        assert traj_len == self.current_step_
        returns = [torch.zeros_like(self.rewards_[0])] * traj_len
        running_return = torch.zeros_like(self.rewards_[0])
        for i in range(traj_len - 1, -1, -1):
            running_return = self.rewards_[i] + self.gamma_ * running_return
            returns[i] = running_return
        self.returns_ = returns