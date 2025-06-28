import numpy as np

class Traj:
    def __init__(self, max_steps, gamma=0.99):
        self.max_steps = max_steps
        self.gamma_ = gamma
        self.observations_ = []
        self.actions_ = []
        self.rewards_ = []
        self.action_log_probs_ = []
        self.values_ = []
        self.dones_ = []
        self.returns_ = []
        self.current_step_ = 0
        self.frozen_ = False

    def remember(self, observation, action, reward, action_log_prob=None, value=None, done=None):
        if self.frozen_:
            raise RuntimeError("Cannot remember to a frozen trajectory")
        if self.current_step_ >= self.max_steps:
            raise RuntimeError("Trajectory buffer is full")
        self.observations_.append(np.array(observation, copy=True))
        self.actions_.append(np.array(action, copy=True))
        self.rewards_.append(np.array(reward, copy=True))
        if action_log_prob is not None:
            self.action_log_probs_.append(np.array(action_log_prob, copy=True))
        if value is not None:
            self.values_.append(np.array(value, copy=True))
        if done is not None:
            self.dones_.append(np.array(done, copy=True))
        self.current_step_ += 1

    def clear(self):
        self.observations_.clear()
        self.actions_.clear()
        self.rewards_.clear()
        self.action_log_probs_.clear()
        self.values_.clear()
        self.dones_.clear()
        self.returns_.clear()
        self.current_step_ = 0
        self.frozen_ = False

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
            return np.array([])
        return np.stack(self.observations_)

    def get_actions(self):
        if not self.actions_:
            return np.array([])
        return np.array(self.actions_).squeeze()

    def get_rewards(self):
        if not self.rewards_:
            return np.array([])
        return np.array(self.rewards_).squeeze()

    def get_action_log_probs(self):
        if not self.action_log_probs_:
            return np.array([])
        return np.array(self.action_log_probs_).squeeze()

    def get_values(self):
        if not self.values_:
            return np.array([])
        return np.array(self.values_).squeeze()

    def get_dones(self):
        if not self.dones_:
            return np.array([])
        return np.array(self.dones_).squeeze()

    def get_returns(self):
        if not self.returns_:
            return np.array([])
        return np.array(self.returns_).squeeze()

    def cut_tail(self):
        traj_length = self.current_step_
        self.observations_ = self.observations_[:traj_length]
        self.actions_ = self.actions_[:traj_length]
        self.rewards_ = self.rewards_[:traj_length]
        self.action_log_probs_ = self.action_log_probs_[:traj_length]
        self.values_ = self.values_[:traj_length]
        self.dones_ = self.dones_[:traj_length]

    def compute_returns(self):
        if self.current_step_ == 0:
            raise RuntimeError("Trajectory is empty. Cannot compute returns.")
        traj_len = len(self.rewards_)
        returns = [0] * traj_len
        running_return = 0.0
        for i in range(traj_len - 1, -1, -1):
            running_return = self.rewards_[i] + self.gamma_ * running_return
            returns[i] = running_return
        self.returns_ = returns