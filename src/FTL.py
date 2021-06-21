import torch


class FTL:
    def __init__(self, T, n_arms, X=None):
        self.T = T
        self.n_arms = n_arms
        self.X = X
        self.num_pulls = torch.zeros(n_arms)
        self.total_rewards = torch.zeros(n_arms)
        self.t = 1
        self.aggregate_reward = torch.zeros(T + 1)
        self.actions = torch.zeros(T + 1)

    def choose_arm_and_observe(self):
        if self.t in range(1, self.n_arms + 1):
            reward = self.X[self.t - 1][self.t - 1]
            self.num_pulls[self.t - 1] += 1
            self.total_rewards[self.t - 1] += reward
            self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
            self.actions[self.t] = self.t - 1
            self.t += 1
        else:
            empirical_means = self.total_rewards / self.num_pulls
            A = torch.argmax(empirical_means)
            reward = self.X[self.t - 1][A]
            self.num_pulls[A] += 1
            self.total_rewards[A] += reward
            self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
            self.actions[self.t] = A
            self.t += 1

    def choose_arm(self):
        if self.t in range(1, self.n_arms + 1):
            arm = self.t - 1
            return arm
        else:
            empirical_means = self.total_rewards / self.num_pulls
            arm = torch.argmax(empirical_means)
            return arm

    def observe(self, arm, reward):
        self.num_pulls[arm] += 1
        self.total_rewards[arm] += reward
        self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
        self.actions[self.t] = arm
        self.t += 1
