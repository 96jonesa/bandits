import torch
import math


class UCBDelta:
    def __init__(self, T, n_arms, delta, X=None):
        self.T = T
        self.n_arms = n_arms
        self.X = X
        self.delta = delta
        self.num_pulls = torch.zeros(n_arms)
        self.total_rewards = torch.zeros(n_arms)
        self.t = 1
        self.aggregate_reward = torch.zeros(T + 1)
        self.actions = torch.zeros(T + 1)
        self.empirical_means = torch.zeros(n_arms)

    def choose_arm_and_observe(self):
        if self.t in range(1, self.n_arms + 1):
            reward = self.X[self.t - 1][self.t - 1]
            self.total_rewards[self.t - 1] += reward
            self.num_pulls[self.t - 1] += 1
            self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
            self.actions[self.t] = self.t - 1
            self.empirical_means[self.t - 1] = reward
            self.t += 1
        else:
            I = torch.argmax(self.empirical_means + torch.sqrt(2 * math.log(1 / self.delta) / self.num_pulls))
            reward = self.X[self.t - 1][I]
            self.total_rewards[I] += reward
            self.num_pulls[I] += 1
            self.empirical_means[I] = self.total_rewards[I] / self.num_pulls[I]
            self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
            self.actions[self.t] = I
            self.t += 1

    def choose_arm(self):
        if self.t in range(1, self.n_arms + 1):
            arm = self.t - 1
            return arm
        else:
            arm = torch.argmax(self.empirical_means + torch.sqrt(2 * math.log(1 / self.delta) / self.num_pulls))
            return arm

    def observe(self, arm, reward):
        self.total_rewards[arm] += reward
        self.num_pulls[arm] += 1
        self.empirical_means[arm] = self.total_rewards[arm] / self.num_pulls[arm]
        self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
        self.actions[self.t] = arm
        self.t += 1
