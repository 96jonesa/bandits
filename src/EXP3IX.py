import torch


def sample(distribution):
    rng = torch.rand(1)
    sum = 0.0

    for i in range(len(distribution)):
        sum += distribution[i]

        if sum > rng:
            return i


class EXP3IX:
    def __init__(self, T, n_arms, eta, gamma, X=None):
        self.T = T
        self.n_arms = n_arms
        self.eta = eta
        self.gamma = gamma
        self.X = X
        self.L = torch.zeros(T + 1, n_arms)
        self.P = torch.zeros(T + 1, n_arms)
        self.t = 1
        self.aggregate_reward = torch.zeros(T + 1)
        self.actions = torch.zeros(T + 1)

    def choose_arm_and_observe(self):
        self.P[self.t] = torch.softmax(-1.0 * self.eta * self.L[self.t - 1], dim=0)
        A = sample(self.P[self.t])
        reward = self.X[self.t - 1][A]
        self.L[self.t] = self.L[self.t - 1]
        self.L[self.t][A] += (1 - reward) / (self.P[self.t][A] + self.gamma)
        self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
        self.actions[self.t] = A

    def choose_arm(self):
        self.P[self.t] = torch.softmax(-1.0 * self.eta * self.L[self.t - 1], dim=0)
        arm = sample(self.P[self.t])
        return arm

    def observe(self, arm, reward):
        self.L[self.t] = self.L[self.t - 1]
        self.L[self.t][arm] += (1 - reward) / (self.P[self.t][arm] + self.gamma)
        self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
        self.actions[self.t] = arm
