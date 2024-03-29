import torch


def sample(distribution):
    rng = torch.rand(1)
    sum = 0.0

    for i in range(len(distribution)):
        sum += distribution[i]

        if sum > rng:
            return i


class EXP3:
    def __init__(self, T, n_arms, eta, X=None):
        self.T = T
        self.n_arms = n_arms
        self.eta = eta
        self.X = X
        self.S = torch.zeros(T + 1, n_arms)
        self.P = torch.zeros(T + 1, n_arms)
        self.t = 1
        self.aggregate_reward = torch.zeros(T + 1)
        self.actions = torch.zeros(T + 1)

    def choose_arm_and_observe(self):
        self.P[self.t] = torch.softmax(self.eta * self.S[self.t - 1], dim=0)
        arm = sample(self.P[self.t])
        reward = self.X[self.t - 1][arm]
        self.S[self.t] = self.S[self.t - 1] + 1
        self.S[self.t][arm] -= (1 - reward) / self.P[self.t][arm]
        self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
        self.actions[self.t] = arm
        self.t += 1

    def choose_arm(self):
        self.P[self.t] = torch.softmax(self.eta * self.S[self.t - 1], dim=0)
        arm = sample(self.P[self.t])
        return arm

    def observe(self, arm, reward):
        self.S[self.t] = self.S[self.t - 1] + 1
        self.S[self.t][arm] -= (1 - reward) / self.P[self.t][arm]
        self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
        self.actions[self.t] = arm
        self.t += 1
