import torch


def sample(distribution):
    rng = torch.rand(1)
    sum = 0.0

    for i in range(len(distribution)):
        sum += distribution[i]

        if sum > rng:
            return i


# TODO: make this faster and/or more accurate
def get_P(n_arms, est_losses, eta, start):
    P = torch.zeros(1, n_arms)

    change = 0.0

    for i in range(n_arms):
        P[0][i] = eta * torch.sum(est_losses[:, i]) + start

    if torch.sum(P ** (-2)) == 1.0:
        return P
    elif torch.sum(P ** (-2)) < 1.0:
        while torch.sum(P ** (-2)) < 1.0:
            P -= eta * 0.1
            change -= eta * 0.1
    else:
        while torch.sum(P ** (-2)) > 1.0:
            P += eta * 0.1
            change += eta * 0.1
        P -= eta * 0.1
        change -= eta * 0.1

    return P, change


class FTRLTsallisINF:
    def __init__(self, T, n_arms, eta, X=None):
        self.T = T
        self.n_arms = n_arms
        self.eta = eta
        self.X = X
        self.est_losses = torch.zeros(T + 1, n_arms)
        self.P = torch.zeros(1, n_arms)
        self.t = 1
        self.aggregate_reward = torch.zeros(T + 1)
        self.actions = torch.zeros(T + 1)
        self.start = 0.0

        for i in range(n_arms):
            self.P[0][i] = 1.0 / n_arms

    def choose_arm_and_observe(self):
        if self.t == 1:
            A = sample(self.P[0])
            reward = self.X[0][A]
            self.est_losses[1][A] = (1.0 - reward) / self.P[0][A]
            self.actions[1] = A
            self.aggregate_reward[1] = reward
            self.t += 1
        else:
            self.P, self.start = get_P(self.est_losses, self.eta, self.start)
            self.P = self.P ** (-2)
            A = sample(self.P[0])
            reward = self.X[self.t - 1][A]
            self.est_losses[self.t][A] = (1.0 - reward) / self.P[0][A]
            self.actions[self.t] = A
            self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
            self.t += 1

    def choose_arm(self):
        if self.t == 1:
            arm = sample(self.P[0])
            return arm
        else:
            self.P, self.start = get_P(self.est_losses, self.eta, self.start)
            self.P = self.P ** (-2)
            arm = sample(self.P[0])
            return arm

    def observe(self, arm, reward):
        self.est_losses[self.t][arm] = (1.0 - reward) / self.P[0][arm]
        self.actions[self.t] = arm
        self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
        self.t += 1
