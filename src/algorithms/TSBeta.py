import torch


class TSBeta:
    def __init__(self, T, n_arms, X=None):
        self.T = T
        self.n_arms = n_arms
        self.X = X
        self.num_pulls = torch.zeros(n_arms)
        self.total_rewards = torch.zeros(n_arms)
        self.alpha = torch.ones(n_arms)
        self.beta = torch.ones(n_arms)
        self.t = 1
        self.aggregate_reward = torch.zeros(T + 1)
        self.actions = torch.zeros(T + 1)

    # alpha and beta updates from: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf (page 14)
    def choose_arm_and_observe(self):
        posterior = torch.distributions.beta.Beta(self.alpha, self.beta)  # compute posterior
        sample = posterior.sample()  # sample from posterior
        I = torch.argmax(sample)  # select arm with highest sample from posterior
        reward = self.X[self.t - 1][I]  # pull selected arm
        self.total_rewards[I] += reward  # increment total reward after pulling arm
        self.num_pulls[I] += 1  # increment total pulls after pulling arm
        self.alpha[I] += reward  # update posterior alpha
        self.beta[I] += 1 - reward  # update posterior beta
        self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
        self.actions[self.t] = I
        self.t += 1

    def choose_arm(self):
        posterior = torch.distributions.beta.Beta(self.alpha, self.beta)  # compute posterior
        sample = posterior.sample()  # sample from posterior
        arm = torch.argmax(sample)  # select arm with highest sample from posterior
        return arm

    # alpha and beta updates from: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf (page 14)
    def observe(self, arm, reward):
        self.total_reward[arm] += reward  # increment total reward after pulling arm
        self.num_pulls[arm] += 1  # increment total pulls after pulling arm
        self.alpha[arm] += reward  # update posterior alpha
        self.beta[arm] += 1 - reward  # update posterior beta
        self.aggregate_reward[self.t] = self.aggregate_reward[self.t - 1] + reward
        self.actions[self.t] = arm
        self.t += 1
