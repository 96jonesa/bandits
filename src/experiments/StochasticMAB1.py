import torch
from src.algorithms.EXP3 import EXP3
from src.algorithms.EXP3IX import EXP3IX
from src.algorithms.FTL import FTL
from src.algorithms.UCBDelta import UCBDelta
from src.algorithms.TSBeta import TSBeta
from src.utils import get_regret
from src.utils import plot
import math
import numpy as np


def main():
    # problem definition
    n_arms = 10
    T = 10000

    # algorithm parameters
    eta_exp3 = math.sqrt(math.log(n_arms) / (n_arms * T))
    eta_exp3ix = math.sqrt(2 * math.log(n_arms + 1) / (T * n_arms))
    gamma_exp3ix = math.sqrt(2 * math.log(n_arms + 1) / (T * n_arms)) / 2

    iters = 30
    regrets = {'exp3': torch.zeros(iters, T + 1),
               'exp3ix': torch.zeros(iters, T + 1),
               'ftl': torch.zeros(iters, T + 1),
               'ucb_delta': torch.zeros(iters, T + 1),
               'ts_beta': torch.zeros(iters, T + 1)}

    for i in range(iters):
        mu = torch.empty((T, n_arms)).fill_(0.75)
        for t in range(T):
            mu[t][0] = 0.25
        X = 1.0 - torch.bernoulli(mu)

        # algorithms
        exp3 = EXP3(T, n_arms, eta_exp3, X)
        exp3ix = EXP3IX(T, n_arms, eta_exp3ix, gamma_exp3ix, X)
        ftl = FTL(T, n_arms, X)
        ucb_delta = UCBDelta(T, n_arms, 0.1, X)
        ts_beta = TSBeta(T, n_arms, X)

        # explore and exploit
        for t in range(T):
            exp3.choose_arm_and_observe()
            exp3ix.choose_arm_and_observe()
            ftl.choose_arm_and_observe()
            ucb_delta.choose_arm_and_observe()
            ts_beta.choose_arm_and_observe()

        # compute regrets
        regrets['exp3'][i] = get_regret(exp3.aggregate_reward, X, 0)
        regrets['exp3ix'][i] = get_regret(exp3ix.aggregate_reward, X, 0)
        regrets['ftl'][i] = get_regret(ftl.aggregate_reward, X, 0)
        regrets['ucb_delta'][i] = get_regret(ucb_delta.aggregate_reward, X, 0)
        regrets['ts_beta'][i] = get_regret(ts_beta.aggregate_reward, X, 0)

        print(str(i + 1), '/', str(iters), 'iterations done')

    plot(range(T + 1),
         [torch.min(regrets['exp3'], 0)[0].numpy(),
          torch.max(regrets['exp3'], 0)[0].numpy(),
          torch.mean(regrets['exp3'], 0).numpy()],
         '$t =$number of pulls',
         'cumulative regret',
         ['minimum', 'maximum', 'mean'],
         'EXP3 Regret Over Time (' + str(iters) + 'Iterations)')

    plot(range(T + 1),
         [torch.min(regrets['exp3ix'], 0)[0].numpy(),
          torch.max(regrets['exp3ix'], 0)[0].numpy(),
          torch.mean(regrets['exp3ix'], 0).numpy()],
         '$t =$number of pulls',
         'cumulative regret',
         ['minimum', 'maximum', 'mean'],
         'EXP3-IX Regret Over Time (' + str(iters) + 'Iterations)')

    plot(range(T + 1),
         [torch.min(regrets['ftl'], 0)[0].numpy(),
          torch.max(regrets['ftl'], 0)[0].numpy(),
          torch.mean(regrets['ftl'], 0).numpy()],
         '$t =$number of pulls',
         'cumulative regret',
         ['minimum', 'maximum', 'mean'],
         'Follow-the-Leader Regret Over Time (' + str(iters) + 'Iterations)')

    plot(range(T + 1),
         [torch.min(regrets['ucb_delta'], 0)[0].numpy(),
          torch.max(regrets['ucb_delta'], 0)[0].numpy(),
          torch.mean(regrets['ucb_delta'], 0).numpy()],
         '$t =$number of pulls',
         'cumulative regret',
         ['minimum', 'maximum', 'mean'],
         'UCB (delta) Regret Over Time (' + str(iters) + 'Iterations)')

    plot(range(T + 1),
         [torch.min(regrets['ts_beta'], 0)[0].numpy(),
          torch.max(regrets['ts_beta'], 0)[0].numpy(),
          torch.mean(regrets['ts_beta'], 0).numpy()],
         '$t =$number of pulls',
         'cumulative regret',
         ['minimum', 'maximum', 'mean'],
         'Thompson Sampling (beta) Regret Over Time (' + str(iters) + 'Iterations)')


if __name__ == "__main__":
    main()
