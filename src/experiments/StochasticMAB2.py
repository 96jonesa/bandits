import torch
from src.algorithms.EXP3 import EXP3
from src.algorithms.EXP3IX import EXP3IX
from src.algorithms.FTL import FTL
from src.algorithms.UCBDelta import UCBDelta
from src.algorithms.TSBeta import TSBeta
from src.utils import get_regret
from src.utils import plot
import math


def main():
    # problem definition
    n_arms = 10
    T = 10000
    mu = torch.empty((T, n_arms))
    for t in range(T):
        for i in range(n_arms):
            mu[t][i] = 0.25 + 0.5 * math.sqrt(i / (n_arms - 1))
    X = 1.0 - torch.bernoulli(mu)

    # algorithm parameters
    eta_exp3 = math.sqrt(math.log(n_arms) / (n_arms * T))
    eta_exp3ix = math.sqrt(2 * math.log(n_arms + 1) / (T * n_arms))
    gamma_exp3ix = math.sqrt(2 * math.log(n_arms + 1) / (T * n_arms)) / 2

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
    regrets = {'exp3': get_regret(exp3.aggregate_reward, X, 0),
               'exp3ix': get_regret(exp3ix.aggregate_reward, X, 0),
               'ftl': get_regret(ftl.aggregate_reward, X, 0),
               'ucb_delta': get_regret(ucb_delta.aggregate_reward, X, 0),
               'ts_beta': get_regret(ts_beta.aggregate_reward, X, 0)}

    # plot regrets
    plot(range(T + 1),
         [regrets['exp3'],
          regrets['exp3ix'],
          regrets['ftl'],
          regrets['ucb_delta'],
          regrets['ts_beta']],
         '$t =$number of pulls',
         'cumulative regret',
         ['EXP3', 'EXP3-IX', 'Follow-the-Leader', 'UCB', 'Thompson Sampling'],
         'Regret Over Time')


if __name__ == "__main__":
    main()
