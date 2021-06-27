import matplotlib.pyplot as plt
import torch


def plot(X, Y, xlabel, ylabel, legend, title):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(len(Y)):
        ax.plot(X, Y[i], label=legend[i])

    plt.grid(color='0.95')
    plt.legend()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.show()


def get_regret(reward, X, best):
    X_max_tensor = X[:, best]
    max_reward = torch.zeros(len(reward))
    for t in range(1, len(reward)):
        max_reward[t] = max_reward[t - 1] + X_max_tensor[t - 1]

    return max_reward - reward
