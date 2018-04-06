"""
Main file for UCB algorithm
"""

from math import *

from constants import *

# array with the number of times each arm has been sampled
counters = [0] * K

# array of empirical means
emp_means = [0] * K

# Gain
gain = [0]

def new_empirical_mean(old_mean, old_nb, value):
    """
    Compute the new empirical mean of arm k when arm k is sampled again

    old_nb is the previous number of times arm k has been sampled
    """

    return (old_mean * old_nb + value) / (old_nb + 1)


def sample(k):
    """
    Sample arm k
    """

    return ARMS[k].rvs()


def update_arm(k):
    """
    Update arm k
    """

    val = sample(k)
    emp_means[k] = new_empirical_mean(emp_means[k], counters[k], val)
    counters[k] += 1
    gain[0] += val


def init():
    """
    Initialize by sampling each arm once
    """

    for k in range(K):
        update_arm(k)


def ucb_step(t):
    """
    Realize one UCB step
    """

    optimistic_means = np.array(emp_means) + np.sqrt(UCB_constant * log(t) / np.array(counters))
    k = np.argmax(optimistic_means)
    update_arm(k)


def ucb():
    """
    Run the full UCB algorithm
    """

    init()
    print(emp_means)
    for t in range(K + 1, T + 1):
        ucb_step(t)


def regret():
    """
    Return the regret
    """

    return BEST_MEAN * T - gain[0]

ucb()
R = regret()

print('K={:d}'.format(K))
print('Best arm: {:d} with mean {:.2f}'.format(BEST_ARM, BEST_MEAN))
print('Means: ', MEANS)
print('Counters: ', counters)
print('Regret={:.2f}'.format(R))
