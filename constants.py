"""
All constants useful for the UCB algorithm
"""

#import numpy.random as rd
import numpy as np
import scipy.stats as st

# number of arms
K = 3

# total time
T = 100000

# multiplicative constant in the UCB bias (\sqrt{UCB_constant * \log(t) / s})
UCB_constant = 1.5

# set of arms
ARMS = [0] * K

# set of means
MEANS = [0] * K

# choice of the arms' distributions
# ARMS[0] = st.truncnorm(0, 5)
ARMS[0] = st.expon(scale = 1. / 2.5)
ARMS[1] = st.bernoulli(0.3)
ARMS[2] = st.poisson(0.5)
# ARMS[0] = st.randint(0, 1)  # constant = 0
# ARMS[1] = st.randint(1, 2)  # constant = 1


# true means
for k in range(K):
    MEANS[k] = ARMS[k].stats(moments='m').tolist()

# Best arm and best mean
BEST_ARM = np.argmax(np.array(MEANS))
BEST_MEAN = np.max(np.array(MEANS))

# Delta
DELTAS = np.array([BEST_MEAN] * K) - np.array(MEANS)
