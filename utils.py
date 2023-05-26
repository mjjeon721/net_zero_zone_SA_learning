import numpy as np
from scipy.stats import truncnorm

class Env:
    '''
        Environment model. Modeling customer. Generate next state and compute reward value for current state and action.
        Customer utility function : Quadratic utility function
        Renewable distribution : Truncated normal distribution.
    '''
    def __init__(self, util_param, renewable_param):
        self.a = util_param[0]
        self.b = util_param[1]

        self.g_mean = renewable_param[0]
        self.g_std = renewable_param[1]
        self.g_low = renewable_param[2]
        self.g_high = renewable_param[3]

    def get_next_state(self) :
        low = (self.g_low - self.g_mean) / self.g_std
        high = (self.g_high - self.g_mean) / self.g_std
        return truncnorm.rvs(low, high, loc = self.g_mean, scale = self.g_std, size = 1)

    def get_reward(self, state, action):
        net_cons = np.sum(action) - state[0]
        reward = np.matmul(self.a, action) - 0.5 * np.matmul(self.b, action ** 2) - np.max(state[1:] * net_cons)
        return reward

