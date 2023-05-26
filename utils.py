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

class History :
    def __init__(self):
        self.keys = ['state', 'action', 'reward', 'utility']
        self.history = dict.fromkeys(self.keys)

    def push(self, state, action, reward, utility):
        if self.history['reward'] is not None:
            self.history['state'] = np.vstack((self.history['state'], state))
            self.history['action'] = np.vstack((self.history['action'], action))
            self.history['reward'] = np.vstack((self.history['reward'], reward))
            self.history['utility'] = np.vstack((self.history['utility'], utility))
        else :
            self.history['state'] = state
            self.history['action'] = action
            self.history['reward'] = reward
            self.history['utility'] = utility

    def sample(self, batch_size):
        batch = np.random.choice(np.arange(len(self.history['reward'])), size = batch_size, replace = False)

        state_batch = self.history['state'][batch, :]
        utility_batch = self.history['utility'][batch, :]
        action_batch = self.history['action'][batch, :]
        reward_batch = self.history['reward'][batch, :]

        return state_batch, action_batch, reward_batch, utility_batch
