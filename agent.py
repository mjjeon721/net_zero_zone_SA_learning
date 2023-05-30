from model import *
import numpy as np
import torch
import torch.optim as optim
import numpy.random as npr
from utils import *

class Agent():
    '''
        Learning Agent
        Learning threshold values and net-zero zone action network.
    '''
    def __init__(self, d_max, action_dim, env):
        self.d_max = d_max
        self.action_dim = action_dim

        # Thresholds policy module
        self.policy = Thresh_pol(d_max, action_dim)

        self.env = env

        # Trajectory history storage
        #self.history = History()
        self.policy_optim = optim.Adam(self.policy.parameters(), 1e-3)
        self.grad_history = []
        self.thresh_grad_history = np.zeros((2,action_dim))
        for param in self.policy.parameters():
            self.grad_history.append(torch.zeros(param.size()))
    def get_action(self, state) :
        if state[0] > np.sum(self.policy.d_minus) or state[0] < np.sum(self.policy.d_plus) :
            return self.policy.thresh_action(state)
        else :
            state = torch.Tensor(np.array([state[0]])).view(-1)
            return self.policy.nz_action(state).detach().numpy()

    def d_plus_update(self, state,  update_count):
        a_n = 1e-2 / (1 + update_count) ** (0.2)
        c_n = 1e-2 / (1 + update_count) ** (0.02)

        vec = c_n * (npr.binomial(1, 0.5, self.action_dim) * 2 - 1) * (npr.rand(self.action_dim) + 0.5)
        d1 = self.policy.d_plus.reshape(-1) + vec
        d2 = self.policy.d_plus.reshape(-1) - vec
        r1 = self.env.get_reward(state, d1)
        r2 = self.env.get_reward(state, d2)
        grad_est = (r1 - r2) / vec / 2
        self.thresh_grad_history[0,:] = self.thresh_grad_history[0,:] * 0.9 + 0.1 * grad_est
        self.policy.d_plus += a_n * self.thresh_grad_history[0,:]


    def d_minus_update(self, state, update_count):
        a_n = 1e-2 / (1 + update_count) ** (0.2)
        c_n = 1e-2 / (1 + update_count) ** (0.02)

        vec = c_n * (npr.binomial(1, 0.5, self.action_dim) * 2 - 1) * (npr.rand(self.action_dim) + 0.5)
        d1 = self.policy.d_minus.reshape(-1) + vec
        d2 = self.policy.d_minus.reshape(-1) - vec
        r1 = self.env.get_reward(state, d1)
        r2 = self.env.get_reward(state, d2)
        grad_est = (r1 - r2) / vec / 2
        self.thresh_grad_history[1, :] = self.thresh_grad_history[1, :] * 0.99 + 0.01 * grad_est
        self.policy.d_minus += a_n * self.thresh_grad_history[1,:]

    def nz_update(self,current_state, update_count):
        a_n = 1e-4 / (1 + update_count) ** (0.5)  #* 1 / (1 + 0.1 * (update_count // 10000))
        c_n = 1e-4 / (1 + update_count) ** (0.2)
        vecs = []
        for param in self.policy.parameters():
            #vec = c_n * (torch.rand(param.size()) * 2 - 1)
            vec = c_n * (torch.Tensor(npr.binomial(1, 0.5, param.size())) * 2 -1) * (torch.rand(param.size()) + 0.5)
            param.data.add_(vec)
            vecs.append(vec)

        state = torch.Tensor([current_state[0]])
        d0_plus = self.policy.nz_action(state).detach().numpy()
        r_plus = self.env.get_reward(current_state, d0_plus )
        i = 0
        for param in self.policy.parameters():
            vec = vecs[i]
            param.data.add_(-2 * vec)
            i += 1
        d0_minus = self.policy.nz_action(state).detach().numpy()
        r_minus = self.env.get_reward(current_state, d0_minus)
        i = 0
        for param in self.policy.parameters():
            vec = vecs[i]
            grad_est = (r_plus - r_minus) / (vec) / 2
            self.grad_history[i] = 0.99 * self.grad_history[i] + 0.01 * grad_est
            param.data.add_(vec + a_n * self.grad_history[i])
            if np.isinf(np.max(param.data.detach().numpy())) :
                raise Exception('Encountered infinite gradient')
            i += 1
#        print(i)
        '''
        c_n = current_action[1,0] - current_action[0,0]


        dr = np.array([(current_reward[1] - current_reward[0]) / c_n, (current_reward[2] - current_reward[0]) / c_n])
        dr = torch.Tensor(dr)

        action = self.policy.nz_action(state)
        action[0].backward()
        grads0 = []
        for param in self.policy.parameters():
            grads0.append(param.grad * dr[0])

        self.policy_optim.zero_grad()
        action = self.policy.nz_action(state)
        action[1].backward()
        grads1 = []
        for param in self.policy.parameters():
            grads1.append(param.grad * dr[1])

        self.policy.fc1.weight.data += a_n * (grads0[0] + grads1[0])
        self.policy.fc1.bias.data += a_n * (grads0[1] + grads1[1])

        self.policy.fc2.weight.data += a_n * (grads0[2] + grads1[2])
        self.policy.fc2.bias.data += a_n * (grads0[3] + grads1[3])

        self.policy.fc3.weight.data += a_n * (grads0[4] + grads1[4])
        self.policy.fc3.bias.data += a_n * (grads0[5] + grads1[5])

        self.policy_optim.zero_grad()
        '''


