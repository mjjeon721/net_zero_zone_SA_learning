from model import *
import numpy as np
import torch
import torch.optim as optim

class Agent():
    '''
        Learning Agent
        Learning threshold values and net-zero zone action network.
    '''
    def __init__(self, d_max, action_dim):
        self.d_max = d_max
        self.action_dim = action_dim

        # Thresholds policy module
        self.policy = Thresh_pol(d_max, action_dim)

        # Trajectory history storage
        #self.history = History()
        self.policy_optim = optim.Adam(self.policy.parameters(), 1e-3)
    def get_action(self, state) :
        if state[0] > np.sum(self.policy.d_minus) or state[0] < np.sum(self.policy.d_plus) :
            return self.policy.thresh_action(state)
        else :
            state = torch.Tensor(np.array([state[0]])).view(-1)
            return self.policy.nz_action(state).detach().numpy()

    def thresh_update(self, current_state, current_action, current_reward, update_count):
        lr = 1e-3 * 1 / (1 + 0.1 * (update_count // 10000))

        c_n = current_action[1,0] - current_action[0,0]

        dr = np.array([(current_reward[1] - current_reward[0]) / c_n, (current_reward[2] - current_reward[0]) / c_n])

        if current_state[0] > sum(current_action[0,:]) :
            self.policy.d_minus = self.policy.d_minus + lr * dr
        else :
            self.policy.d_plus = self.policy.d_plus + lr * dr

    def nz_update(self,current_state, current_action, current_reward, update_count):
        a_n = 1e-3 * 1 / (1 + 0.1 * (update_count // 10000))
        c_n = current_action[1,0] - current_action[0,0]
        state = torch.Tensor([current_state[0]])

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


