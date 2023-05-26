import numpy as np
import torch
import torch.nn as nn
import torch.autograd

def fanin_init(size) :
    '''
        :param size: Neural network size
        :return: Fan-in initialized neural network
        :Explanation: fan-in initialization of the neural network.
    '''
    fanin = size[0]
    v = 1 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)
class Thresh_pol(nn.Module) :
    '''
        Threshold_policy class
        Threshold actions if DER is in net-cons / net-prod. region
        Net-zero actions if DER is in net-zero zone, which is determined by net-zero zone network
            Input : g_t (current DER generation)
            Output : d_t (Consumption Schedule)

            Explanation : Learning net-zero zone network using SA (Stochastic Approximation) for the reward gradient estimation.
            Updating network parameter using gradient ascent of reward function.
    '''
    def __init__(self, d_max, action_dim, hidden1 = 100, hidden2 = 64):
        super(Thresh_pol, self).__init__()
        self.d_max = d_max
        self.action_dim = action_dim
        self.fc1 = nn.Linear(1, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 0)

        self.init_weight()

        self.d_plus = 0.5 * self.d_max * np.random.rand(2)
        self.d_minus = 0.5 * self.d_max * np.random.rand(2) + 0.5 * self.d_max

        self.d_plus = self.d_plus.reshape(1, self.action_dim)
        self.d_minus = self.d_minus.reshape(1, self.action_dim)

    def init_weight(self):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())


    def nz_action(self, state):
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        #out = self.sigmoid(out)
        allocation = self.softmax(out)
        #allocation = out / torch.sum(out)

        return state * allocation

    def thresh_action(self, state):
        state = state.reshape(-1,3)

        actions = (state[:,0] < np.sum(self.d_plus)).reshape(-1,1) * self.d_plus \
                  + (state[:,0] > np.sum(self.d_minus)).reshape(-1,1) * self.d_minus

        return actions.reshape(-1)
