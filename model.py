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
class NZ_action(nn.Module) :
    '''
        Net-zero zone network model
            Input : g_t (current DER generation)
            Output : d_t (Consumption Schedule)

            Explanation : Learning net-zero zone network using SA (Stochastic Approximation) for the reward gradient estimation.
            Updating network parameter using gradient ascent of reward function.
    '''
    def __init__(self, d_max, action_dim, hidden1 = 100, hidden2 = 64):
        super(NZ_action, self).__init__()
        self.d_max = d_max
        self.action_dim = action_dim
        self.fc1 = nn.Linear(1, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weight()

    def init_weight(self):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())


    def forward(self, state):
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        #out = self.sigmoid(out)
        allocation = self.softmax(out)
        #allocation = out / torch.sum(out)

        return state * allocation



