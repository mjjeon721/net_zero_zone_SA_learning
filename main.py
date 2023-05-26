import torch
import numpy as np
from model import NZ_action
import torch.optim as optim
from utils import *
import matplotlib.pyplot as plt
import time

a = np.array([3, 2.4])
b = np.array([1, 1])

pi_p = 2
pi_m = 0.2

opt_d_plus = (a - pi_p) / b
opt_d_minus = (a - pi_m) / b

d_max = 3
action_dim = len(a)

g_mean = 2
g_std = 1

net_zero_action = NZ_action(d_max, action_dim)

env = Env([a, b], [g_mean, g_std, sum(opt_d_plus), sum(opt_d_minus)])

nz_optim = optim.Adam(net_zero_action.parameters(), 1e-3)

max_iter = 20000

tic = time.perf_counter()
for i in range(max_iter) :
    g_n = torch.Tensor(np.array([env.get_next_state()]))
    n = 1
    for j in range(5):
        a_n = 1e-3 * 1 / (n)
        c_n = 1e-3 * 1 / (n) ** (1 / 3)
        action = net_zero_action.forward(g_n)
        d_n = action.view(-1)
        d_n1 = d_n + torch.Tensor(np.array([c_n, 0]))
        d_n2 = d_n + torch.Tensor(np.array([0, c_n]))

        r_n = env.get_reward(np.array([g_n, pi_p, pi_m]), d_n.detach().numpy())
        r_n1 = env.get_reward(np.array([g_n, pi_p, pi_m]), d_n1.detach().numpy())
        r_n2 = env.get_reward(np.array([g_n, pi_p, pi_m]), d_n2.detach().numpy())

        dr_d = np.array([(r_n1 - r_n) / c_n, (r_n2 - r_n) / c_n])
        action[0,0].backward()
        grads0 = []
        #print(net_zero_action.fc1.weight.data[:20])
        for param in net_zero_action.parameters():
            grads0.append(param.grad * dr_d[0])

        nz_optim.zero_grad()
        action = net_zero_action.forward(g_n)
        action[0,1].backward()
        grads1 = []
        for param in net_zero_action.parameters():
            grads1.append(param.grad * dr_d[1])

        net_zero_action.fc1.weight.data += a_n * (grads0[0] + grads1[0])
        net_zero_action.fc1.bias.data += a_n * (grads0[1] + grads1[1])

        net_zero_action.fc2.weight.data += a_n * (grads0[2] + grads1[2])
        net_zero_action.fc2.bias.data += a_n * (grads0[3] + grads1[3])

        net_zero_action.fc3.weight.data += a_n * (grads0[4] + grads1[4])
        net_zero_action.fc3.bias.data += a_n * (grads0[5] + grads1[5])

        nz_optim.zero_grad()

        n += 1
        #print(net_zero_action.fc1.weight.data[:20])
    if i % 1000 == 999 :
        toc = time.perf_counter()
        print("{0} th iteration time : {1:.4f}(s)".format(i, toc-tic))


x = torch.arange(sum(opt_d_plus), sum(opt_d_minus), 0.01).view(-1,1)
z = net_zero_action.forward(x)

x = x.detach().numpy().reshape(-1,1)

opt_d_nz = opt_d_plus + (x - sum(opt_d_plus)) * (opt_d_minus - opt_d_plus) / (sum(opt_d_minus) - sum(opt_d_plus))
z = z.detach().numpy()

plt.plot(x, z[:,0], label = 'Learned')
plt.plot(x, opt_d_nz[:,0], label = 'Optimal')
plt.legend()
plt.grid()
plt.show()

plt.plot(x, z[:,1], label = 'Learned')
plt.plot(x, opt_d_nz[:,1], label = 'Optimal')
plt.legend()
plt.grid()
plt.show()

np.mean(np.abs(z - opt_d_nz)) * 100