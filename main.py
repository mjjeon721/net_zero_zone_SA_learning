import copy
import torch
import numpy as np
from agent import Agent
import torch.optim as optim
from utils import *
import matplotlib.pyplot as plt
import time
from scipy.stats import truncnorm

a = np.array([3, 2.4])
b = np.array([1, 1])

pi_p = 2
pi_m = 0.2

opt_d_plus = (a - pi_p) / b
opt_d_minus = (a - pi_m) / b

d_max = 3
action_dim = len(a)

g_mean = 4
g_std = 2

env = Env([a, b], [g_mean, g_std, 0, 9])
agent_lr = Agent(d_max, action_dim, env)

#g_low = 0
#g_high = 9

#low = (g_low - g_mean) / g_std
#high = (g_high - g_mean) / g_std
#truncnorm.rvs(low, high, loc = g_mean, scale = g_std, size = 1)

epoch_size = 100
num_epoch = 2000

THL_reward = []
THL_avg_reward = []

OPT_reward = []
OPT_avg_reward = []

d_plus_history = []
d_minus_history = []

tic = time.perf_counter()
update_count = 1
d_plus_update_count = 0
d_minus_update_count = 0
nz_update_count = 0
interaction = 0
for epoch in range(num_epoch) :
    g_n = env.get_next_state()
    state = np.array([g_n.item(), pi_p, pi_m])

    epoch_reward_thl = 0
    epoch_reward_opt = 0

    for episode in range(epoch_size) :
        c_n = 1e-3
        action = agent_lr.get_action(state)
        d_n = action.reshape(-1)
        r_n = env.get_reward(state, d_n)
        if all(np.isnan(d_n)):
            raise Exception('Encountered nan')
        if state[0] < np.sum(agent_lr.policy.d_plus) :
            agent_lr.d_plus_update(state, d_plus_update_count)
            d_plus_update_count += 1
        elif state[0] > np.sum(agent_lr.policy.d_minus):
            agent_lr.d_minus_update(state, d_minus_update_count)
            d_minus_update_count += 1
        else :
            nz_update_count += 1
            agent_lr.nz_update(state, nz_update_count)

        '''
        d_n1 = d_n + np.array([c_n, 0])
        d_n2 = d_n + np.array([0, c_n])

        r_n = env.get_reward(state, d_n)
        r_n1 = env.get_reward(state, d_n1)
        r_n2 = env.get_reward(state, d_n2)

        current_action = np.array([d_n, d_n1, d_n2])
        current_reward = np.array([r_n, r_n1, r_n2])
        if np.abs(np.sum(action) - state[0]) <= 1e-6 :
            agent_lr.nz_update(state, d_n, current_reward, interaction)
        else :
            agent_lr.thresh_update(state, d_n, current_reward, update_count)
        '''

        if state[0] < sum(opt_d_plus) :
            action_opt = opt_d_plus
        elif state[0] > sum(opt_d_minus) :
            action_opt = opt_d_minus
        else :
            action_opt = opt_d_plus + 0.5 * (state[0] - sum(opt_d_plus))
        reward_opt = env.get_reward(state, action_opt)

        g_n = env.get_next_state()
        state = np.array([g_n.item(), pi_p, pi_m])

        epoch_reward_thl += r_n
        epoch_reward_opt += reward_opt
        interaction += 1
        if interaction % 20 == 1:
            d_plus_history.append(copy.copy(agent_lr.policy.d_plus))
            d_minus_history.append(copy.copy(agent_lr.policy.d_minus))

    THL_reward.append(epoch_reward_thl)
    OPT_reward.append(epoch_reward_opt)

    THL_avg_reward.append(np.mean(THL_reward[-100:]))
    OPT_avg_reward.append(np.mean(OPT_reward[-100:]))

    if epoch % 100 == 99 :
        toc = time.perf_counter()
        print('1 Epoch running time : {0:.4f} (s)'.format(toc - tic))
        print('Epoch : {0}, Threshold_learning : {1:.4f}, Optimal_avg_reward : {2:.4f}'.format(
            epoch, THL_avg_reward[-1], OPT_avg_reward[-1]))
        tic = time.perf_counter()


d_minus_history = np.vstack(d_minus_history)
d_plus_history = np.vstack(d_plus_history)

plt.plot(np.arange(0, interaction, 20), d_plus_history[:,0])
plt.plot(np.arange(0, interaction, 20), np.ones(int(interaction/20)) * opt_d_plus[0])
plt.title('Threshold learning trajectory')
plt.xlabel('Interactions')
plt.ylabel('$d_1^+$')
plt.grid()
plt.show()

plt.plot(np.arange(0, interaction, 20), d_plus_history[:,1])
plt.plot(np.arange(0, interaction, 20), np.ones(int(interaction/20)) * opt_d_plus[1])
plt.title('Threshold learning trajectory')
plt.xlabel('Interactions')
plt.ylabel('$d_2^+$')
plt.grid()
plt.show()

plt.plot(np.arange(0, interaction, 20), d_minus_history[:,0])
plt.plot(np.arange(0, interaction, 20), np.ones(int(interaction/20))* opt_d_minus[0])
plt.title('Threshold learning trajectory')
plt.xlabel('Interactions')
plt.ylabel('$d_1^-$')
plt.grid()
plt.show()

plt.plot(np.arange(0, interaction, 20), d_minus_history[:,1])
plt.plot(np.arange(0, interaction, 20), np.ones(int(interaction/20)) * opt_d_minus[1])
plt.title('Threshold learning trajectory')
plt.xlabel('Interactions')
plt.ylabel('$d_2^-$')
plt.grid()
plt.show()

nsmoothed_curve_thl = np.array([])
nsmoothed_curve_opt = np.array([])
for i in range(num_epoch) :
    nsmoothed_curve_thl = np.append(nsmoothed_curve_thl, np.mean(THL_avg_reward[np.maximum(i -10, 0):i + 1]))
    nsmoothed_curve_opt = np.append(nsmoothed_curve_opt, np.mean(OPT_avg_reward[np.maximum(i-10, 0):i + 1]))
plt.plot(np.arange(0, interaction, epoch_size),nsmoothed_curve_thl, label = 'Threshold_learning')
plt.plot(np.arange(0, interaction, epoch_size),nsmoothed_curve_opt, label = 'OPT')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Performance')
plt.grid()
plt.show()

regret_thl = np.abs(nsmoothed_curve_opt - nsmoothed_curve_thl) / nsmoothed_curve_opt * 100
plt.plot(np.arange(0, interaction, epoch_size),regret_thl)
plt.grid()
plt.show()

x = np.arange(0, 8, 0.01)
z = []
for i in range(len(x)) :
    z.append(agent_lr.get_action(np.array([x[i], pi_p, pi_m])))

opt_action = opt_d_plus + (x.reshape(-1,1) - sum(opt_d_plus)) * (opt_d_minus - opt_d_plus) / (sum(opt_d_minus) - sum(opt_d_plus))
opt_action = np.maximum(np.minimum(opt_action, opt_d_minus), opt_d_plus)

z = np.vstack(z)

plt.plot(x, z[:,0], label = 'Learned policy')
plt.plot(x, opt_action[:,0], label = 'OPT')
plt.grid()
plt.legend()
plt.show()

plt.plot(x, z[:,1], label = 'Learned policy')
plt.plot(x, opt_action[:,1], label = 'OPT')
plt.grid()
plt.legend()
plt.show()

'''
    for j in range(1):
        a_n = 1e-3 * 1 / (1 + (i // 5000))
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

print(np.mean(np.abs(z - opt_d_nz)) * 100)
'''