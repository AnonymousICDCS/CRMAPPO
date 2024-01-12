import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ContinousPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, log_std=0):
        super(ContinousPolicy, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)

        # new adding!
        self.mu_head.weight.data.mul_(0.1)
        self.mu_head.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        mu_a = torch.softmax(mu, dim=1)
        return mu_a

    def get_dist(self, state):
        mu = self.forward(state)
        #print(mu)
        action_log_std = self.action_log_std.expand_as(mu)
        action_std = torch.exp(action_log_std)
        dist = Normal(mu, action_std)
        return dist

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(DiscretePolicy, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        #print(state.shape)
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = 0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob


class HybridValue(nn.Module):
    def __init__(self, state_dim, net_width):
        super(HybridValue, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, s):
        v = torch.tanh(self.C1(s))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v

class RewardDiscrete(nn.Module):
    def __init__(self, state_dim, a_r_dim, a_f_dim, a_p_dim, net_width):
        super(RewardDiscrete, self).__init__()
        self.s = nn.Linear(state_dim, net_width)
        self.af = nn.Linear(a_f_dim, net_width)
        self.ap = nn.Linear(a_p_dim, net_width)
        self.l1 = nn.Linear(net_width+net_width+net_width, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, a_r_dim)
    def forward(self, s, a_f, a_p):
        h_s = self.s(s)
        h_af = self.af(a_f)
        h_ap = self.ap(a_p)
        h = torch.cat([h_s, h_af, h_ap], -1)
        r = torch.tanh(self.l1(h))
        r = torch.tanh(self.l2(r))
        r = self.l3(r)
        return r

class RewardContinuous(nn.Module):
    def __init__(self, state_dim, a_f_dim, a_p_dim, net_width):
        super(RewardContinuous, self).__init__()
        self.s = nn.Linear(state_dim, net_width)
        self.ar = nn.Linear(1, net_width)
        self.af = nn.Linear(a_f_dim, net_width)
        self.ap = nn.Linear(a_p_dim, net_width)
        self.l1 = nn.Linear(net_width+net_width+net_width+net_width, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)

    def forward(self, s, a_r, a_f, a_p):
        h_s = self.s(s)
        a_r = a_r.to(torch.float32)  # Convert the integer tensor to a float tensor
        h_ar = self.ar(a_r)
        h_af = self.af(a_f)
        h_ap = self.ap(a_p)
        h = torch.cat([h_s, h_ar, h_af, h_ap], -1)
        r = torch.tanh(self.l1(h))
        r = torch.tanh(self.l2(r))
        r = self.l3(r)
        return r
