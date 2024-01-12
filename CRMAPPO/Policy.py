import copy
import numpy as np
import torch
from torch.distributions import Categorical
import math
from model import DiscretePolicy, ContinousPolicy, HybridValue, RewardDiscrete, RewardContinuous
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
import time
from environment import video

class PPO(object):
    def __init__(
            self,
            env_with_Dead,
            state_dim,  # we take the state_c as the global state
            action_r_dim,
            action_f_dim,
            action_p_dim,
            beta=0.6,
            gamma=0.99,
            lambd=0.95,
            net_width=200,
            a_lr=1e-4,
            v_lr=1e-4,
            clip_rate=0.2,
            K_epochs=10,
            batch_size=64,
            l2_reg=1e-3,
            entropy_coef=1e-3,
            adv_normalization=False,
            entropy_coef_decay=0.99,
            user_num = 6,
            load_reward=False
    ):
        env = video(user_num)
        self.actor_r = DiscretePolicy(state_dim, action_r_dim, net_width).to(device)
        self.actor_f = ContinousPolicy(state_dim+1, action_f_dim, net_width).to(device)
        self.actor_p = ContinousPolicy(state_dim+1 + env.user_num, action_p_dim, net_width).to(device)
        self.actor_r_optimizer = torch.optim.Adam(self.actor_r.parameters(), lr=a_lr)
        self.actor_f_optimizer = torch.optim.Adam(self.actor_f.parameters(), lr=a_lr)
        self.actor_p_optimizer = torch.optim.Adam(self.actor_p.parameters(), lr=a_lr)
        self.critic = HybridValue(state_dim, net_width).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=v_lr)
        self.reward_d = RewardDiscrete(state_dim, action_r_dim, action_f_dim, action_p_dim, net_width).to(device)
        self.reward_c = RewardContinuous(state_dim, action_f_dim, action_p_dim, net_width).to(device)
        self.reward_d_optimizer = torch.optim.Adam(self.reward_d.parameters(), lr=v_lr)
        self.reward_c_optimizer = torch.optim.Adam(self.reward_c.parameters(), lr=v_lr)
        self.data = []
        self.env_with_Dead = False
        self.gamma = gamma
        self.beta = beta
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.optim_batch_size = batch_size
        self.l2_reg = l2_reg
        self.entropy_coef = entropy_coef
        self.adv_normalization = adv_normalization
        self.entropy_coef_decay = entropy_coef_decay
        self.Lreward=load_reward

    def select_action(self, state, stage):
        '''Stochastic Policy'''
        with torch.no_grad():
            if stage == 'resolution':
                pi = self.actor_r.pi(state, softmax_dim=0)
                m = Categorical(pi)
                a = m.sample().item()
                pi_a = pi[a].item()
                return a, pi_a
            elif stage == 'frequency':
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                dist = self.actor_f.get_dist(state)
                a = dist.sample()
                a = torch.clamp(a, 0, 1)
                a = torch.softmax(a, dim=1)
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                return a.cpu().numpy().flatten(), logprob_a
            elif stage == 'power':
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                dist = self.actor_p.get_dist(state)
                a = dist.sample()
                a = torch.clamp(a, 0, 1)
                a = torch.softmax(a, dim=1)
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                return a.cpu().numpy().flatten(), logprob_a
            else:
                raise NotImplementedError('Unknown stage {}'.format(stage))

    def sample_action(self, state, stage):
        with torch.no_grad():
            if stage == 'frequency':
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                dist = self.actor_f.get_dist(state)
                # sample times, can be higher or pretrain reward network in more complicated scenarios
                a_num = 100
                a = dist.sample((a_num,))
                a = torch.clamp(a, 0, 1)
                a = torch.softmax(a, dim=2)
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                return [sample.cpu().numpy().flatten() for sample in a], logprob_a
            elif stage == 'power':
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                dist = self.actor_p.get_dist(state)
                # sample times, can be higher or pretrain reward network in more complicated scenarios
                a_num = 100
                a = dist.sample((a_num,))
                a = torch.clamp(a, 0, 1)
                a = torch.softmax(a, dim=2)
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                return [sample.cpu().numpy().flatten() for sample in a], logprob_a
            else:
                raise NotImplementedError('Unknown stage {}'.format(stage))

    def evaluate(self, state, stage):
        '''Deterministic Policy'''
        with torch.no_grad():
            if stage == 'resolution':
                pi = self.actor_r.pi(state, softmax_dim=0)
                a = torch.argmax(pi).item()
                return a, 1.0
            elif stage == 'frequency':
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                a = self.actor_f(state)
                return a.cpu().numpy().flatten(), 0.0
            elif stage == 'power':
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                a = self.actor_p(state)
                return a.cpu().numpy().flatten(), 0.0
            else:
                raise NotImplementedError('Unknown stage {}'.format(stage))

    def train(self):
        # r is a tuple
        s, a_r, a_f, a_p, a_f_sample, a_p_sample, r_r, r_f, r_p, s_prime, old_prob_r, old_log_prob_f, old_log_prob_p, terminal, dw = self.make_batch()
        self.entropy_coef *= self.entropy_coef_decay  # exploring decay

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            # (bs, r_dim)
            vs = self.critic(s)
            vs_ = self.critic(s_prime)
            pr_r = self.reward_d(s, a_f, a_p)
            # [1000, sample_num, user_num]
            _, sample_num_f, a_f_dim = a_f_sample.shape
            _, sample_num_p, a_p_dim = a_p_sample.shape

            # unsqueeze and expand for duplicating for sample num ties to fit dimension
            pr_f = self.reward_c(s.unsqueeze(1).expand(-1, sample_num_f, -1),
                                      a_r.unsqueeze(1).expand(-1, sample_num_f, -1),
                                      a_f_sample,
                                      a_p.unsqueeze(1).expand(-1, sample_num_f, -1))
            pr_f = pr_f.squeeze(-1)
            pr_p = self.reward_c(s.unsqueeze(1).expand(-1, sample_num_p, -1),
                                 a_r.unsqueeze(1).expand(-1, sample_num_p, -1),
                                 a_f.unsqueeze(1).expand(-1, sample_num_p, -1),
                                 a_p_sample)
            pr_p = pr_p.squeeze(-1)

            # calculate the average as expectation for calculating Advantages
            ex_r = pr_r.mean(dim=1, keepdim=True)
            ex_f = pr_f.mean(dim=1, keepdim=True)
            ex_p = pr_p.mean(dim=1, keepdim=True)

            adv_r, td_target_r = self.get_adv_td(vs, vs_, r_r, ex_r, terminal, dw)  # just use adv_c to update actor 1, the critic is updated by td_target_g
            adv_p, td_target_p = self.get_adv_td(vs, vs_, r_p, ex_f, terminal, dw)  # just use adv_p to update actor 2, the critic is updated by td_target_g
            adv_f, td_target_f = self.get_adv_td(vs, vs_, r_f, ex_p, terminal, dw)  # just use td_garget_g to update critic, the adv_f is not used (it is the concatenation of adv_c and adv_f)

        """PPO update"""
        # Slice long trajectopy into short trajectory and perform mini-batch PPO update
        c_optim_iter_num = int(math.ceil(s.shape[0] / self.optim_batch_size))
        p_optim_iter_num = int(math.ceil(s.shape[0] / self.optim_batch_size))
        for _ in range(self.K_epochs):
            # Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            s, a_r, a_f, a_p, td_target_r, td_target_f, td_target_p, adv_r, adv_f, adv_p, old_prob_r, old_log_prob_f, old_log_prob_p = \
                s[perm].clone(), a_r[perm].clone(), a_f[perm].clone(), a_p[perm].clone(), td_target_r[perm].clone(), td_target_f[perm].clone(), td_target_p[perm].clone(),\
                adv_r[perm].clone(), adv_f[perm].clone(), adv_p[perm].clone(), old_prob_r[perm].clone(), old_log_prob_f[perm].clone(), old_log_prob_p[perm].clone()

            policy_r_loss, policy_f_loss, policy_p_loss, reward_d_loss, reward_c_loss, value_loss = 0., 0., 0., 0., 0., 0.

            # start time
            start = time.perf_counter()
            '''mini-batch PPO update'''
            for i in range(c_optim_iter_num):
                index = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, s.shape[0]))
                #print(index)
                '''actor update'''

                '''resolution actor update'''
                prob = self.actor_r.pi(s[index], softmax_dim=1)
                entropy = Categorical(prob).entropy().sum(0, keepdim=True)
                prob_a = prob.gather(1, a_r[index])
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_r[index]))  # a/b == exp(log(a)-log(b))
                #adv_c_sum = torch.sum(adv_c[index], dim=1, keepdim=True)
                surr1_r = ratio * adv_r[index]
                surr2_r = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv_r[index]
                a_r_loss = -torch.min(surr1_r, surr2_r) - self.entropy_coef * entropy

                policy_r_loss += a_r_loss.mean()
                self.actor_r_optimizer.zero_grad()
                a_r_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor_r.parameters(), 40)
                self.actor_r_optimizer.step()

                '''frequency actor update'''
                s_f = torch.cat([s[index], a_r[index]], -1)
                #s_p = s[index]
                distribution = self.actor_f.get_dist(s_f)
                dist_entopy = distribution.entropy().sum(1, keepdim=True)
                logprob_now = distribution.log_prob(a_f[index])
                ratio_f = torch.exp(logprob_now.sum(1, keepdim=True) - old_log_prob_f[index].sum(1, keepdim=True))
                # !!!
                #adv_p_sum = torch.sum(adv_p[index], dim=1, keepdim=True)
                surr1_f = ratio_f * adv_f[index]
                surr2_f = torch.clamp(ratio_f, 1 - self.clip_rate, 1 + self.clip_rate) * adv_f[index]

                actor_f_loss = -torch.min(surr1_f, surr2_f) - self.entropy_coef * dist_entopy
                policy_f_loss += actor_f_loss.mean()
                self.actor_f_optimizer.zero_grad()
                actor_f_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor_f.parameters(), 40)
                self.actor_f_optimizer.step()

                '''power actor update'''
                s_p = torch.cat([s[index], a_r[index], a_f[index]], -1)
                # s_p = s[index]
                distribution = self.actor_p.get_dist(s_p)
                dist_entopy = distribution.entropy().sum(1, keepdim=True)
                logprob_now = distribution.log_prob(a_p[index])
                ratio_p = torch.exp(logprob_now.sum(1, keepdim=True) - old_log_prob_p[index].sum(1, keepdim=True))
                surr1_p = ratio_p * adv_p[index]
                surr2_p = torch.clamp(ratio_p, 1 - self.clip_rate, 1 + self.clip_rate) * adv_p[index]
                actor_p_loss = -torch.min(surr1_p, surr2_p) - self.entropy_coef * dist_entopy
                policy_p_loss += actor_p_loss.mean()
                self.actor_f_optimizer.zero_grad()
                actor_p_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor_p.parameters(), 40)
                self.actor_p_optimizer.step()

                '''critic update'''

                # (bs,r_dim)
                c_loss = (self.critic(s[index]) - td_target_r[index]).pow(2).mean() + (self.critic(s[index]) - td_target_f[index]).pow(2).mean() + (self.critic(s[index]) - td_target_p[index]).pow(2).mean()
                value_loss += c_loss

                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

                #if not self.Lreward:
                '''reward update'''
                pre_d = self.reward_d(s[index], a_f[index], a_p[index])
                pre_d = pre_d.gather(1, a_r[index])
                r_d_loss = (pre_d-r_r[index]).pow(2).mean()
                pre_c = self.reward_c(s[index], a_r[index], a_f[index], a_p[index])
                r_c_loss = (pre_c-r_f[index]).pow(2).mean()

                reward_d_loss += r_d_loss
                reward_c_loss += r_c_loss

                # l2 regression
                for name, param in self.reward_d.named_parameters():
                    if 'weight' in name:
                        r_d_loss += param.pow(2).sum() * self.l2_reg
                for name, param in self.reward_c.named_parameters():
                    if 'weight' in name:
                        r_c_loss += param.pow(2).sum() * self.l2_reg

                # discrete reward network update
                self.reward_d_optimizer.zero_grad()
                r_d_loss.backward()
                self.reward_d_optimizer.step()

                # continuous reward network update
                self.reward_c_optimizer.zero_grad()
                r_c_loss.backward()
                self.reward_c_optimizer.step()

            end = time.perf_counter()
            policy_r_loss /= c_optim_iter_num
            policy_f_loss /= c_optim_iter_num
            policy_p_loss /= c_optim_iter_num
            reward_d_loss /= c_optim_iter_num
            reward_c_loss /= c_optim_iter_num
            value_loss /= c_optim_iter_num

        return policy_r_loss, policy_f_loss, policy_p_loss, value_loss, reward_d_loss, reward_c_loss

    def get_adv_td(self, vs, vs_, r, ex, terminal, dw):
        '''dw for TD_target and Adv'''
        # DAE advantage
        deltas = r + self.gamma * vs_ * (1 - dw) - vs - self.beta * ex
        deltas = deltas.cpu().flatten(end_dim=0).numpy()
        adv = [0]

        '''done for GAE'''
        for dlt, mask in zip(deltas[::-1], terminal.cpu().flatten().numpy()[::-1]):
            advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
            adv.append(advantage)
        adv.reverse()
        adv = copy.deepcopy(adv[0:-1])
        # adv = torch.tensor(adv).unsqueeze(1).float().to(device)
        adv = torch.tensor(adv).float().to(device)

        td_target = adv + vs

        return adv, td_target

    def make_batch(self):
        # s, a_r, a_f, a_p, r_r, r_f, r_p, s_prime, old_prob_r, old_log_prob_f, old_log_prob_p, terminal, dw
        s_lst, a_r_lst, a_f_lst, a_p_lst, a_f_sample_lst, a_p_sample_lst, r_r_lst, r_f_lst, r_p_lst, s_prime_lst, old_prob_r_lst, old_log_prob_f_lst, old_log_prob_p_lst, done_lst, dw_lst = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for transition in self.data:
            s, a_r, a_f, a_p, a_f_sample, a_p_sample, r_r, r_f, r_p, s_prime, old_prob_r, old_log_prob_f, old_log_prob_p, done, dw = transition
            s_lst.append(s)
            a_r_lst.append([a_r])
            a_f_lst.append(a_f)
            a_p_lst.append(a_p)
            a_f_sample_lst.append(a_f_sample)
            a_p_sample_lst.append(a_p_sample)
            r_r_lst.append([r_r])
            r_f_lst.append([r_f])
            r_p_lst.append([r_p])
            s_prime_lst.append(s_prime)
            old_prob_r_lst.append([old_prob_r])
            old_log_prob_f_lst.append(old_log_prob_f)
            old_log_prob_p_lst.append(old_log_prob_p)
            done_lst.append([done])
            dw_lst.append([dw])


        if not self.env_with_Dead:
            '''Important!!!'''
            # env_without_DeadAndWin: deltas = r + self.gamma * vs_ - vs
            # env_with_DeadAndWin: deltas = r + self.gamma * vs_ * (1 - dw) - vs
            dw_lst = (np.array(dw_lst)*False).tolist()

        self.data = []  # Clean history trajectory

        '''list to tensor'''
        with torch.no_grad():
            s, a_r, a_f, a_p, a_f_sample, a_p_sample, r_r, r_f, r_p, s_prime, old_prob_r, old_log_prob_f, old_log_prob_p, done_mask, dw_mask = \
                torch.tensor(s_lst, dtype=torch.float).to(device), \
                torch.tensor(a_r_lst, dtype=torch.int64).to(device), \
                torch.tensor(a_f_lst, dtype=torch.float).to(device), \
                torch.tensor(a_p_lst, dtype=torch.float).to(device), \
                torch.tensor(a_f_sample_lst, dtype=torch.float).to(device), \
                torch.tensor(a_p_sample_lst, dtype=torch.float).to(device), \
                torch.tensor(r_r_lst, dtype=torch.float).to(device), \
                torch.tensor(r_f_lst, dtype=torch.float).to(device), \
                torch.tensor(r_p_lst, dtype=torch.float).to(device), \
                torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                torch.tensor(old_prob_r_lst, dtype=torch.float).to(device), \
                torch.tensor(old_log_prob_f_lst, dtype=torch.float).to(device), \
                torch.tensor(old_log_prob_p_lst, dtype=torch.float).to(device), \
                torch.tensor(done_lst, dtype=torch.float).to(device), \
                torch.tensor(dw_lst, dtype=torch.float).to(device),

        return s, a_r, a_f, a_p, a_f_sample, a_p_sample, r_r, r_f, r_p, s_prime, old_prob_r, old_log_prob_f, old_log_prob_p, done_mask, dw_mask


    def put_data(self, transition):
        self.data.append(transition)

    def save(self, episode):
        torch.save(self.critic.state_dict(), "./model/hybrid_critic{}.pth".format(episode))
        torch.save(self.actor_r.state_dict(), "./model/actor_r{}.pth".format(episode))
        torch.save(self.actor_p.state_dict(), "./model/actor_p{}.pth".format(episode))

    def save_reward(self, episode):
        torch.save(self.reward_d.state_dict(), "./model/reward_d{}.pth".format(episode))
        torch.save(self.reward_c.state_dict(), "./model/reward_c{}.pth".format(episode))

    def load(self, episode):
        self.critic.load_state_dict(torch.load("./model/hybrid_critic{}.pth".format(episode)))
        self.actor_r.load_state_dict(torch.load("./model/actor_r{}.pth".format(episode)))
        self.actor_p.load_state_dict(torch.load("./model/actor_p{}.pth".format(episode)))

    def load_reward(self, episode):
        self.reward_d.load_state_dict(torch.load("./model/reward_d{}.pth".format(episode)))
        self.reward_c.load_state_dict(torch.load("./model/reward_c{}.pth".format(episode)))