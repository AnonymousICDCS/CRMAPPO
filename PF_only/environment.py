import gym
import numpy as np
from gym import spaces
from typing import Optional
import gym
from gym import spaces
import math
from copy import deepcopy as dp

# user type
User = np.dtype({
    'names': ['p', 'g', 'data', 'res', 'rate', 'comp', 'reward', 'f', 'x', 'y', 'cumulate_delay', 'fps'],
    'formats': ['float', 'float', 'float', 'int', 'float', 'int', 'float', 'float', 'int', 'int', 'float', 'int']
})

p3k = 3840 * 1920 * 16 / 2
p2k = 2560 * 1440 * 16 / 2
p1080 = 1920 * 1080 * 16 / 2
p720 = 1280 * 720 * 16 / 2
data = [p720, p1080, p2k, p3k]
channel_loc = [5.0, 5.0]


def action_map(x, usr, res):
    num = np.zeros(usr)
    r = np.zeros(usr + 1)
    r[0] = x
    for i in range(usr):
        num[usr - i - 1] = r[i] // (res ** (usr - 1 - i))
        r[i + 1] = r[i] % (res ** (usr - 1 - i))
    return num.astype(np.int32)


def rician(usr, chal, seed=None):
    rician = np.zeros((usr, chal))
    if seed != None:
        np.random.seed(seed)
    for i in range(usr):
        for j in range(chal):
            x = np.random.randn()
            y = np.random.randn()
            A = math.sqrt(x * x + y * y)
            rician[i][j] = A
    return rician


def init_user(user_num):
    user_loc = [[8.1, 5.5],
                [3.3, 3.3],
                [5.7, 3.2],
                [9.5, 1.4],
                [6.4, 0.5],
                [0.7, 6.6],
                [7.2, 5.9],
                [6.2, 3.9],
                [3.6, 9.2],
                [6.3, 7.7]]
    # channel_loc = np.random.uniform(0, 10, size=2)
    channel_loc = [5.0, 5.0]

    comp = [15, 24, 18, 20, 27, 30, 15, 30, 23, 19]

    wavelength = 1
    user = []
    gain = rician(user_num, 1)
    for i in range(user_num):
        dist = np.linalg.norm(
            np.array((user_loc[i][0], user_loc[i][1]), dtype=float) - np.array((channel_loc[0], channel_loc[1]),
                                                                               dtype=float))
        path_loss = (3e8 / (4 * math.pi * dist * 28e9)) ** 2
        g = (math.sqrt(3 / (3 + 1)) + math.sqrt(1 / 3) * gain[i][0]) * path_loss
        # 'p', 'g', 'data', 'res', 'rate', 'comp', 'reward', 'f', 'x', 'y', 'cumulate_delay', 'fps'
        each = np.array((0, g, p3k // 300, 0, 0, comp[i], 0, 0, user_loc[i][0], user_loc[i][1], 0, 0),dtype=User)
        user.append(each)
    init_user = np.array(user, dtype=User)
    return init_user


class video(gym.Env):
    def __init__(self, user_num):
        # 6 users, 3 channel
        self.max_step = 90
        self.count = 0
        self.left_frame = self.max_step
        self.state = None
        self.res_num = 4
        # band width
        self.B = 1e7
        self.sigma = 9.9e-7
        self.K = 3
        self.res = list(range(1, self.res_num))
        self.user_num = user_num
        # user [num, p, g, data, channel, rate, frequency, req, comp, reward]
        self.user = init_user(self.user_num)
        # self.user['data'] = np.random.uniform(3e7, 3.2e7, size=self.user_num)//self.compression
        self.step_beyond_done = None
        self.loss = [[int(0)] for _ in range(self.user_num)]
        self.p_max = 80
        self.f_max = 10e8

        # state: [user data, total time]
        self.observation_space = spaces.Box(
            low = np.array([0 for _ in range(2 * self.user_num + 1)], dtype=np.float32),
            high = np.array([np.finfo(np.float32).max for _ in range(2 * self.user_num + 1)], dtype=np.float32))
        self.action_num = self.res_num ** self.user_num
        self.action_r_space = spaces.Discrete(self.action_num)
        self.action_f_space = spaces.Box(low=0, high=1, shape=(self.user_num,), dtype=np.float32)
        self.action_p_space = spaces.Box(low=0, high=1, shape=(self.user_num,), dtype=np.float32)


    def reset(self):
        self.count = 0
        self.left_frame = self.max_step
        self.loss = [[int(0)] for _ in range(self.user_num)]
        # reset user data
        self.state = [self.left_frame]
        self.user = init_user(self.user_num)
        # self.user['data'] = [p2k//np.random.uniform(300,600) for _ in range(self.user_num)]

        for i in range(self.user_num):
            for j in ['comp', 'g']:
                self.state.append(self.user[i][j])
        return np.array(self.state, dtype=np.float32)

    def step(self, action_r, action_f, action_p):
        #res = [1 for _ in range(self.user_num)]  # no.1 resolution for 1080p
        res = action_map(int(action_r), self.user_num, self.res_num)
        f = action_f
        power = action_p

        reward_r, reward_f, reward_p = 0.0, 0.0, 0.0
        resol_change = 0
        # check if the resolution is change:
        for i in range(self.user_num):
            if self.user[i]['res'] != res[i]:
                reward_r -= 0.2
                resol_change += 1

        # initial basic attributes
        self.user['res'] = res
        self.user['p'] = power
        self.user['f'] = f

        # resolution reward
        r_resol = 0
        cyc_bit = 150
        ave_consecutive, ave_copying, ave_resol_score = 0.0, 0.0, 0.0
        segment_length = [[] for _ in range(self.user_num)]
        copying_length = []
        # calculate delay
        for i in range(self.user_num):
            self.user[i]['rate'] = self.B * np.log2(1 + self.user[i]['p'] * self.user[i]['g'] / (self.B * self.sigma ** 2))
            self.user[i]['data'] = data[self.user[i]['res']] // np.random.uniform(1400, 1500)
            if self.user[i]['f'] != 0:
                exe_delay = self.user[i]['data'] * cyc_bit / self.user[i]['f']
            else:
                exe_delay = 1
            if self.user[i]['rate'] != 0:
                delay = self.user[i]['data'] / self.user[i]['rate']
            else:
                delay = 1

            if (delay + exe_delay) > 0.02:  # unsatisfied
                self.loss[i].append(0)
                self.user[i]['comp'] -= 1
                reward_r -= 1
                reward_f -= 1

            else:  # satisfied
                # receive reward
                self.loss[i].append(1)
                self.user[i]['fps'] += 1
                reward_r += 1
                reward_f += 1
                # resolution reward
                if self.user[i]['res'] == 0:  # 720
                    reward_r += 0.4
                    r_resol += 4
                elif self.user[i]['res'] == 1:  # 1080
                    reward_r += 0.44
                    r_resol += 4.4
                elif self.user[i]['res'] == 2:  # 2k
                    reward_r += 0.48
                    r_resol += 4.8
                elif self.user[i]['res'] == 3:  # 3k
                    reward_r += 0.49
                    r_resol += 4.9

            # consecutive loss penalty
            current_segment_length = 0
            loss_penalty = 0
            copy_penalty = 0

            for loss in self.loss[i]:
                if loss == 0:  # 0 for loss and 1 for receive
                    current_segment_length += 1

                # if receive:
                elif current_segment_length > 0:
                    segment_length[i].append(current_segment_length)
                    loss_penalty += current_segment_length ** 2
                    current_segment_length = 0

            # append the end tile segment (frame copying)
            if self.count == 89:
                if current_segment_length > 0:
                    segment_length[i].append(current_segment_length)
                    copy_penalty += current_segment_length ** 2
                    loss_penalty += current_segment_length ** 2
                    copying_length.append(current_segment_length)

            reward_r -= (loss_penalty+copy_penalty)/100
            reward_f -= (loss_penalty+copy_penalty)/100

        #ave_consecutive = np.mean(segment_lengths)

        # every time step
        self.count += 1
        self.left_frame -= 1

        self.channel_gain = rician(self.user_num, self.res_num)
        for i in range(self.user_num):
            # update channel gain
            dist = np.linalg.norm(np.array((self.user[i]['x'], self.user[i]['y']))
                                  - np.array((channel_loc[0], channel_loc[1])))
            path_loss = (3e8 / (4 * math.pi * dist * 28e9)) ** 2
            self.user[i]['g'] = (math.sqrt(self.K / (self.K + 1)) + math.sqrt(1 / self.K) * self.channel_gain[i][0]) * path_loss

        # change state
        self.state = [self.left_frame]

        for i in range(self.user_num):
            for j in ['comp', 'g']:
                self.state.append(self.user[i][j])

        done = False

        if self.left_frame <= 0:
            done = True

        fps = 0
        # average resolution score
        ave_resol_score = r_resol / self.user_num
        ave_resol_change = resol_change / self.user_num
        if done:
            # average consecutive losses
            ave_consecutive = sum([sum(user_consecutive) for user_consecutive in segment_length])/self.user_num
            # average frame copyings
            ave_copying = sum(copying_length)/self.user_num
            # average fps
            fps = np.mean(self.user['fps'])
            self.reset()
        # print(reward)

        reward_p = reward_f
        # 5 for regularization
        return np.array(self.state, dtype=np.float32), reward_r/5, reward_f/5, reward_p/5, done, 1, fps, ave_consecutive, ave_copying, ave_resol_score, ave_resol_change

    def render(self, mode='human'):
        pass