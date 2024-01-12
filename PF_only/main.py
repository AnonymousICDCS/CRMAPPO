import time

import torch
import numpy as np
from Policy import device, PPO
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from environment import action_map
from environment import video

def encode_action(user_resolution_list, usr, res):
    x = 0
    for i in range(usr):
        x += user_resolution_list[i] * (res ** (usr - 1 - i))
    return int(x)

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--Loadreward', type=str2bool, default=False, help='Load pretrained reward networks')
parser.add_argument('--Savereward', type=str2bool, default=False, help='Save reward network')
parser.add_argument('--ModelIdex', type=int, default=400000, help='which model to load')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--T_horizon', type=int, default=1000, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=200000, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=500, help='Model evaluating interval, in steps.')
parser.add_argument('--beta', type=float, default=0.0, help='Variance Factor')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--v_lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
parser.add_argument('--user_num', type=int, default=8, help='number of users')
opt = parser.parse_args()
print(opt)


def Action_adapter(a, p_max):
    #return a * (max_action-min_action) + min_action
    return a * p_max

def evaluate_policy(env, model, f_max, p_max, render):
    scores = 0
    turns = 3
    all_fps, all_consecutive, all_copying, all_resol, all_change = 0, 0, 0, 0, 0
    for j in range(turns):
        action = []
        s, done = env.reset(), False
        ep_r, ep_resol, ep_change = 0, 0, 0
        while not done:
            # Take deterministic actions at test time
            start = time.perf_counter()
            a_r = encode_action([1 for _ in range(env.user_num)], env.user_num, env.res_num)
            s_f = np.append(s, a_r)
            a_f, log_f = model.evaluate(s_f, 'frequency')
            act_f = Action_adapter(a_f, f_max)
            s_p = np.append(s_f, a_f)
            a_p, log_p = model.evaluate(s_p, 'power')
            act_p = Action_adapter(a_p, p_max)
            end = time.perf_counter()
            s_prime, r_r, r_f, r_p, done, info, fps, ave_consecutive, ave_copying, resol_score, resol_change = env.step(a_r, act_f, act_p)
            # r_r is the total reward since r_f and r_p are extracted from it
            ep_r += r_r
            ep_resol += resol_score
            ep_change += resol_change
            s = s_prime
        all_fps += fps
        all_consecutive += ave_consecutive
        all_copying += ave_copying
        all_resol += ep_resol
        all_change += ep_change
        scores += ep_r
    return scores/turns, all_fps/turns, all_consecutive/turns, all_copying/turns, all_resol/turns, all_change/turns

def main():
    env_with_Dead = [True, True]
    EnvIdex = opt.EnvIdex
    env = video(opt.user_num)
    eval_env = video(opt.user_num)
    state_dim = env.observation_space.shape[0]
    action_r_dim = env.action_r_space.n
    action_f_dim = env.action_f_space.shape[0]
    action_p_dim = env.action_p_space.shape[0]
    max_e_steps = env.max_step
    p_max = env.p_max
    f_max = env.f_max

    write = opt.write
    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}-user-seed{}-beta{}'.format(env.user_num, opt.seed, opt.beta) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    T_horizon = opt.T_horizon
    render = opt.render
    Loadmodel = opt.Loadmodel
    ModelIdex = opt.ModelIdex #which model to load
    Loadreward = opt.Loadreward
    Savereward = opt.Savereward
    Max_train_steps = opt.Max_train_steps #in steps
    eval_interval = opt.eval_interval #in steps
    save_interval = opt.save_interval #in steps

    seed = opt.seed
    torch.manual_seed(seed)
    env.seed(seed)
    eval_env.seed(seed)
    np.random.seed(seed)

    print('Env:',env.user_num,'  state_dim:',state_dim,' action_r_dim:', action_r_dim,'  action_f_dim,action_p_dim:', action_f_dim, action_p_dim, '   Random Seed:',seed, '  max_e_steps:',max_e_steps)
    print('\n')

    kwargs = {
        "env_with_Dead": env_with_Dead[EnvIdex],
        "state_dim": state_dim,
        "action_r_dim": action_r_dim,
        "action_f_dim": action_f_dim,
        "action_p_dim": action_p_dim,
        "beta": opt.beta,
        "gamma": opt.gamma,
        "lambd": opt.lambd,
        "net_width": opt.net_width,
        "a_lr": opt.a_lr,
        "v_lr": opt.v_lr,
        "clip_rate": opt.clip_rate,
        "K_epochs": opt.K_epochs,
        "batch_size": opt.batch_size,
        "l2_reg": opt.l2_reg,
        "entropy_coef": opt.entropy_coef,  #hard env needs large value
        "adv_normalization": opt.adv_normalization,
        "entropy_coef_decay": opt.entropy_coef_decay,
        "user_num": opt.user_num
    }

    if not os.path.exists('model'): os.mkdir('model')
    model = PPO(**kwargs)
    if Loadmodel: model.load(ModelIdex)
    if Loadreward: model.load_reward(ModelIdex)

    traj_lenth = 0
    total_steps = 0
    while total_steps < Max_train_steps:
        s, done, steps, ep_r = env.reset(), False, 0, 0

        '''Interact & trian'''
        while not done:
            traj_lenth += 1
            steps += 1
            a_r = encode_action([1 for _ in range(env.user_num)], env.user_num, env.res_num)
            pi_r=0
            s_f = np.append(s, a_r)
            a_f, log_f = model.select_action(s_f, 'frequency')
            a_f_sample, _ = model.sample_action(s_f, 'frequency')
            act_f = Action_adapter(a_f, f_max)
            s_p = np.append(s_f, a_f)
            a_p, log_p = model.select_action(s_p, 'power')
            a_p_sample, _ = model.sample_action(s_p, 'power')
            act_p = Action_adapter(a_p, p_max)
            s_prime, r_r, r_f, r_p, done, info, fps, ave_consecutive, ave_copying, ave_resol, resol_change = env.step(a_r, act_f, act_p)

            if (done and steps != max_e_steps):
                dw = False  #dw: dead and win
            else:
                dw = False
            model.put_data((s, a_r, a_f, a_p, a_f_sample, a_p_sample, r_r, r_f, r_p, s_prime, pi_r, log_f, log_p, done, dw))  # r_g = r_c in this scenario
            s = s_prime

            '''update if its time'''
            if traj_lenth % T_horizon == 0:
                a_r_loss, a_f_loss, a_p_loss, v_loss, r_d_loss, r_c_loss = model.train()

                traj_lenth = 0
                if write:
                    writer.add_scalar('a_c_loss', a_r_loss, global_step=total_steps)
                    writer.add_scalar('a_f_loss', a_f_loss, global_step=total_steps)
                    writer.add_scalar('a_p_loss', a_p_loss, global_step=total_steps)
                    writer.add_scalar('v_loss', v_loss, global_step=total_steps)
                    writer.add_scalar('r_d_loss', r_d_loss, global_step=total_steps)
                    writer.add_scalar('r_c_loss', r_c_loss, global_step=total_steps)

            '''record & log'''
            if total_steps % eval_interval == 0:
                score, fps, ave_consecutive, ave_copying, ave_resol, resol_change = evaluate_policy(eval_env, model, f_max, p_max, False)

                if write:
                    writer.add_scalar('ep_r', score, global_step=total_steps)
                    writer.add_scalar('ep_fps', fps, global_step=total_steps)
                    writer.add_scalar('ep_consecutive', ave_consecutive, global_step=total_steps)
                    writer.add_scalar('ep_copying', ave_copying, global_step=total_steps)
                    writer.add_scalar('ep_resol_score', ave_resol, global_step=total_steps)
                    writer.add_scalar('ep_resol_change', resol_change, global_step=total_steps)
                print('seed:', seed, 'steps: {}'.format(int(total_steps)), 'score:', score, 'ave_fps:', fps, 'ave_consecutive:', ave_consecutive,
                      'ave_copying', ave_copying, 'resol_score:', ave_resol, 'resol_change:', resol_change)
            total_steps += 1

            '''save model'''
            if total_steps % save_interval == 0:
                model.save(total_steps)
                if Savereward:
                    model.save_reward(total_steps)

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()