import argparse
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from hppo.hppo import *
from hppo.hppo_utils import *
from envs.sensornet import sensornet


class Trainer(object):
    """
    A RL trainer.
    """

    def __init__(self, args):
        self.experiment_name = args.experiment_name

        self.device = args.device
        self.max_episodes = args.max_episodes
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.agent_save_freq = args.agent_save_freq
        self.agent_update_freq = args.agent_update_freq

        # agent's hyperparameters
        self.mid_dim = args.mid_dim
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_actor_param
        self.lr_std = args.lr_std
        self.lr_decay_rate = args.lr_decay_rate
        self.target_kl_dis = args.target_kl_dis
        self.target_kl_con = args.target_kl_con
        self.gamma = args.gamma
        self.lam = args.lam
        self.epochs_update = args.epochs_update
        self.v_iters = args.v_iters
        self.eps_clip = args.eps_clip
        self.max_norm_grad = args.max_norm_grad
        self.init_log_std = args.init_log_std
        self.coeff_dist_entropy = args.coeff_dist_entropy
        self.action_con_low = np.array([0, 0])
        self.action_con_high = np.array([360, 30])
        self.random_seed = args.random_seed
        #self.if_use_active_selection = args.if_use_active_selection

        # For save
        self.save_path = 'log/' + self.experiment_name + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.env = sensornet()
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dis_dim = self.env.action_space[0].n  # Discrete 动作维度
        self.action_dis_len = self.env.action_dis_len
        self.action_con_dim = self.env.action_space[1].shape[0]  # Box 动作维度

        self.history = {}
        log_dir = os.path.join("runs", f"{self.experiment_name}")
        self.writer = SummaryWriter(log_dir=log_dir)

    def push_history_dis(self, obs, act_dis, logp_act_dis, val):
        self.history = {
            'obs': obs,
            'act_dis': act_dis,
            'logp_act_dis': logp_act_dis,
            'val': val
        }

    def push_history_hybrid(self, obs, act_dis, act_con, logp_act_dis, logp_act_con, val):
        self.history = {
            'obs': obs,
            'act_dis': act_dis,
            'act_con': act_con,
            'logp_act_dis': logp_act_dis,
            'logp_act_con': logp_act_con,
            'val': val
        }

    def unbatchify(self, value_action_logp: dict):
        state_value = value_action_logp[0]
        actions = value_action_logp[1]
        logp_actions = value_action_logp[2]

        # actions = np.array([action_dis, action_con])
        # logp_actions = np.array([log_prob_dis, log_prob_con])

        return state_value, actions, logp_actions

    def initialize_agents(self, random_seed):
        """
        Initialize environment and agent.

        :param random_seed: could be regarded as worker index
        :return: instance of agent and env
        """

        # return PPO_Hybrid(self.obs_dim, self.action_dis_dim, self.action_len, self.action_con_dim, self.mid_dim, self.lr_actor, self.lr_critic, self.lr_decay_rate, self.buffer_size, self.target_kl_dis, self.target_kl_con, self.gamma, self.lam, self.epochs_update,self.v_iters, self.eps_clip, self.max_norm_grad, self.coeff_dist_entropy, random_seed, self.device, self.lr_std, self.init_log_std, self.if_use_active_selection)
        return PPO_Hybrid(self.obs_dim,
                          self.action_dis_dim,
                          self.action_dis_len,
                          self.action_con_dim,
                          self.mid_dim,
                          self.lr_actor,
                          self.lr_critic,
                          self.lr_decay_rate,
                          self.buffer_size,
                          self.target_kl_dis,
                          self.target_kl_con,
                          self.gamma,
                          self.lam,
                          self.epochs_update,
                          self.v_iters,
                          self.eps_clip,
                          self.max_norm_grad,
                          self.coeff_dist_entropy,
                          random_seed,
                          self.device,
                          self.lr_std,
                          self.init_log_std,
                          self.action_con_low,
                          self.action_con_high,
                          # self.if_use_active_selection
                          )

    def train(self, worker_idx):
        """

        :param worker_idx:
        :return:
        """

        agent = self.initialize_agents(worker_idx)

        norm_mean = np.zeros(self.obs_dim)
        norm_std = np.ones(self.obs_dim)

        i_episode = 0

        ### TRAINING LOGIC ###
        while i_episode < self.max_episodes:
            # collect an episode
            with torch.no_grad():
                state, info = self.env.reset()
                next_state = state
                total_reward = 0

                while True:
                    # Every update, we will normalize the state_norm(the input of the actor_con and critic) by
                    # mean and std retrieve from the last update's buf, in other word observations normalization
                    observations_norm = (state - norm_mean) / np.maximum(norm_std, 1e-6)
                    # Select action with policy
                    value_action_logp = agent.select_action(observations_norm)
                    values, actions, logp_actions = self.unbatchify(value_action_logp)

                    next_state, reward, done, info = self.env.step(actions)

                    self.push_history_dis(state, actions, logp_actions, values)
                    agent.buffer.store_dis(self.history['obs'], self.history['act_dis'],
                                           reward, self.history['val'], self.history['logp_act_dis'])

                    total_reward += reward

                    state = next_state

                    if done:
                        if i_episode % 100 == 0:
                            print("sensor_remaining_data:\n", self.env.sensor_remaining_data)
                            print("sensor_estimated_positions:\n", self.env.sensor_estimated_positions)
                            print("sensor_estimated_radii:\n", self.env.sensor_estimated_radii)
                            # 从 info 中提取数据
                            distance_errors = info.get("distance_errors", [])
                            # 记录到 TensorBoard
                            for i in range(self.env.sensor_num):
                                # 记录剩余数据
                                self.writer.add_scalar(
                                    f'Sensor_{i}/RemainingData',
                                    self.env.sensor_remaining_data[i],
                                    i_episode
                                )
                                # 记录估计半径
                                self.writer.add_scalar(
                                    f'Sensor_{i}/EstimatedRadius',
                                    self.env.sensor_estimated_radii[i],
                                    i_episode
                                )
                                # 记录估计位置（拆分为 x 和 y 坐标）
                                self.writer.add_scalar(
                                    f'Sensor_{i}/Position_X',
                                    self.env.sensor_estimated_positions[i][0],  # x 坐标
                                    i_episode
                                )
                                self.writer.add_scalar(
                                    f'Sensor_{i}/Position_Y',
                                    self.env.sensor_estimated_positions[i][1],  # y 坐标
                                    i_episode
                                )
                                self.writer.add_scalar(
                                    f'Sensor_{i}/EstimateErrors',
                                    distance_errors[i],  # y 坐标
                                    i_episode
                                )
                    i_episode += 1
                    agent.buffer.finish_path(0)
                    break
                print(f"Episode {i_episode} - Total Reward: {total_reward}")
                self.writer.add_scalar('Reward/Total', total_reward, i_episode)

            if i_episode % self.agent_update_freq == 0:
                norm_mean = agent.buffer.filter()[0]
                norm_std = agent.buffer.filter()[1]
                if i_episode > self.agent_save_freq:
                    agent.update(self.batch_size)
                agent.buffer.clear()
        self.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device.')
    parser.add_argument('--max_episodes', type=int, default=3000, help='The max episodes per agent per run.')
    parser.add_argument('--buffer_size', type=int, default=6000, help='The maximum size of the PPOBuffer.')
    parser.add_argument('--batch_size', type=int, default=64, help='The sample batch size.')
    parser.add_argument('--agent_save_freq', type=int, default=10, help='The frequency of the agent saving.')
    parser.add_argument('--agent_update_freq', type=int, default=10, help='The frequency of the agent updating.')
    parser.add_argument('--lr_actor', type=float, default=0.0003, help='The learning rate of actor_con.')  # carefully!
    parser.add_argument('--lr_actor_param', type=float, default=0.001, help='The learning rate of critic.')
    parser.add_argument('--lr_std', type=float, default=0.004, help='The learning rate of log_std.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.995, help='Factor of learning rate decay.')
    parser.add_argument('--mid_dim', type=list, default=[256, 128, 64], help='The middle dimensions of both nets.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted of future rewards.')
    parser.add_argument('--lam', type=float, default=0.8,
                        help='Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)')
    parser.add_argument('--epochs_update', type=int, default=20,
                        help='Maximum number of gradient descent steps to take on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)')
    parser.add_argument('--v_iters', type=int, default=1,
                        help='Number of gradient descent steps to take on value function per epoch.')
    parser.add_argument('--target_kl_dis', type=float, default=0.025,
                        help='Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)')
    parser.add_argument('--target_kl_con', type=float, default=0.05,
                        help='Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='The clip ratio when calculate surr.')
    parser.add_argument('--max_norm_grad', type=float, default=5.0, help='max norm of the gradients.')
    parser.add_argument('--init_log_std', type=float, default=-1.0,
                        help='The initial log_std of Normal in continuous pattern.')
    parser.add_argument('--coeff_dist_entropy', type=float, default=0.005,
                        help='The coefficient of distribution entropy.')
    parser.add_argument('--random_seed', type=int, default=1, help='The random seed.')
    parser.add_argument('--record_mark', type=str, default='renaissance',
                        help='The mark that differentiates different experiments.')
    # parser.add_argument('--if_use_active_selection', type=bool, default=False,
    #                     help='Whether use active selection in the exploration.')
    parser.add_argument('--experiment_name', type=str, default='hppo', help='The name of the experiment.')

    args = parser.parse_args()

    # training through multiprocess
    trainer = Trainer(args)
    trainer.train(1)
