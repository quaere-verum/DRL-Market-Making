import gymnasium as gym
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import numpy as np
import os
import shutil
from pathlib import Path
from option_hedging.gym_envs import make_env

log_dir = os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent.absolute(), '.logs')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir, exist_ok=True)
logger = TensorboardLogger(SummaryWriter(log_dir=log_dir), train_interval=256)

seed_value = 123
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'

gym.envs.register('OptionHedgingEnv', 'option_hedging.gym_envs:OptionHedgingEnv')


def dqn_trial(trainer_kwargs,
              epsilon,
              sigma,
              rho,
              discount_factor,
              estimation_step,
              target_update_freq,
              is_double,
              clip_loss_grad,
              action_bins,
              duration_bounds,
              buffer_size,
              lr,
              subproc=False,
              net_arch=(64, 32, 16, 4)
              ):

    env = gym.make('OptionHedgingEnv', epsilon=0)
    if subproc:
        train_envs = SubprocVectorEnv([make_env(epsilon=epsilon,
                                                sigma=sigma,
                                                rho=rho,
                                                action_bins=action_bins,
                                                duration_bounds=duration_bounds,
                                                seed=k) for k in range(20)])
        test_envs = SubprocVectorEnv([make_env(epsilon=epsilon,
                                               sigma=sigma,
                                               rho=0.,
                                               action_bins=action_bins,
                                               duration_bounds=duration_bounds,
                                               seed=k * 50) for k in range(10)])
    else:
        train_envs = DummyVectorEnv([make_env(epsilon=epsilon,
                                              sigma=sigma,
                                              rho=rho,
                                              action_bins=action_bins,
                                              duration_bounds=duration_bounds,
                                              seed=k) for k in range(20)])
        test_envs = DummyVectorEnv([make_env(epsilon=epsilon,
                                             sigma=sigma,
                                             rho=0.,
                                             action_bins=action_bins,
                                             duration_bounds=duration_bounds,
                                             seed=k * 50) for k in range(10)])
    net = Net(state_shape=env.observation_space.shape,
              hidden_sizes=net_arch, concat=True, device=device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    policy = DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=discount_factor,
        estimation_step=estimation_step,
        target_update_freq=target_update_freq,
        is_double=is_double,
        clip_loss_grad=clip_loss_grad,
        observation_space=env.observation_space
    )

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)
    return OffpolicyTrainer(
                policy=policy,
                train_collector=train_collector,
                test_collector=test_collector,
                logger=logger,
                **trainer_kwargs
            )
