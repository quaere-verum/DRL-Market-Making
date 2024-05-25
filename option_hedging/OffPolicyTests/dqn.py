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
from typing import Dict, Tuple, Any

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


def epsilon_greedy_scheduler(epsilon_greedy, policy, max_steps):
    if epsilon_greedy is None:
        return None

    def train_fn(_, steps_taken):
        policy.set_eps(max(epsilon_greedy['start']*(1-steps_taken/max_steps), epsilon_greedy['end']))
    return train_fn


def dqn_trial(trainer_kwargs: Dict[str, int],
              env_kwargs: Dict[str, Any],
              discount_factor: float,
              estimation_step: int,
              target_update_freq: int,
              is_double: bool,
              clip_loss_grad: bool,
              buffer_size: int,
              lr: float,
              epsilon_greedy: Dict[str, float],
              subproc: bool = False,
              net_arch: Tuple[int] = (64, 32, 16, 4)
              ) -> OffpolicyTrainer:

    env = gym.make('OptionHedgingEnv', epsilon=0)
    if subproc:
        train_envs = SubprocVectorEnv([make_env(seed=k, **env_kwargs) for k in range(20)])
        test_envs = SubprocVectorEnv([make_env(seed=k * 50, **env_kwargs) for k in range(10)])
    else:
        train_envs = DummyVectorEnv([make_env(seed=k, **env_kwargs) for k in range(20)])
        test_envs = DummyVectorEnv([make_env(seed=k * 50, **env_kwargs) for k in range(10)])
    net = Net(state_shape=env.observation_space.shape,
              hidden_sizes=net_arch, concat=True, device=device).to(device)
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
    max_steps = trainer_kwargs['max_epoch']*trainer_kwargs['step_per_epoch']
    train_fn = epsilon_greedy_scheduler(epsilon_greedy=epsilon_greedy,
                                        policy=policy,
                                        max_steps=max_steps)
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)
    return OffpolicyTrainer(
                policy=policy,
                train_collector=train_collector,
                test_collector=test_collector,
                logger=logger,
                train_fn=train_fn,
                **trainer_kwargs
            )
