import gymnasium as gym
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from utils.torch_modules import QNet
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


def epsilon_greedy_scheduler(epsilon_greedy, policy):
    if epsilon_greedy is None:
        return None

    def train_fn(_, steps_taken):
        policy.set_eps(max(epsilon_greedy['start']*(1-steps_taken/epsilon_greedy['max_steps']), epsilon_greedy['end']))
    return train_fn


def dqn_trial(trainer_kwargs: Dict[str, int],
              env_kwargs: Dict[str, Any],
              policy_kwargs: Dict[str, Any],
              net_kwargs: Dict[str, Tuple[int]],
              buffer_size: int,
              lr: float,
              epsilon_greedy: Dict[str, float],
              subproc: bool = False,
              ) -> OffpolicyTrainer:
    if env_kwargs['action_bins'] == 0:
        new_bins = int(input('DQN requires discrete action space. Set new action_bins:\n'))
        assert new_bins > 0
        env_kwargs['action_bins'] = new_bins
    env = gym.make('OptionHedgingEnv', epsilon=0)
    if subproc:
        train_envs = SubprocVectorEnv([make_env(seed=k, **env_kwargs) for k in range(20)])
        test_envs = SubprocVectorEnv([make_env(seed=k * 50, **env_kwargs) for k in range(10)])
    else:
        train_envs = DummyVectorEnv([make_env(seed=k, **env_kwargs) for k in range(20)])
        test_envs = DummyVectorEnv([make_env(seed=k * 50, **env_kwargs) for k in range(10)])
    net = QNet(state_shape=env.observation_space.shape,
               action_shape=env.action_space.shape,
               device=device,
               **net_kwargs)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    policy = DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        observation_space=env.observation_space,
        **policy_kwargs
    )
    train_fn = epsilon_greedy_scheduler(epsilon_greedy=epsilon_greedy,
                                        policy=policy)
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
