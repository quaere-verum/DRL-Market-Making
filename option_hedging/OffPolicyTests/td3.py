import gymnasium as gym
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.policy import TD3Policy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
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


def td3_trial(trainer_kwargs: Dict[str, int],
              env_kwargs: Dict[str, Any],
              tau: float,
              buffer_size: int,
              lr: float,
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
    actor_net = Net(state_shape=env.observation_space.shape,
                    hidden_sizes=net_arch, device=device)
    critic_state_shape = (env.observation_space.shape[0] + 1,)
    critic_net = Net(state_shape=critic_state_shape,
                     hidden_sizes=net_arch, concat=True, device=device)
    critic2_net = Net(state_shape=critic_state_shape,
                      hidden_sizes=net_arch, concat=True, device=device)
    actor = Actor(preprocess_net=actor_net, action_shape=env.action_space.shape, device=device).to(device)
    critic = Critic(preprocess_net=critic_net, device=device).to(device)
    critic2 = Critic(preprocess_net=critic2_net, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=lr)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)

    policy = TD3Policy(
        actor=actor,
        critic=critic,
        critic2=critic2,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        critic2_optim=critic2_optim,
        action_space=env.action_space,
        action_scaling=False,
        tau=tau,
        exploration_noise='default'
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
