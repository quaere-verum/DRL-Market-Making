import gymnasium as gym
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.trainer import OffpolicyTrainer
from utils.torch_modules import PreprocessNet
from tianshou.utils import TensorboardLogger
from tianshou.utils.lr_scheduler import MultipleLRSchedulers
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


def sac_trial(trainer_kwargs: Dict[str, int],
              env_kwargs: Dict[str, Any],
              policy_kwargs: Dict[str, Any],
              net_kwargs: Dict[str, Tuple[int]],
              lr_scheduler_kwargs: Dict[str, Any],
              buffer_size: int,
              lr: float,
              subproc: bool = False
              ):

    env = gym.make('OptionHedgingEnv', epsilon=0, action_bins=env_kwargs['action_bins'])
    if subproc:
        train_envs = SubprocVectorEnv([make_env(seed=k, **env_kwargs) for k in range(20)])
        test_envs = SubprocVectorEnv([make_env(seed=k * 50, **env_kwargs) for k in range(10)])
    else:
        train_envs = DummyVectorEnv([make_env(seed=k, **env_kwargs) for k in range(20)])
        test_envs = DummyVectorEnv([make_env(seed=k * 50, **env_kwargs) for k in range(10)])
    if env_kwargs['action_bins'] == 0:
        from tianshou.utils.net.continuous import ActorProb, Critic
        actor_net = PreprocessNet(state_shape=env.observation_space.shape, device=device, **net_kwargs)
        critic_state_shape = (env.observation_space.shape[0] + 1,)
        critic_net = PreprocessNet(state_shape=critic_state_shape, device=device, **net_kwargs)
        critic2_net = PreprocessNet(state_shape=critic_state_shape, device=device, **net_kwargs)
        actor = ActorProb(preprocess_net=actor_net, action_shape=env.action_space.shape, device=device).to(device)
        critic = Critic(preprocess_net=critic_net, device=device).to(device)
        critic2 = Critic(preprocess_net=critic2_net, device=device).to(device)
    else:
        from tianshou.utils.net.discrete import Actor, Critic
        actor_net = PreprocessNet(state_shape=env.observation_space.shape, device=device, **net_kwargs)
        critic_net = PreprocessNet(state_shape=env.observation_space.shape, device=device, **net_kwargs)
        critic2_net = PreprocessNet(state_shape=env.observation_space.shape, device=device, **net_kwargs)
        actor = Actor(preprocess_net=actor_net, action_shape=env.action_space.n, device=device).to(device)
        critic = Critic(preprocess_net=critic_net, last_size=env.action_space.n, device=device).to(device)
        critic2 = Critic(preprocess_net=critic2_net, last_size=env.action_space.n, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=lr)
    if lr_scheduler_kwargs is not None:
        actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=actor_optim,
                                                               start_factor=1,
                                                               **lr_scheduler_kwargs)
        critic_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=critic_optim,
                                                                start_factor=1,
                                                                **lr_scheduler_kwargs)
        critic2_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=critic2_optim,
                                                                 start_factor=1,
                                                                 **lr_scheduler_kwargs)
        lr_scheduler = MultipleLRSchedulers(*[actor_lr_scheduler,
                                              critic_lr_scheduler,
                                              critic2_lr_scheduler])
    else:
        lr_scheduler = None
    if env_kwargs['action_bins'] == 0:
        from tianshou.policy import SACPolicy
        policy = SACPolicy(
            actor=actor,
            critic=critic,
            critic2=critic2,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            critic2_optim=critic2_optim,
            action_space=env.action_space,
            lr_scheduler=lr_scheduler,
            **policy_kwargs
        )
    else:
        from tianshou.policy import DiscreteSACPolicy
        policy = DiscreteSACPolicy(
            actor=actor,
            critic=critic,
            critic2=critic2,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            critic2_optim=critic2_optim,
            action_space=env.action_space,
            lr_scheduler=lr_scheduler,
            **policy_kwargs
        )
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(buffer_size, len(train_envs)))
    train_collector.collect(buffer_size, random=True)
    test_collector = Collector(policy, test_envs)
    return OffpolicyTrainer(
                policy=policy,
                train_collector=train_collector,
                test_collector=test_collector,
                logger=logger,
                **trainer_kwargs
            )
