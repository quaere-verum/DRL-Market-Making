import gymnasium as gym
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import numpy as np
import os
import shutil
from pathlib import Path

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

gym.envs.register('MarketMakingEnv', 'phase_1.gym_envs:MarketMakingEnv')

if __name__ == '__main__':
    epsilon = 0.01
    rho = 0.1
    env = gym.make('MarketMakingEnv', epsilon=epsilon, seed=seed_value)
    train_envs = SubprocVectorEnv([lambda: gym.make('MarketMakingEnv',
                                                    epsilon=epsilon,
                                                    rho=rho,
                                                    seed=seed_value * k,
                                                    duration_bounds=(6, 12)) for k in range(20)])
    test_envs = SubprocVectorEnv([lambda: gym.make('MarketMakingEnv',
                                                   epsilon=epsilon,
                                                   rho=0.,
                                                   seed=seed_value * k,
                                                   duration_bounds=(6, 12),
                                                   benchmark=True) for k in range(10)])
    net = Net(state_shape=env.observation_space.shape, hidden_sizes=[64, 32, 16, 4], device=device)

    actor = Actor(preprocess_net=net, action_shape=env.action_space.shape, device=device).to(device)
    critic = Critic(preprocess_net=net, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.001)

    def dist_fn(mean):
        return torch.distributions.Normal(mean, 0.2)
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        action_space=env.action_space,
        action_scaling=False
    )
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(2000, len(train_envs)))
    test_collector = Collector(policy, test_envs)
    train_result = OnpolicyTrainer(
        policy=policy,
        batch_size=256,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=10,
        step_per_epoch=100000,
        repeat_per_collect=5,
        episode_per_test=100,
        step_per_collect=1000,
        logger=logger
    ).run()
