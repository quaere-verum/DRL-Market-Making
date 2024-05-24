import gymnasium as gym
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
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
from phase_1.gym_envs import make_env

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
    epsilon = 0.1
    rho = 0.3
    env = gym.make('MarketMakingEnv', epsilon=epsilon)
    train_envs = SubprocVectorEnv([make_env(epsilon=epsilon,
                                            rho=rho,
                                            action_bins=10,
                                            duration_bounds=(6, 12),
                                            benchmark=False,
                                            seed=k) for k in range(20)])
    test_envs = SubprocVectorEnv([make_env(epsilon=epsilon,
                                           rho=0.,
                                           action_bins=10,
                                           duration_bounds=(6, 12),
                                           benchmark=True,
                                           seed=k*50) for k in range(10)])
    net = Net(state_shape=env.observation_space.shape,
              hidden_sizes=[64, 32, 16, 4], concat=True, device=device)
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    tau = 0.01

    policy = DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=0.99,
        estimation_step=1,
        target_update_freq=0,
        is_double=True,
        clip_loss_grad=True,
        observation_space=env.observation_space
    )
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(2000, len(train_envs)))
    test_collector = Collector(policy, test_envs)
    train_result = OffpolicyTrainer(
        policy=policy,
        batch_size=256,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=25,
        step_per_epoch=2000,
        repeat_per_collect=5,
        episode_per_test=1000,
        step_per_collect=500,
        logger=logger
    ).run()
