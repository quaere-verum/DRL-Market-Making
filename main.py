import gymnasium as gym
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import Actor, Critic
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

gym.envs.register('MarketMakingEnv', 'phase_1.gym_envs:MarketMakingEnv')

if __name__ == '__main__':
    env = gym.make('MarketMakingEnv', epsilon=0.1)
    train_envs = DummyVectorEnv([lambda: gym.make('MarketMakingEnv', epsilon=0.1) for _ in range(20)])
    test_envs = DummyVectorEnv([lambda: gym.make('MarketMakingEnv', epsilon=0.1) for _ in range(10)])
    net = Net(state_shape=env.observation_space.shape, hidden_sizes=[256, 128, 64, 32], device=device)
    actor = Actor(preprocess_net=net, action_shape=env.action_space.shape, device=device).to(device)
    critic = Critic(preprocess_net=net, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.01)

    def dist_fn(mean):
        return torch.distributions.Normal(mean, 0.1)
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        action_space=env.action_space,
        action_scaling=False,
    )
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)))
    test_collector = Collector(policy, test_envs)
    train_result = OnpolicyTrainer(
        policy=policy,
        batch_size=256,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=10,
        step_per_epoch=50000,
        repeat_per_collect=10,
        episode_per_test=10,
        step_per_collect=2000,
    ).run()
