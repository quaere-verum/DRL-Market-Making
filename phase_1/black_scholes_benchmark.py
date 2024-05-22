import gymnasium as gym
import numpy as np

gym.envs.register('MarketMakingEnv', 'phase_1.gym_envs:MarketMakingEnv')


if __name__ == '__main__':
    epsilon = 0.1
    env = gym.make('MarketMakingEnv', benchmark=True, seed=123, duration_bounds=(6, 24))
    n_test_episodes = 1000
    rewards = np.zeros((n_test_episodes, 2))
    for episode in range(1, n_test_episodes + 1):
        obs, info = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            action = np.random.uniform(0, 1)
            obs, reward, done, truncated, info = env.step([action])
            rewards[episode-1, :] = info['reward'], info['benchmark_reward']
    mean_rewards = np.mean(rewards, axis=0)
    print(f'Mean reward for random agent: {mean_rewards[0]:.3f}')
    print(f'Mean reward for Black-Scholes agent: {mean_rewards[1]:.3f}')