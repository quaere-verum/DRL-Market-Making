import gymnasium as gym

gym.envs.register('MarketMakingEnv', 'phase_1.gym_envs:MarketMakingEnv')

if __name__ == '__main__':
    env = gym.make('MarketMakingEnv', epsilon=0.1)
    n_test_episodes = 1000
    for _ in range(n_test_episodes):
        obs, info = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            action = env.unwrapped.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
