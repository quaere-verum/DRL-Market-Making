import gymnasium as gym
import numpy as np

gym.envs.register('OptionHedgingEnv', 'option_hedging.gym_envs:OptionHedgingEnv')


def black_scholes_benchhmark(env, n_trials):
    rewards = np.zeros(n_trials)
    for trial in range(n_trials):
        _, _ = env.reset()
        total_reward = 0
        done, truncated = False, False
        while not done and not truncated:
            action = np.array([env.portfolio.black_scholes_hedge(env.sigma)])
            _, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        rewards[trial] = total_reward
    return np.mean(rewards), np.std(rewards)


if __name__ == '__main__':
    epsilon = 0.1
    env = gym.make('OptionHedgingEnv', duration_bounds=(6, 12))
    n_trials = 1000
    mean, std = black_scholes_benchhmark(env=env, n_trials=n_trials)
    print(mean, std)