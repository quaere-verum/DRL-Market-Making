import numpy as np
from typing import Tuple
from option_hedging.gym_envs import OptionHedgingEnv


def black_scholes_benchmark(env: OptionHedgingEnv, n_trials: int, verbose: bool = False) -> Tuple[float, float]:
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


def random_agent_benchmark(env: OptionHedgingEnv, n_trials: int) -> Tuple[float, float]:
    rewards = np.zeros(n_trials)
    for trial in range(n_trials):
        _, _ = env.reset()
        total_reward = 0
        done, truncated = False, False
        while not done and not truncated:
            action = env.action_space.sample()
            _, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        rewards[trial] = total_reward
    return np.mean(rewards), np.std(rewards)
