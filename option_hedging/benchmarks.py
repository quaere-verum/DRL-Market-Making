import numpy as np
from typing import Tuple
from option_hedging.gym_envs import OptionHedgingEnv, make_env


def black_scholes_benchmark(env: OptionHedgingEnv, n_trials: int) -> Tuple[float, float]:
    rewards = np.zeros(n_trials)
    for trial in range(n_trials):
        _, _ = env.reset()
        total_reward = 0
        done, truncated = False, False
        while not done and not truncated:
            hedge = env.portfolio.black_scholes_hedge(env.sigma)
            if env.action_bins > 0:
                hedge = int((1+env.epsilon)*env.action_bins*hedge)
            action = np.array([hedge])
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


if __name__ == '__main__':
    env = make_env(epsilon=0.01,
                   sigma=0.3,
                   rho=0.,
                   action_bins=20,
                   T=2,
                   rebalance_frequency=20,
                   seed=123,
                   transaction_fees=0.)()
    print(black_scholes_benchmark(env, 100))
    print(random_agent_benchmark(env, 100))
