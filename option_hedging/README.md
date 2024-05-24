# Phase 1: Delta Neutrality
To prototype a model, we begin with the simplest case where the agent's only task is to maintain a delta neutral portfolio. All prices will be relative to $S(0)$, the price of the underlying at the time the option is written, so we set $S(0)=100$. We will assume that we are hedging a short position in a call option, for it stands to reason that if the agent can learn this task, then put-call parity implies that it can also learn how to hedge a short position in a put option.
## Designing the environment
As a first step we design the environment in which our agents get to act. More specifically, this means we need to:
1. Determine the actions our agent is allowed to take
2. Determine the state that is returned after each step

In phase 1, the action space is simply $\mathcal{A}=[0,1+\varepsilon]$, where $\varepsilon > 0$ is a hyperparameter of the model. The reason for this is that we assume that Black-Scholes offers a good approximation to reality, and in this case, the hedging amount is $\Phi(d_+)$ where $\Phi:\mathbb{R}\to[0,1]$ is the cumulative distribution function of the standard normal. Our action $a$ will be the replacement for $\Phi(d_+)$, and so we expect that the optimal action $a^*\in[0,1]$. To give additional room for exploration, we add the constant epsilon. Furthermore, we might discretise the action space in order to be able to employ specific reinforcement learning algorithms like DQN.

The state space is given by the parameters which are used in the Black-Scholes model: the price $S(t)$, the remaining time until expiry $T-t$, the amount of stock currently held, the strike price $K$, and a forecast of the volatility $\widetilde{\sigma}$ (if not assumed to be constant). Additionally, we will include the Black-Scholes hedge as part of the state, to give the agent an anchor. Our deep neural network will be predicting the residual of the optimal hedging amount, compared to the Black-Scholes solution.

## Designing the reward function
Let $P_t$ denote the value of our portfolio at time $t$. This consists of the stock, and the option. Let we let our reward at time $t$ be $P_t-P_{t-1}-c_t-\rho(P_t-P_{t-1})^2$, where $\rho$ is a hyperparameter that determines how much we punish variance in the portfolio value.
### Note
We will include a "benchmark" parameter in the environment. If set to `True`, the reward will be the policy's reward relative to the benchmark reward (i.e. Black-Scholes).  

# Experimental observations
## Constant volatility
### 23/05/2024
Both a random agent and a basic Black-Scholes agent will, on average, make a loss. From experiments, it is quite evident that the on-policy algorithms are able to outperform both of these on average, and in fact make a profit. However, the standard deviation is much greater, even when increasing the value of $\rho$. Possible solutions:
1. Generate more training data
2. Discretise the action space
3. Enhance the network architecture and/or engineer relevant features for the model

### 24/05/2024
After trying some different network architectures, it appears PPO can significantly outperform the baseline,
both in terms of mean reward and in terms of standard deviation.
This was observed within a relatively short training process (~100k steps), so it is likely that performance
can be increased with better hyperparameters, network architecture, and data collection.

Benchmark: -17.02 +/- 21.23
PPO: -5.39 +/- 8.78

Parameters used:
```python
trainer_kwargs = {
        'max_epoch': 10,
        'batch_size': 512,
        'step_per_epoch': 10000,
        'repeat_per_collect': 5,
        'episode_per_test': 1000,
        'update_per_step': 1.,
        'step_per_collect': 2000,
        'verbose': True,
        'show_progress': True
}
ppo_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'epsilon': 0.1,
        'sigma': 0.05,
        'rho': 0.2,
        'action_bins': 32,
        'duration_bounds': (6, 12),
        'buffer_size': 6000,
        'lr': 0.001,
        'subproc': False,
        'net_arch': tuple(32 for k in range(10)),
        'dist_std': 0.1
}
```

