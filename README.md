# Option Hedging
To prototype a model, we begin with the simplest case where the agent's only task is to hedge a portfolio. All prices will be relative to $S(0)$, the price of the underlying at the time the option is written, so we set $S(0)=100$. We will assume that we are hedging a short position in a call option, for it stands to reason that if the agent can learn this task, then put-call parity implies that it can also learn how to hedge a short position in a put option.
## Designing the environment
As a first step we design the environment in which our agents get to act. More specifically, this means we need to:
1. Determine the actions our agent is allowed to take
2. Determine the state that is returned after each step

The action space is simply $\mathcal{A}=[0,1+\varepsilon]$, where $\varepsilon > 0$ is a hyperparameter of the model. The reason for this is that we assume that Black-Scholes offers a good approximation to reality, and in this case, the hedging amount is $\Phi(d_+)$ where $\Phi:\mathbb{R}\to[0,1]$ is the cumulative distribution function of the standard normal. Our action $a$ will correspond to using $a$ as a hedge. We might discretise the action space in order to be able to employ specific reinforcement learning algorithms like DQN.

The state space is given by the parameters which are used in the Black-Scholes model: the price $S(t)$, the remaining time until expiry $T-t$, the amount of stock currently held, the strike price $K$, and a forecast of the volatility $\widetilde{\sigma}$ (if not assumed to be constant). Additionally, we will include the Black-Scholes hedge as part of the state, and potentially other features. Our deep neural network will be predicting the residual of the optimal hedging amount, compared to the Black-Scholes solution.

## Designing the reward function
Let $P_t$ denote the value of our portfolio at time $t$. This consists of the stock, and the option. Let we let our reward at time $t$ be $P_t-P_{t-1}-c_t-\rho(P_t-P_{t-1})^2$, where $\rho$ is a hyperparameter that determines how much we punish variance in the portfolio value.
### Note
Determining the value $P_t$ cannot be done unambiguously, since we hold a short position in the option, which is part of the portfolio. Therefore, we have the following three choices:
1. Use an option pricing model to evaluate the option's value at each point in time
2. Consider the value of the option at each time to be the value it would have if it was exercised at that point in time
3. Consider the value of the option to be $0$ until the expiry date, at which point its value is the payout value

Each of these choices has certain disadvantages. The first requires us to assume an option pricing model, the second valuates the option in a way that is clearly incorrect, and the third would distribute the rewards per step very unevenly making it difficult for the DRL agent to learn the right policy. We will therefore discard the third choice as it would presumably lead to instability. The second choice does have the advantage that it distributes rewards more evenly, and the sum of the reward component coming from the option will add up to the payoff at expiry.
> **Lemma**. Let $P_t=\max(S_t-K, 0)$. If $S(0)\leq K$ then $\mathbb{E}[\sum P_t-P_{t-1}]=\mathbb{E}[C(T)]$ where $C(T)$ is the option value. If $S_0 > K$, the same holds up to an additive constant of $\max(S_0-K,0)$ 

Note: summation bounds are omitted due to formatting issues. First summation is from 1 to T, the second and third from 1 to T-1.

**Proof.** We have 
$\mathbb{E}[\sum P_t-P_{t-1}]=\sum\mathbb{E}[P_t] + \mathbb{E}[P_T]-\mathbb{E}[P_0]-\sum \mathbb{E}[P_t] = \mathbb{E}[P_T]=\mathbb{E}[C(T)]$ 
where we have used that $\mathbb{E}[P_0]=0$, assuming $S(0)\leq K$. If $S_0 > K$ then the expected sum of rewards is still the expected option value, up to an additive constant.

Therefore, we should be able to use both 1. and 2. to teach an agent how to hedge an option.
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

### 26/05/2024
After experimenting with the DQN parameters, found a combination that seemingly results in a stable learning process. (Note: also changed the reward function so the actual numbers cannot be compared to the ones above).


![Mean Reward](./option_hedging/Figures/mean_reward.png)

![Std Reward](./option_hedging/Figures/std_reward.png)

### 03/06/2024
Let DDQN train for a million environment steps with a stable learning trajectory. The resulting policy beats the mean reward of the Black-Scholes policy, but it's standard deviation is also greater. Nevertheless, the algorithm achieved what it was meant to so we consider the project to be a success.


![Mean Reward](./option_hedging/Figures/ddqn_03062024_mean.png)

![Std Reward](./option_hedging/Figures/ddqn_03062024_std.png)

Parameters:
```python
training_kwargs = {
        'trainer_kwargs': {
                'max_epoch': 50,
                'batch_size': 32,  # Small batch size (8-64) has been shown to improve DRL training performance
                'step_per_epoch': 20_000,
                'episode_per_test': 1_000,
                'update_per_step': 1.,  # Off-policy
                'repeat_per_collect': 3,  # On-policy
                'step_per_collect': 10_000,
                'verbose': True,
                'show_progress': True
        },
        'buffer_size': 500_000,
        'subproc': True
}

lr_kwargs = {
        'lr_scheduler_kwargs': {
                'end_factor': 0.1,
                'total_iters': 10
        },
        'lr': 0.0025
}

env_kwargs = {
        'epsilon': 0.01,
        'sigma': 0.15,
        'rho': 0.02,
        'action_bins': 20,
        'T': 1,
        'rebalance_frequency': 12
}

net_kwargs = {
        'linear_dims': tuple([256, 128, 64]),
        'residual_dims': None,
        'activation_fn': 'relu',
        'norm_layer': True
}
dqn_kwargs = {
        'policy_kwargs': {
                'discount_factor': 0.999,
                'estimation_step': 1,
                'target_update_freq': 5_000,
                'is_double': True,
                'clip_loss_grad': True
        },
        'epsilon_greedy': {'start': 1.,
                           'end': 0.1,
                           'max_steps': 500_000},
        'net_kwargs': net_kwargs,
        'env_kwargs': env_kwargs,
        **training_kwargs,
        **lr_kwargs
}
```

