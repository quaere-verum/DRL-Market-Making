# The problem
We imagine the following scenario: we are market makers for a European (to start with) options market. We are competing with other market makers. Therefore, whenever we quote the bid and ask prices, we need to ensure that they are both appealing to the market takers, and better than the prices of the competition, but we also need to have a large enough spread to be profitable and to cover the risk that we take on. Furthermore, we need to ensure that our portfolio remains delta neutral.

# Modelling assumptions
We are going to make some assumptions about the market dynamics in order to have a well-defined environment in which to test our model. We will go through multiple prototyping phases, gradually increasing the difficulty of the task that the DRL agent is faced with solving. In each case, we make the following assumptions:
- The price of the underlying asset is a martingale, but not necessarily geometric Brownian motion
- We can borrow and lend any amount of the underlying asset, or cash

# Research question
The fair price of a call option with strike price $K$ and expiry time $T$ at time $t_0$ with respect to the risk neutral measure $\mathbb{Q}$ is given by $\mathbb{E}(\max(S(T)-K, 0)\mid S(0)=x)$.
Under the assumptions of the Black-Scholes model, there is a way to perfectly hedge our position when selling an option by continuously re-adjusting our portfolio without market friction. In practice, it is not possible to continuously hedge, and there is friction. Therefore, the question becomes: is it possible for a deep reinforcement learning (DRL) agent to learn to be a market maker and beat the baseline Black-Scholes model, both by adapting to a more realistic environment, and by beating its competition on the bid/ask spread, while remaining profitable?

# Model implementation
We propose using self-play deep reinforcement learning (DRL), to train an agent to perform the task of market making, because the problem at hand is nothing but a problem of optimal control. In [this](https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf) paper, the optimal bid/ask spread is derived under certain assumptions, from a Hamilton-Jacobi-Bellman equation, which provides a PDE to be solved. We will use the obtained solution to set the bid/ask spread of the Black-Scholes benchmark. By using DRL, a solution to the Bellman equation (in discrete time) is found, but for a possibly different objective function and with less restrictive assumptions. Additionally, in a situation where there are multiple market makers, they evidently cannot all use the same strategy to determine their bid/ask prices, so we want to train an agent which anticipates adversarial behaviour from other market makers.

The second task of the agent, besides setting the bid/ask spread, will be to maintain a delta neutral portfolio, while processing incoming orders which follow a Poisson arrival process.

The idea is to have an "old" version of the agent "play" against itself. That is to say, we imagine there are two market makers, namely the "old agent" and the "new agent". We view each of these as policies $\pi:\mathcal{S}\to\mathcal{A}$, assigning an action $\pi(s)=a\in\mathcal{A}$ to an observed state $s\in\mathcal{S}$. During training, we keep two models stored in memory: $\pi_\text{old}$ and $\pi_\text{new}$. We let them play against each other to fill the replay buffer, and then do the backpropagation to update the network weights, resulting in $\pi_\text{new}\mapsto\pi_\text{old}$ and $\pi_\text{new}-\nabla L\mapsto \pi_\text{new}$.

We will implement this model in phases, building up the complexity in each phase. We have the following phases in mind:

1. Delta neutrality only - single agent, single order

    a. Constant volatility
    
    b. Autocorrelated volatility
    
2. Delta neutrality only - single agent, multiple order arrivals over time
3. Delta neutrality + bid/ask spread, single agent vs. Black-Scholes baseline
4. Adversarial reinforcement learning - multiple agents competing for the market

Each phase will have to pass a benchmark test before we proceed to the next phase.

## Phase 1: Delta Neutrality
To prototype a model, we begin with the simplest case where the agent's only task is to maintain a delta neutral portfolio. All prices will be relative to $S(0)$, the price of the underlying at the time the option is written, so we set $S(0)=1$. We will assume that we are hedging a short position in a call option, for it stands to reason that if the agent can learn this task, then put-call parity implies that it can also learn how to hedge a short position in a put option.
### Designing the environment
As a first step we design the environment in which our agents get to act. More specifically, this means we need to:
1. Determine the actions our agent is allowed to take
2. Determine the state that is returned after each step

In phase 1, the action space is simply $\mathcal{A}=[0,1+\varepsilon]$, where $\varepsilon > 0$ is a hyperparameter of the model. The reason for this is that we assume that Black-Scholes offers a good approximation to reality, and in this case, the hedging amount is $\Phi(d_+)$ where $\Phi:\mathbb{R}\to[0,1]$ is the cumulative distribution function of the standard normal. Our action $a$ will be the replacement for $\Phi(d_+)$, and so we expect that the optimal action $a^*\in[0,1]$. To give additional room for exploration, we add the constant epsilon.

The state space is then given by the parameters which are used in the Black-Scholes model: the price $S(t)$, the remaining time until expiry $T-t$, the amount of stock currently held, the strike price $K$, and a forecast of the volatility $\widetilde{\sigma}$ (if not assumed to be constant). Additionally, we will include the Black-Scholes hedge as part of the state, to give the agent an anchor. Our deep neural network will be predicting the residual of the optimal hedging amount, compared to the Black-Scholes solution.
