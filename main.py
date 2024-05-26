from option_hedging.gym_envs import make_env
from config import options


def main() -> None:
    model = input(f'Select a model: {list(options.keys())}\n')
    assert model.lower() in options.keys()
    selection = options[model.lower()]
    kwargs = selection['kwargs']
    benchmark = input(f'Benchmark against Black-Scholes? y/n\n')
    assert benchmark.lower() in ('y', 'n')
    if benchmark.lower() == 'y':
        from option_hedging.benchmarks import black_scholes_benchmark
        env = make_env(seed=123, **kwargs['env_kwargs'])()
        mean, std = black_scholes_benchmark(env, n_trials=1000)
        print(f'Black-Scholes benchmark: {mean} +/- {std}\n')
    random_benchmark = input(f'Benchmark against random agent? y/n\n')
    assert random_benchmark.lower() in ('y', 'n')
    if random_benchmark.lower() == 'y':
        from option_hedging.benchmarks import random_agent_benchmark
        env = make_env(seed=123, **kwargs['env_kwargs'])()
        mean, std = random_agent_benchmark(env, n_trials=1000)
        print(f'Random agent benchmark: {mean} +/- {std}\n')
    trainer = selection['trainer'](**kwargs)
    trainer.run()


if __name__ == '__main__':
    main()
