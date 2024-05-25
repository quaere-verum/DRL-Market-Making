from option_hedging.OnPolicyTests.a2c import a2c_trial
from option_hedging.OnPolicyTests.ppo import ppo_trial
from option_hedging.OffPolicyTests.dqn import dqn_trial
from option_hedging.OffPolicyTests.ddpg import ddpg_trial
from option_hedging.OffPolicyTests.sac import sac_trial
from option_hedging.OffPolicyTests.td3 import td3_trial

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

env_kwargs = {
        'epsilon': 0.1,
        'sigma': 0.05,
        'rho': 0.2,
        'action_bins': 32,
        'T': 1,
        'rebalance_frequency': 10
}

ppo_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'buffer_size': 6000,
        'lr': 0.001,
        'subproc': False,
        'net_arch': tuple(32 for k in range(10)),
        'dist_std': 0.1,
        **env_kwargs
}

sac_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'tau': 0.05,
        'buffer_size': 2000,
        'lr': 0.001,
        'subproc': False,
        'net_arch': (64, 32, 16, 4),
        'dist_std': 0.2,
        **env_kwargs
}

ddpg_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'rho': 0.2,
        'tau': 0.05,
        'buffer_size': 2000,
        'lr': 0.001,
        'subproc': False,
        'net_arch': (64, 32, 16, 4),
        **env_kwargs
}

td3_kwargs = ddpg_kwargs.copy()

a2c_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'rho': 0.2,
        'vf_coef': 0.5,
        'ent_coef': 0.02,
        'buffer_size': 2000,
        'lr': 0.001,
        'subproc': False,
        'net_arch': (64, 32, 16, 4),
        **env_kwargs
}

dqn_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'discount_factor': 0.99,
        'estimation_step': 1,
        'target_update_freq': 10000,
        'is_double': True,
        'clip_loss_grad': True,
        'buffer_size': 2000,
        'lr': 0.00025,
        'epsilon_greedy': {'start': 1.,
                           'end': 0.1},
        'subproc': False,
        'net_arch': tuple(32 for _ in range(10)),
        **env_kwargs
}

options = {
        'ppo': {'kwargs': ppo_kwargs,
                'trainer': ppo_trial},
        'a2c': {'kwargs': a2c_kwargs,
                'trainer': a2c_trial},
        'sac': {'kwargs': sac_kwargs,
                'trainer': sac_trial},
        'ddpg': {'kwargs': ddpg_kwargs,
                 'trainer': ddpg_trial},
        'td3': {'kwargs': td3_kwargs,
                'trainer': td3_trial},
        'dqn': {'kwargs': dqn_kwargs,
                'trainer': dqn_trial}
}