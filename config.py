from option_hedging.OnPolicyTests.a2c import a2c_trial
from option_hedging.OnPolicyTests.ppo import ppo_trial
from option_hedging.OffPolicyTests.dqn import dqn_trial
from option_hedging.OffPolicyTests.ddpg import ddpg_trial
from option_hedging.OffPolicyTests.sac import sac_trial
from option_hedging.OffPolicyTests.td3 import td3_trial

trainer_kwargs = {
        'max_epoch': 100,
        'batch_size': 512,
        'step_per_epoch': 10000,
        'repeat_per_collect': 5,
        'episode_per_test': 1000,
        'update_per_step': 2.,
        'step_per_collect': 2000,
        'verbose': True,
        'show_progress': True
}

env_kwargs = {
        'epsilon': 0.,
        'sigma': 0.3,
        'rho': 0.2,
        'action_bins': 20,
        'T': 1,
        'rebalance_frequency': 12
}

ppo_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'buffer_size': 5000,
        'lr': 0.00025,
        'subproc': False,
        'net_arch': tuple(64 for k in range(6)),
        'policy_kwargs': {
                'eps_clip': 0.2,
                'dual_clip': None,
                'value_clip': None,
                'vf_coef': 0.7,
                'ent_coef': 0.005,
                'max_grad_norm': 1,
                'gae_lambda': 0.999,
                'discount_factor': 0.99
        },
        'env_kwargs': env_kwargs
}

sac_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'policy_kwargs': {
                'tau': 0.01
        },
        'buffer_size': 2000,
        'lr': 0.00025,
        'subproc': False,
        'net_arch': (64, 32, 16, 4),
        'env_kwargs': env_kwargs
}

ddpg_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'policy_kwargs': {
                'exploration_noise': 'default',
                'tau': 0.01
        },
        'buffer_size': 5000,
        'lr': 0.00025,
        'subproc': False,
        'net_arch': tuple(64 for _ in range(6)),
        'env_kwargs': env_kwargs
}

td3_kwargs = ddpg_kwargs.copy()

a2c_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'policy_kwargs':{
                'vf_coef': 0.5,
                'ent_coef': 0.005,
                'max_grad_norm': 1.,
                'gae_lambda': 0.999,
                'discount_factor': 0.99
        },
        'buffer_size': 2000,
        'lr': 0.001,
        'subproc': False,
        'net_arch': tuple(64 for _ in range(6)),
        'env_kwargs': env_kwargs
}

dqn_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'policy_kwargs': {
                'discount_factor': 0.99,
                'estimation_step': 1,
                'target_update_freq': 10000,
                'is_double': True,
                'clip_loss_grad': True
        },

        'buffer_size': 10000,
        'lr': 0.00025,
        'epsilon_greedy': {'start': 1.,
                           'end': 0.1},
        'subproc': False,
        'net_arch': tuple(64 for _ in range(6)),
        'env_kwargs': env_kwargs
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