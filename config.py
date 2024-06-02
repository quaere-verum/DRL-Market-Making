from option_hedging.OnPolicyTests.a2c import a2c_trial
from option_hedging.OnPolicyTests.ppo import ppo_trial
from option_hedging.OffPolicyTests.dqn import dqn_trial
from option_hedging.OffPolicyTests.ddpg import ddpg_trial
from option_hedging.OffPolicyTests.sac import sac_trial
from option_hedging.OffPolicyTests.td3 import td3_trial

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

ppo_kwargs = {
        'net_kwargs': net_kwargs,
        'policy_kwargs': {
                'eps_clip': 0.2,
                'dual_clip': None,
                'value_clip': None,
                'vf_coef': 0.5,
                'ent_coef': 0.005,
                'max_grad_norm': 1,
                'gae_lambda': 0.999,
                'discount_factor': 0.99
        },
        'env_kwargs': env_kwargs,
        **training_kwargs,
        **lr_kwargs
}

sac_kwargs = {
        'policy_kwargs': {
                'tau': 0.02
        },
        'net_kwargs': net_kwargs,
        'env_kwargs': env_kwargs,
        **training_kwargs,
        **lr_kwargs
}

ddpg_kwargs = {
        'policy_kwargs': {
                'exploration_noise': 'default',
                'tau': 0.01
        },
        'net_kwargs': net_kwargs,
        'env_kwargs': env_kwargs,
        **training_kwargs,
        **lr_kwargs
}

td3_kwargs = ddpg_kwargs.copy()

a2c_kwargs = {
        'policy_kwargs':{
                'vf_coef': 0.5,
                'ent_coef': 0.005,
                'max_grad_norm': 1.,
                'gae_lambda': 0.999,
                'discount_factor': 0.99
        },
        'net_kwargs': net_kwargs,
        'env_kwargs': env_kwargs,
        **training_kwargs,
        **lr_kwargs
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
