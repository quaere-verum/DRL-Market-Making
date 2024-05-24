from option_hedging.OnPolicyTests.a2c import a2c_trial
from option_hedging.OnPolicyTests.ppo import ppo_trial
from option_hedging.OffPolicyTests.dqn import dqn_trial
from option_hedging.OffPolicyTests.ddpg import ddpg_trial
from option_hedging.OffPolicyTests.sac import sac_trial

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

sac_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'epsilon': 0.1,
        'sigma': 0.05,
        'rho': 0.2,
        'tau': 0.05,
        'action_bins': 25,
        'duration_bounds': (6, 12),
        'buffer_size': 2000,
        'lr': 0.001,
        'subproc': False,
        'net_arch': (64, 32, 16, 4),
        'dist_std': 0.2
}

ddpg_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'epsilon': 0.1,
        'sigma': 0.05,
        'rho': 0.2,
        'tau': 0.05,
        'action_bins': 25,
        'duration_bounds': (6, 12),
        'buffer_size': 2000,
        'lr': 0.001,
        'subproc': False,
        'net_arch': (64, 32, 16, 4),
}

a2c_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'epsilon': 0.1,
        'sigma': 0.05,
        'rho': 0.2,
        'vf_coef': 0.5,
        'ent_coef': 0.02,
        'action_bins': 25,
        'duration_bounds': (6, 12),
        'buffer_size': 2000,
        'lr': 0.001,
        'subproc': False,
        'net_arch': (64, 32, 16, 4),
}

dqn_kwargs = {
        'trainer_kwargs': trainer_kwargs,
        'epsilon': 0.1,
        'sigma': 0.05,
        'rho': 0.2,
        'discount_factor': 1.,
        'estimation_step': 1,
        'target_update_freq': 2,
        'is_double': True,
        'clip_loss_grad': True,
        'action_bins': 25,
        'duration_bounds': (6, 12),
        'buffer_size': 2000,
        'lr': 0.001,
        'subproc': False,
        'net_arch': tuple(32 for k in range(10))
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
        'dqn': {'kwargs': dqn_kwargs,
                'trainer': dqn_trial}
}