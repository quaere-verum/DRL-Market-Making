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
        'train_kwargs': trainer_kwargs,
        'epsilon': 0.1,
        'sigma': 0.05,
        'rho': 0.2,
        'action_bins': 25,
        'duration_bounds': (6, 12),
        'buffer_size': 2000,
        'lr': 0.001,
        'subproc': False,
        'net_arch': (64, 32, 16, 4),
        'dist_std': 0.2
}

sac_kwargs = {
        'train_kwargs': trainer_kwargs,
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
        'train_kwargs': trainer_kwargs,
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
        'train_kwargs': trainer_kwargs,
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
        'train_kwargs': trainer_kwargs,
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
        'net_arch': (64, 32, 16, 4),
        'dist_std': 0.2
}