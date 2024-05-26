import torch
from torch import nn
import numpy as np
from typing import Tuple, Any

activation_fns = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}


class PreprocessNet(nn.Module):
    def __init__(self,
                 state_shape,
                 linear_dims: Tuple[int],
                 device: str,
                 activation_fn: str | None = 'relu',
                 norm_layer: bool = True):
        super().__init__()
        self.output_dim = linear_dims[-1]
        self.device = device
        activation_fn = activation_fns[activation_fn]
        model = []
        if len(state_shape) > 1:
            model.append(nn.Flatten())
        model.extend([nn.Linear(np.prod(state_shape), linear_dims[0]), activation_fn()])
        if norm_layer:
            model.append(nn.BatchNorm1d(num_features=linear_dims[0]))
        for k in range(1, len(linear_dims)):
            model.extend([nn.Linear(linear_dims[k-1], linear_dims[k]), activation_fn()])
            if norm_layer and k != len(linear_dims) - 1:
                model.append(nn.BatchNorm1d(num_features=linear_dims[k]))
        self.model = nn.Sequential(*model).to(device)

    def forward(self, obs: np.ndarray | torch.Tensor, state: Any = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        obs = torch.as_tensor(obs, device=self.device)
        return self.model(obs), state

class QNet(nn.Module):
    def __init__(self,
                 state_shape: Tuple[int],
                 action_shape: Tuple[int],
                 linear_dims: Tuple[int],
                 device: str,
                 activation_fn: str | None = 'relu',
                 norm_layer: bool = True):
        super().__init__()
        self.output_dim = linear_dims[-1]
        self.device = device
        activation_fn = activation_fns[activation_fn]
        model = []
        if len(state_shape) > 1:
            model.append(nn.Flatten())
        model.extend([nn.Linear(np.prod(state_shape), linear_dims[0]), activation_fn()])
        if norm_layer:
            model.append(nn.BatchNorm1d(num_features=linear_dims[0]))
        for k in range(1, len(linear_dims)):
            model.extend([nn.Linear(linear_dims[k - 1], linear_dims[k]), activation_fn()])
            if norm_layer:
                model.append(nn.BatchNorm1d(num_features=linear_dims[k]))
        model.append(nn.Linear(linear_dims[-1], np.prod(action_shape)))
        self.model = nn.Sequential(*model).to(device)

    def forward(self, obs: np.ndarray | torch.Tensor, state: Any = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        obs = torch.as_tensor(obs, device=self.device)
        return self.model(obs), state
