import torch
from torch import nn
import numpy as np
from typing import Tuple, Any, Optional

activation_fns = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}


class PreprocessNet(nn.Module):
    def __init__(self,
                 state_shape,
                 linear_dims: Tuple[int, ...],
                 device: str,
                 activation_fn: str | None = 'relu',
                 norm_layer: bool = True,
                 residual_dims: Optional[Tuple[int, ...]] | None = None):
        super().__init__()
        self.output_dim = linear_dims[-1] if residual_dims is None else residual_dims[-1]
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
        if residual_dims is not None:
            model.append(ResidualBlock(
                in_features=linear_dims[-1],
                out_features=residual_dims[0],
                activation_fn=activation_fn,
                norm_layer=norm_layer,
                device=device
            ))
            for k in range(1, len(residual_dims)):
                model.append(ResidualBlock(
                    in_features=residual_dims[k - 1],
                    out_features=residual_dims[k],
                    activation_fn=activation_fn,
                    norm_layer=norm_layer,
                    device=device
                ))
        self.model = nn.Sequential(*model).to(device)

    def forward(self, obs: np.ndarray | torch.Tensor, state: Any = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        obs = torch.as_tensor(obs, device=self.device)
        return self.model(obs), state


class QNet(nn.Module):
    def __init__(self,
                 state_shape: Tuple[int, ...],
                 action_shape: Tuple[int, ...] | int,
                 linear_dims: Tuple[int, ...],
                 device: str,
                 activation_fn: str | nn.Module | None = 'relu',
                 norm_layer: bool = True,
                 residual_dims: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self.device = device
        self.model = PreprocessNet(state_shape=state_shape,
                                   linear_dims=linear_dims,
                                   device=device,
                                   activation_fn=activation_fn,
                                   norm_layer=norm_layer,
                                   residual_dims=residual_dims).to(device)
        self.action_shape = (action_shape,) if isinstance(action_shape, (int, np.int32, np.int64)) \
            else tuple(action_shape)
        self.output_layer = nn.Linear(linear_dims[-1] if residual_dims is None else residual_dims[-1],
                                      np.prod(action_shape)).to(device)

    def forward(self, obs: np.ndarray | torch.Tensor, state: Any = None, **kwargs) -> Tuple[torch.Tensor, Any]:
        obs = torch.as_tensor(obs, device=self.device)
        logits, _ = self.model(obs)
        return self.output_layer(logits).view((-1,)+self.action_shape), state


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation_fn: str | nn.Module | None,
                 norm_layer: bool,
                 device: str | torch.device):
        super().__init__()
        self.device = device
        if isinstance(activation_fn, str):
            activation_fn = activation_fns[activation_fn]
        l1 = [nn.Linear(in_features, in_features), activation_fn()]
        if norm_layer:
            l1.append(nn.BatchNorm1d(num_features=in_features))
        self.l1 = nn.Sequential(*l1).to(device)
        l2 = [nn.Linear(in_features, out_features), activation_fn()]
        if norm_layer:
            l2.append(nn.BatchNorm1d(num_features=out_features))
        self.l2 = nn.Sequential(*l2).to(device)

    def forward(self, input: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(input, device=self.device)
        x = self.l1(x) + x
        x = self.l2(x)
        return x
