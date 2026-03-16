"""Meta-learning layers with fast weight support for MAML-style training.

These layers extend standard PyTorch layers with a `.fast` attribute on weights,
enabling inner-loop gradient updates in MAML without modifying the original parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaLinear(nn.Linear):
    """Linear layer with fast weight support for MAML.

    During meta-training inner loop, set `weight.fast` and `bias.fast`
    to use updated parameters without modifying the original weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.fast is not None and self.bias.fast is not None:
            return F.linear(x, self.weight.fast, self.bias.fast)
        return super().forward(x)


class MetaConv2d(nn.Conv2d):
    """Conv2d layer with fast weight support for MAML."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if self.bias is not None:
            self.bias.fast = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.fast if self.weight.fast is not None else self.weight
        bias = None
        if self.bias is not None:
            bias = self.bias.fast if self.bias.fast is not None else self.bias
        if weight is not self.weight or (bias is not None and bias is not self.bias):
            return F.conv2d(x, weight, bias,
                            stride=self.stride, padding=self.padding)
        return super().forward(x)


class MetaBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d with fast weight support for MAML."""

    def __init__(self, num_features: int, affine: bool = False,
                 momentum: float = 0.1, track_running_stats: bool = True):
        super().__init__(num_features, affine=affine, momentum=momentum,
                         track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.fast if self.weight.fast is not None else self.weight
        bias = self.bias.fast if self.bias.fast is not None else self.bias
        if self.track_running_stats:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                weight, bias, training=self.training, momentum=self.momentum,
            )
        return F.batch_norm(
            x,
            torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
            torch.ones(x.size(1), dtype=x.dtype, device=x.device),
            weight, bias, training=True, momentum=1,
        )


class MetaBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d with fast weight support for MAML."""

    def __init__(self, num_features: int, momentum: float = 0.1,
                 track_running_stats: bool = True):
        super().__init__(num_features, momentum=momentum,
                         track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.fast if self.weight.fast is not None else self.weight
        bias = self.bias.fast if self.bias.fast is not None else self.bias
        if self.track_running_stats:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                weight, bias, training=self.training, momentum=self.momentum,
            )
        return F.batch_norm(
            x,
            torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
            torch.ones(x.size(1), dtype=x.dtype, device=x.device),
            weight, bias, training=True, momentum=1,
        )
