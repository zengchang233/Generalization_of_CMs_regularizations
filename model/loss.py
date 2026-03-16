"""Loss functions for anti-spoofing training.

Includes:
- FocalLoss: Class-imbalance-aware cross entropy for domain classification.
- AutomaticWeightedLoss: Uncertainty-based multi-task loss weighting.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in classification.

    Reference: https://arxiv.org/abs/1708.02002

    Args:
        alpha: Per-class weights. None for uniform weighting.
        gamma: Focusing parameter. Higher values down-weight easy examples.
        reduction: 'mean', 'sum', or 'none'.
        ignore_index: Class label to ignore in loss computation.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"Reduction must be one of: 'mean', 'sum', 'none'. Got '{reduction}'")

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index
        )

    def __repr__(self) -> str:
        args = ', '.join(
            f'{k}={getattr(self, k)!r}'
            for k in ['alpha', 'gamma', 'ignore_index', 'reduction']
        )
        return f'{type(self).__name__}({args})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            x: Predictions, shape (N, C) or (N, C, d1, ..., dK).
            y: Targets, shape (N,) or (N, d1, ..., dK).
        """
        if x.ndim > 2:
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0., device=x.device)
        x = x[unignored_mask]

        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        log_pt = log_p[torch.arange(len(x)), y]
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        loss = focal_term * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class AutomaticWeightedLoss(nn.Module):
    """Automatically weighted multi-task loss using learned uncertainty.

    Learns task-specific weights via homoscedastic uncertainty, as described in:
    "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018).

    Args:
        num: Number of loss terms to combine.

    Example:
        awl = AutomaticWeightedLoss(3)
        total = awl(loss1, loss2, loss3)
    """

    def __init__(self, num: int = 2):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num))

    def forward(self, *losses: Tensor) -> Tensor:
        total = torch.tensor(0., device=self.params.device)
        for i, loss in enumerate(losses):
            precision = 0.5 / (self.params[i] ** 2)
            regularization = torch.log(1 + self.params[i] ** 2)
            total = total + precision * loss + regularization
        return total
