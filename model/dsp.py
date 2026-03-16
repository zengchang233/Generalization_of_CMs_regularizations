"""Signal processing utilities for DCT-based feature extraction.

DCT code adapted from https://github.com/zh217/torch-dct (MIT License).
Original author: Xin Wang (wangxin@nii.ac.jp), Copyright 2020-2021.
"""

import numpy as np
import torch
import torch.nn as nn


def rfft_wrapper(x: torch.Tensor, onesided: bool = True,
                 inverse: bool = False) -> torch.Tensor:
    """FFT wrapper that returns real-valued tensor with (real, imag) last dim.

    For forward: returns tensor with shape (..., 2).
    For inverse: returns real-valued tensor.
    """
    if not inverse:
        data = torch.fft.rfft(x) if onesided else torch.fft.fft(x)
        return torch.stack([data.real, data.imag], dim=-1)
    else:
        real_imag = torch.chunk(x, 2, dim=1)
        x_complex = torch.complex(
            real_imag[0].squeeze(-1), real_imag[1].squeeze(-1)
        )
        if onesided:
            return torch.fft.irfft(x_complex)
        return torch.fft.ifft(x_complex).real


def dct1(x: torch.Tensor) -> torch.Tensor:
    """Discrete Cosine Transform, Type I."""
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    return rfft_wrapper(
        torch.cat([x, x.flip([1])[:, 1:-1]], dim=1)
    )[:, :, 0].view(*x_shape)


def idct1(X: torch.Tensor) -> torch.Tensor:
    """Inverse DCT-I: idct1(dct1(x)) == x."""
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    """Discrete Cosine Transform, Type II.

    Args:
        x: Input signal.
        norm: Normalization mode, None or 'ortho'.
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = rfft_wrapper(v, onesided=False)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    return 2 * V.view(*x_shape)


def idct(X: torch.Tensor, norm: str = None) -> torch.Tensor:
    """Inverse DCT-II (DCT-III): idct(dct(x)) == x.

    Args:
        X: Input signal.
        norm: Normalization mode, None or 'ortho'.
    """
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = rfft_wrapper(V, onesided=False, inverse=True)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


class LinearDCT(nn.Linear):
    """DCT implemented as a fixed linear transformation.

    Weight matrix is initialized from the DCT basis and frozen (not learnable).
    Input signal must have fixed length matching `in_features`.

    Args:
        in_features: Expected signal length.
        dct_type: One of 'dct', 'idct', 'dct1', 'idct1'.
        norm: Normalization mode, 'ortho' or None.
        bias: Whether to add bias (default False).
    """

    DCT_FUNCTIONS = {
        'dct': lambda self, x: dct(x, norm=self.norm),
        'idct': lambda self, x: idct(x, norm=self.norm),
        'dct1': lambda self, x: dct1(x),
        'idct1': lambda self, x: idct1(x),
    }

    def __init__(self, in_features: int, dct_type: str,
                 norm: str = None, bias: bool = False):
        if dct_type not in self.DCT_FUNCTIONS:
            raise ValueError(f"Unknown DCT type: {dct_type}. "
                             f"Must be one of {list(self.DCT_FUNCTIONS.keys())}")
        self.dct_type = dct_type
        self.N = in_features
        self.norm = norm
        super().__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        I = torch.eye(self.N)
        transform = self.DCT_FUNCTIONS[self.dct_type]
        self.weight.data = transform(self, I).data.t()
        self.weight.requires_grad = False
