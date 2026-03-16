"""Front-end feature extraction for audio anti-spoofing.

Provides LFCC (Linear Frequency Cepstral Coefficients) extraction,
based on asvspoof.org baseline code.

Original author: Xin Wang (wangxin@nii.ac.jp), Copyright 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import model.dsp as dsp


def trimf(x: torch.Tensor, params: list) -> torch.Tensor:
    """Triangular membership function (similar to Matlab's trimf).

    Args:
        x: Input tensor.
        params: [a, b, c] where a <= b <= c defines the triangle.
    """
    if len(params) != 3:
        raise ValueError("trimf requires params to be a list of 3 elements")
    a, b, c = params
    if a > b or b > c:
        raise ValueError(f"trimf(x, [a, b, c]) requires a<=b<=c, got [{a}, {b}, {c}]")

    y = torch.zeros_like(x, dtype=torch.float32)
    if a < b:
        mask = torch.logical_and(a < x, x < b)
        y[mask] = (x[mask] - a) / (b - a)
    if b < c:
        mask = torch.logical_and(b < x, x < c)
        y[mask] = (c - x[mask]) / (c - b)
    y[x == b] = 1
    return y


def delta(x: torch.Tensor) -> torch.Tensor:
    """Compute delta (first derivative) along time dimension.

    Args:
        x: Input tensor of shape (batch, length, dim).

    Returns:
        Delta features of the same shape.
    """
    length = x.shape[1]
    x_padded = F.pad(x.unsqueeze(1), (0, 0, 1, 1), 'replicate').squeeze(1)
    return -1 * x_padded[:, :length] + x_padded[:, 2:]


class LFCC(nn.Module):
    """Linear Frequency Cepstral Coefficients (LFCC) extractor.

    Based on asvspoof.org baseline Matlab code. Extracts LFCC features
    with optional energy, pre-emphasis, and delta coefficients.

    Args:
        fl: Frame length in waveform samples.
        fs: Frame shift in waveform samples.
        fn: FFT points.
        sr: Sampling rate (Hz).
        filter_num: Number of filters in the linear filter bank.
        with_energy: Whether to replace 1st coefficient with energy.
        with_emphasis: Whether to apply pre-emphasis.
        with_delta: Whether to append delta and delta-delta.
        num_coef: Number of cepstral coefficients (None = filter_num).
        min_freq: Min frequency as fraction of Nyquist (0.0-1.0).
        max_freq: Max frequency as fraction of Nyquist (0.0-1.0).
    """

    def __init__(self, fl: int, fs: int, fn: int, sr: int, filter_num: int,
                 with_energy: bool = False, with_emphasis: bool = True,
                 with_delta: bool = True, num_coef: int | None = None,
                 min_freq: float = 0.0, max_freq: float = 1.0):
        super().__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.filter_num = filter_num
        self.num_coef = num_coef if num_coef is not None else filter_num

        if not (0 <= min_freq < max_freq <= 1):
            raise ValueError(
                f"Invalid frequency range: min_freq={min_freq}, max_freq={max_freq}"
            )
        self.min_freq_bin = int(min_freq * (fn // 2 + 1))
        self.max_freq_bin = int(max_freq * (fn // 2 + 1))
        num_fft_bins = self.max_freq_bin - self.min_freq_bin

        # Build triangular filter bank
        f = (sr / 2) * torch.linspace(min_freq, max_freq, num_fft_bins)
        filter_bands = torch.linspace(min(f), max(f), filter_num + 2)

        filter_bank = torch.zeros([num_fft_bins, filter_num])
        for idx in range(filter_num):
            filter_bank[:, idx] = trimf(
                f, [filter_bands[idx], filter_bands[idx + 1], filter_bands[idx + 2]]
            )
        self.lfcc_fb = nn.Parameter(filter_bank, requires_grad=False)

        # DCT as a fixed linear transformation
        self.l_dct = dsp.LinearDCT(filter_num, 'dct', norm='ortho')

        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta

        # Buffer for window coefficients (lazily initialized)
        self.window_buf = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract LFCC features from waveform.

        Args:
            x: Waveform tensor of shape (batch, length).

        Returns:
            LFCC features of shape (batch, frame_num, dim).
        """
        # Pre-emphasis
        if self.with_emphasis:
            x = x.clone()
            x[:, 1:] = x[:, 1:] - 0.97 * x[:, :-1]

        if self.window_buf is None:
            self.window_buf = torch.hamming_window(self.fl).to(x.device)

        # STFT
        x_stft = torch.stft(
            x, self.fn, self.fs, self.fl,
            window=self.window_buf, onesided=True,
            pad_mode="constant", return_complex=False,
        )

        # Power spectrum: (batch, frame_num, fft_bins)
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()

        if self.min_freq_bin > 0 or self.max_freq_bin < (self.fn // 2 + 1):
            sp_amp = sp_amp[:, :, self.min_freq_bin:self.max_freq_bin]

        # Apply filter bank + log
        fb_feature = torch.log10(
            torch.matmul(sp_amp, self.lfcc_fb) + torch.finfo(torch.float32).eps
        )

        # DCT
        lfcc = self.l_dct(fb_feature)

        # Truncate coefficients if needed
        if self.num_coef != self.filter_num:
            lfcc = lfcc[:, :, :self.num_coef]

        # Replace first coefficient with energy
        if self.with_energy:
            power_spec = sp_amp / self.fn
            energy = torch.log10(
                power_spec.sum(axis=2) + torch.finfo(torch.float32).eps
            )
            lfcc[:, :, 0] = energy

        # Append delta and delta-delta
        if self.with_delta:
            lfcc_delta = delta(lfcc)
            lfcc_delta_delta = delta(lfcc_delta)
            lfcc = torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), dim=2)

        return lfcc
