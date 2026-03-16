"""LCNN (Light CNN) model with meta-learning and domain alignment.

Architecture:
- LFCC front-end feature extraction
- LCNN backbone with MaxFeatureMap activations
- Domain alignment branch with Gradient Reversal Layer (GRL)
- Meta-learning support via fast weight layers (MAML)
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch_adapt.layers import GradientReversal

from model import feature as front_end
from model.meta_layers import MetaConv2d, MetaLinear, MetaBatchNorm1d


class MaxFeatureMap2D(nn.Module):
    """Max Feature Map activation along channel dimension.

    Splits channels in half and takes element-wise maximum,
    reducing channel count by 2x.

    Input:  (batch, channel, ...)
    Output: (batch, channel//2, ...)
    """

    def __init__(self, max_dim: int = 1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = list(inputs.size())
        if self.max_dim >= len(shape):
            raise ValueError(
                f"MaxFeatureMap: max_dim={self.max_dim} >= input dims={len(shape)}"
            )
        if shape[self.max_dim] % 2 != 0:
            raise ValueError(
                f"MaxFeatureMap: dim {self.max_dim} has odd size {shape[self.max_dim]}"
            )
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)
        m, _ = inputs.view(*shape).max(self.max_dim)
        return m


class DomainAlignment(nn.Module):
    """Domain alignment module with Gradient Reversal Layer.

    Uses GRL to learn domain-invariant features by adversarial training
    against a domain classifier.

    Args:
        in_dim: Input embedding dimension.
        out_dim: Number of domain classes.
        hidden_dim: Hidden layer dimension.
        grl_weight: Weight for gradient reversal.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int,
                 grl_weight: float):
        super().__init__()
        self.grl = GradientReversal(weight=grl_weight)
        self.classifier = nn.Sequential(
            MetaLinear(in_dim, hidden_dim),
            nn.ReLU(),
            MetaBatchNorm1d(hidden_dim),
            MetaLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            MetaBatchNorm1d(hidden_dim),
            MetaLinear(hidden_dim, out_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.grl(embedding))


class LCNN(nn.Module):
    """Light CNN for audio anti-spoofing with meta-learning and domain alignment.

    Args:
        in_dim: Input dimension (typically 1 for raw waveform).
        out_dim: Output dimension (1 for binary classification).
        domain_align_weight: Weight for gradient reversal layer.
        domain_num: Number of domain classes (genres).
    """

    # Feature extraction configuration
    SAMPLE_RATE = 16000
    FRAME_HOP = 160
    FRAME_LEN = 320
    FFT_N = 512
    LFCC_DIM = 20

    def __init__(self, in_dim: int = 1, out_dim: int = 1,
                 domain_align_weight: float = 1.0,
                 domain_num: int = 10):
        super().__init__()

        self.v_emd_dim = out_dim
        self.alignment_dim = domain_num

        # Truncation length for feature frames
        self.trunc_len = 10 * 16 * 750 // self.FRAME_HOP

        # LFCC dimension with delta and delta-delta
        lfcc_dim = self.LFCC_DIM * 3

        # Front-end: LFCC feature extractor
        self.frontend = front_end.LFCC(
            self.FRAME_LEN, self.FRAME_HOP, self.FFT_N,
            self.SAMPLE_RATE, self.LFCC_DIM, with_energy=True,
        )

        # Backbone: LCNN with MaxFeatureMap activations
        self.backbone = nn.Sequential(
            MetaConv2d(1, 64, (5, 5), 1, padding=(2, 2)),
            MaxFeatureMap2D(),
            nn.MaxPool2d((2, 2), (2, 2)),

            MetaConv2d(32, 64, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            MetaConv2d(32, 96, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),

            nn.MaxPool2d((2, 2), (2, 2)),
            nn.BatchNorm2d(48, affine=False),

            MetaConv2d(48, 96, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(48, affine=False),
            MetaConv2d(48, 128, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),

            nn.MaxPool2d((2, 2), (2, 2)),

            MetaConv2d(64, 128, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(64, affine=False),
            MetaConv2d(64, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),

            MetaConv2d(32, 64, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            MetaConv2d(32, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            nn.MaxPool2d((2, 2), (2, 2)),
        )

        # Embedding layer: flatten → dense → MFM
        flatten_dim = (self.trunc_len // 16) * (lfcc_dim // 16) * 32
        self.embedding = nn.Sequential(
            nn.Dropout(0.7),
            MetaLinear(flatten_dim, 160),
            MaxFeatureMap2D(),
        )

        # Output head: binary classification score
        self.output_head = MetaLinear(80, self.v_emd_dim)

        # Domain alignment head: genre classification
        self.domain_head = DomainAlignment(
            80, domain_num, 128, domain_align_weight,
        )

    def _extract_features(self, wav: torch.Tensor,
                          datalength: List[int]) -> torch.Tensor:
        """Extract LFCC features with per-sample truncation/padding.

        Args:
            wav: Raw waveform, shape (batch, length, 1).
            datalength: List of actual audio lengths per sample.

        Returns:
            Features of shape (batch, trunc_len, lfcc_dim).
        """
        with torch.no_grad():
            x = self.frontend(wav.squeeze(-1))       # (batch, frame_num, feat_dim)
            x = x.permute(0, 2, 1)                   # (batch, feat_dim, frame_num)

            buffer = torch.zeros(
                [x.shape[0], x.shape[1], self.trunc_len],
                dtype=x.dtype, device=x.device,
            )

            for i in range(x.shape[0]):
                true_frames = datalength[i] // self.FRAME_HOP
                if true_frames > self.trunc_len:
                    pos = int(torch.rand([1]).item() * (true_frames - self.trunc_len))
                    buffer[i] = x[i, :, pos:self.trunc_len + pos]
                else:
                    reps = int(np.ceil(self.trunc_len / true_frames))
                    repeated = x[i, :, :true_frames].repeat(1, reps)
                    buffer[i] = repeated[:, :self.trunc_len]

            return buffer.permute(0, 2, 1)           # (batch, trunc_len, feat_dim)

    def _compute_embedding(self, x: torch.Tensor,
                           datalength: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute anti-spoofing embedding and domain scores.

        Returns:
            cm_scores: (batch, v_emd_dim) anti-spoofing raw scores.
            domain_scores: (batch, domain_num) domain classification logits.
        """
        features = self._extract_features(x, datalength)
        hidden = self.backbone(features.unsqueeze(1))
        embedding = self.embedding(torch.flatten(hidden, 1))
        cm_scores = self.output_head(embedding)
        domain_scores = self.domain_head(embedding)
        return cm_scores, domain_scores

    def forward(self, audio: torch.Tensor,
                datalength: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            audio: Raw waveform, shape (batch, length, 1).
            datalength: List of actual audio lengths.

        Returns:
            scores: (batch,) sigmoid probabilities for real/spoof.
            domain_scores: (batch, domain_num) domain logits.
        """
        cm_scores, domain_scores = self._compute_embedding(audio, datalength)
        scores = torch.sigmoid(cm_scores).squeeze(1)
        return scores, domain_scores

    def inference(self, audio: torch.Tensor, datalength: List[int],
                  filenames: list, target: torch.Tensor) -> None:
        """Inference mode (returns raw scores without sigmoid)."""
        cm_scores, _ = self._compute_embedding(audio, datalength)
        scores = cm_scores.squeeze(1)
        print(f"Output, {filenames[0]}, {target[0]}, {scores.mean():.6f}")


if __name__ == '__main__':
    model = LCNN(1, 1, 1.0, 10)
    waveform = torch.randn(4, 45000, 1, dtype=torch.float32)
    scores, domain_scores = model(waveform, [32000, 33000, 34000, 45000])
    print(f"Scores shape: {scores.shape}")
    print(f"Domain scores shape: {domain_scores.shape}")
