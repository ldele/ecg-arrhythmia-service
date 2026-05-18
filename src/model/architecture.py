"""1D CNN for ECG beat classification.

Why 1D convolutions for ECG signals
-----------------------------------
ECG beats have a stereotyped local structure (P wave, QRS complex, T wave)
that occurs at roughly the same relative position within each beat window
but with variable amplitude, width, and noise. Three properties of 1D
convolutional networks match this structure:

1. **Translation equivariance.** A convolutional filter that learns to
   respond to a QRS-like shape responds to it wherever it appears in the
   window, so the model does not have to relearn the same morphology at
   each offset. R-peak alignment is imperfect (annotation jitter, our
   segmentation is fixed-width) and convolutions absorb that jitter
   cheaply. A fully-connected layer over raw samples cannot.

2. **Local receptive fields with hierarchical composition.** Early layers
   pick up local waveform features (slopes, narrow peaks, short
   oscillations); stacking conv + pool blocks expands the receptive field
   so deeper layers reason about whole sub-waves and inter-wave timing.
   This mirrors how clinicians actually read an ECG: from local
   morphology up to rhythm.

3. **Parameter efficiency.** Weight sharing across time keeps the model
   small (~30k parameters here), which matters when the rare classes
   (S, F) have only a few hundred training beats. A larger model would
   memorize the majority class and overfit the minority classes.

A 1D CNN is essentially the FIR-filter-bank intuition from classical
signal processing, but with the filter coefficients learned end-to-end
from labels instead of hand-designed. That equivalence is made explicit
in Oppenheim and Schafer (2009) for the linear case, and extended to the
learned non-linear case in the references below.

References
----------
- Kiranyaz, S., Avci, O., Abdeljaber, O., Ince, T., Gabbouj, M., &
  Inman, D. J. (2021). 1D Convolutional Neural Networks and Applications:
  A Survey. *Mechanical Systems and Signal Processing*, 151, 107398.
  Comprehensive review of 1D-CNNs for biomedical, vibration, and audio
  signals.
- Hannun, A. Y., Rajpurkar, P., Haghpanahi, M., et al. (2019).
  Cardiologist-level arrhythmia detection and classification in
  ambulatory electrocardiograms using a deep neural network.
  *Nature Medicine*, 25(1), 65-69. The standard reference for deep
  learning applied to ECG arrhythmia classification.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*,
  521(7553), 436-444. Sections on convolutional networks explain
  translation equivariance and hierarchical features.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.pool(self.relu(self.bn(self.conv(x))))  # type: ignore[no-any-return]


class ECGNet(nn.Module):
    """Small 1D CNN: 3 conv blocks, global average pool, linear head.

    Input shape: (batch, 1, 650)
    Output shape: (batch, n_classes) logits
    """

    def __init__(self, n_classes: int = 4) -> None:
        super().__init__()
        self.block1 = ConvBlock(in_ch=1, out_ch=16)
        self.block2 = ConvBlock(in_ch=16, out_ch=32)
        self.block3 = ConvBlock(in_ch=32, out_ch=64)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, 1, window_length)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.global_pool(x).squeeze(-1)  # (batch, 64)
            x = self.dropout(x)
            return self.fc(x)  # type: ignore[no-any-return]