"""Small PyTorch EEGNet-8,2 implementation with explicit inference state.

Architecture: Lawhern et al. (2018), https://doi.org/10.1088/1741-2552/aace8c.
Layer settings checked against the authors' EEGModels.py (EEGNet, not EEGNet_old)
and Braindecode EEGNet. This is a local implementation, not a claim of numerical
equivalence to TensorFlow/Braindecode. It uses PyTorch initializers and Adam.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_all(seed: int, threads: int = 2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class EEGNet(nn.Module):
    """Input (batch, 19, 256), output two unnormalized class logits.

    Asymmetric SAME padding for even kernels preserves sample count. BatchNorm
    eps=0.001 and PyTorch momentum=0.01 correspond to the usual Keras settings.
    Max-norm constraints are applied only after optimizer steps, never forward.
    """
    def __init__(self, channels=19, samples=256, f1=8, depth=2, f2=16,
                 temporal_kernel=64, separable_kernel=16, dropout=0.5):
        super().__init__()
        if channels < 1 or samples < 32 or samples % 32 or temporal_kernel < 1:
            raise ValueError("Invalid EEGNet input geometry")
        self.channels, self.samples = channels, samples
        self.temporal_kernel, self.separable_kernel = temporal_kernel, separable_kernel
        self.temporal = nn.Conv2d(1, f1, (1, temporal_kernel), bias=False)
        self.bn1 = nn.BatchNorm2d(f1, eps=0.001, momentum=0.01)
        self.spatial = nn.Conv2d(f1, f1 * depth, (channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * depth, eps=0.001, momentum=0.01)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.depthwise = nn.Conv2d(f1 * depth, f1 * depth, (1, separable_kernel),
                                   groups=f1 * depth, bias=False)
        self.pointwise = nn.Conv2d(f1 * depth, f2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(f2, eps=0.001, momentum=0.01)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(f2 * (samples // 32), 2)

    @staticmethod
    def _same(x, kernel):
        return F.pad(x, ((kernel - 1) // 2, kernel // 2, 0, 0))

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.channels, self.samples):
            raise ValueError("EEGNet expects (batch, channels, samples)")
        x = self.bn1(self.temporal(self._same(x.unsqueeze(1), self.temporal_kernel)))
        x = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(x)))))
        x = self.pointwise(self.depthwise(self._same(x, self.separable_kernel)))
        x = self.drop2(self.pool2(F.elu(self.bn3(x))))
        return self.classifier(x.flatten(1))

    @torch.no_grad()
    def constrain(self):
        for weight, maximum in ((self.spatial.weight, 1.0), (self.classifier.weight, 0.25)):
            shape = [len(weight)] + [1] * (weight.ndim - 1)
            norms = weight.flatten(1).norm(dim=1).reshape(shape)
            weight.mul_(torch.clamp(maximum / norms.clamp_min(1e-12), max=1.0))


class ChannelScaler:
    """One source-training mean/std per channel; no target statistics."""
    def fit(self, x):
        x = self._validate(x)
        self.mean_ = x.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
        self.scale_ = x.std(axis=(0, 2), keepdims=True, dtype=np.float64)
        self.scale_ = np.where(self.scale_ > 1e-12, self.scale_, 1.0)
        return self

    @staticmethod
    def _validate(x):
        x = np.asarray(x)
        if x.ndim != 3 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected finite nonempty 3D EEG data")
        return x

    def transform(self, x):
        x = self._validate(x)
        if x.shape[1] != self.mean_.shape[1]:
            raise ValueError("Channel count changed")
        return np.asarray((x - self.mean_) / self.scale_, dtype=np.float32)


def infer(model, x, batch_size=128):
    """Freeze BN/Dropout and prohibit all state updates during inference."""
    x = np.asarray(x, dtype=np.float32)
    if not len(x) or not np.isfinite(x).all() or batch_size < 1:
        raise ValueError("Invalid inference input")
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            parts.append(model(torch.from_numpy(x[start:start + batch_size])).softmax(1)[:, 1].numpy())
    if any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise AssertionError("Model state changed during inference")
    result = np.concatenate(parts).astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite probabilities")
    return result


def train_epoch(model, optimizer, x, y, generator, batch_size=128):
    """One permutation of the balanced source pool; no augmentation or dropping.

    The runner asserts identical sample counts for every subject/class cell.
    Thus each epoch gives exactly equal total weight to these cells.
    """
    model.train()
    order = torch.randperm(len(y), generator=generator)
    total = 0.0
    for indices in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[indices]), y[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite training gradient")
        optimizer.step()
        model.constrain()
        total += loss.item() * len(indices)
    return total / len(y)
