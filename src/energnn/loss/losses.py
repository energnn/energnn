# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from flax import nnx
import jax.numpy as jnp
from . import functional as F


class L1Loss(nnx.Module):
    """L1 Loss module."""

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.l1_loss(input, target, reduction=self.reduction)


class MSELoss(nnx.Module):
    """MSE Loss module."""

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.mse_loss(input, target, reduction=self.reduction)


class SmoothL1Loss(nnx.Module):
    """Smooth L1 Loss module."""

    def __init__(self, reduction: str = "mean", beta: float = 1.0):
        self.reduction = reduction
        self.beta = beta

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)


class HuberLoss(nnx.Module):
    """Huber Loss module."""

    def __init__(self, reduction: str = "mean", delta: float = 1.0):
        self.reduction = reduction
        self.delta = delta

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)


class PoissonNLLLoss(nnx.Module):
    """Poisson NLL Loss module."""

    def __init__(self, log_input: bool = True, full: bool = False, eps: float = 1e-8, reduction: str = "mean"):
        self.log_input = log_input
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.poisson_nll_loss(
            input, target, log_input=self.log_input, full=self.full, eps=self.eps, reduction=self.reduction
        )


class BCELoss(nnx.Module):
    """BCE Loss module."""

    def __init__(self, weight: jnp.ndarray | None = None, reduction: str = "mean"):
        self.weight = weight
        self.reduction = reduction

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class BCEWithLogitsLoss(nnx.Module):
    """BCE with Logits Loss module."""

    def __init__(self, weight: jnp.ndarray | None = None, reduction: str = "mean", pos_weight: jnp.ndarray | None = None):
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.binary_cross_entropy_with_logits(
            input, target, weight=self.weight, reduction=self.reduction, pos_weight=self.pos_weight
        )


class SoftMarginLoss(nnx.Module):
    """Soft Margin Loss module."""

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.soft_margin_loss(input, target, reduction=self.reduction)


class CrossEntropyLoss(nnx.Module):
    """Cross Entropy Loss module."""

    def __init__(
        self,
        weight: jnp.ndarray | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class NLLLoss(nnx.Module):
    """NLL Loss module."""

    def __init__(self, weight: jnp.ndarray | None = None, ignore_index: int = -100, reduction: str = "mean"):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.nll_loss(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


class MultiMarginLoss(nnx.Module):
    """Multi Margin Loss module."""

    def __init__(self, p: int = 1, margin: float = 1.0, weight: jnp.ndarray | None = None, reduction: str = "mean"):
        self.p = p
        self.margin = margin
        self.weight = weight
        self.reduction = reduction

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.multi_margin_loss(input, target, p=self.p, margin=self.margin, weight=self.weight, reduction=self.reduction)


class KLDivLoss(nnx.Module):
    """KL Divergence Loss module."""

    def __init__(self, reduction: str = "mean", log_target: bool = False):
        self.reduction = reduction
        self.log_target = log_target

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)


class MultiLabelMarginLoss(nnx.Module):
    """Multi Label Margin Loss module."""

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.multi_label_margin_loss(input, target, reduction=self.reduction)


class MultiLabelSoftMarginLoss(nnx.Module):
    """Multi Label Soft Margin Loss module."""

    def __init__(self, weight: jnp.ndarray | None = None, reduction: str = "mean"):
        self.weight = weight
        self.reduction = reduction

    def __call__(self, input: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return F.multi_label_soft_margin_loss(input, target, weight=self.weight, reduction=self.reduction)
