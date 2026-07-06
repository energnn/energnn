# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import jax
import jax.numpy as jnp
from typing import Optional


def _apply_reduction(loss: jnp.ndarray, reduction: str) -> jnp.ndarray:
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def l1_loss(input: jnp.ndarray, target: jnp.ndarray, reduction: str = "mean") -> jnp.ndarray:
    loss = jnp.abs(input - target)
    return _apply_reduction(loss, reduction)


def mse_loss(input: jnp.ndarray, target: jnp.ndarray, reduction: str = "mean") -> jnp.ndarray:
    loss = (input - target) ** 2
    return _apply_reduction(loss, reduction)


def smooth_l1_loss(input: jnp.ndarray, target: jnp.ndarray, reduction: str = "mean", beta: float = 1.0) -> jnp.ndarray:
    diff = jnp.abs(input - target)
    loss = jnp.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return _apply_reduction(loss, reduction)


def huber_loss(input: jnp.ndarray, target: jnp.ndarray, reduction: str = "mean", delta: float = 1.0) -> jnp.ndarray:
    diff = jnp.abs(input - target)
    loss = jnp.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    return _apply_reduction(loss, reduction)


def poisson_nll_loss(
    input: jnp.ndarray,
    target: jnp.ndarray,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> jnp.ndarray:
    if log_input:
        loss = jnp.exp(input) - target * input
    else:
        loss = input - target * jnp.log(input + eps)

    if full:
        # Stirling approximation: target * log(target) - target + 0.5 * log(2 * pi * target)
        # For target > 1
        stirling = target * jnp.log(target + eps) - target + 0.5 * jnp.log(2 * jnp.pi * target + eps)
        loss += stirling

    return _apply_reduction(loss, reduction)


def binary_cross_entropy(
    input: jnp.ndarray, target: jnp.ndarray, weight: Optional[jnp.ndarray] = None, reduction: str = "mean"
) -> jnp.ndarray:
    # PyTorch clamps log outputs to -100
    eps = 1e-12  # Small value to avoid log(0)
    input = jnp.clip(input, eps, 1.0 - eps)
    loss = -(target * jnp.log(input) + (1.0 - target) * jnp.log(1.0 - input))

    if weight is not None:
        loss = loss * weight

    return _apply_reduction(loss, reduction)


def binary_cross_entropy_with_logits(
    input: jnp.ndarray,
    target: jnp.ndarray,
    weight: Optional[jnp.ndarray] = None,
    reduction: str = "mean",
    pos_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    # stable version of BCE with logits
    max_val = jnp.maximum(-input, 0)
    loss = input - input * target + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp(-input - max_val))

    if pos_weight is not None:
        loss = loss * (target * pos_weight + (1.0 - target))

    if weight is not None:
        loss = loss * weight

    return _apply_reduction(loss, reduction)


def nll_loss(
    input: jnp.ndarray,
    target: jnp.ndarray,
    weight: Optional[jnp.ndarray] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> jnp.ndarray:
    # input: (N, C, ...)
    # target: (N, ...)
    # Move class dimension to the end for easier indexing
    if input.ndim > 2:
        # (N, C, d1, d2, ...) -> (N, d1, d2, ..., C)
        input = jnp.moveaxis(input, 1, -1)

    # num_classes = input.shape[-1]

    # Create mask for ignore_index
    mask = target != ignore_index

    # Clip target to [0, num_classes - 1] to avoid indexing errors,
    # but we'll mask them out anyway
    safe_target = jnp.where(mask, target, 0)

    # Gather log-probs
    # Using jax.nn.one_hot or advanced indexing
    # Advanced indexing: loss = -input[n, d1, d2, ..., target[n, d1, d2, ...]]
    # We can use jnp.take_along_axis
    loss = -jnp.take_along_axis(input, jnp.expand_dims(safe_target, axis=-1), axis=-1).squeeze(axis=-1)

    if weight is not None:
        # weight is (C,)
        curr_weight = weight[safe_target]
        loss = loss * curr_weight
        total_weight = jnp.sum(jnp.where(mask, curr_weight, 0.0))
    else:
        total_weight = jnp.sum(mask.astype(jnp.float32))

    loss = jnp.where(mask, loss, 0.0)

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return jnp.sum(loss)
    elif reduction == "mean":
        return jnp.sum(loss) / (total_weight + 1e-12)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def _cross_entropy_with_label_smoothing(
    log_probs: jnp.ndarray,
    target: jnp.ndarray,
    num_classes: int,
    label_smoothing: float,
    weight: Optional[jnp.ndarray] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> jnp.ndarray:
    mask = target != ignore_index
    safe_target = jnp.where(mask, target, 0)

    # We still need to handle ignore_index and weight
    # nll_loss is not suitable directly for smoothed labels if we want to be exact with ignore_index
    # Recompute target_smoothed with mask
    target_one_hot = jax.nn.one_hot(safe_target, num_classes)
    target_smoothed = target_one_hot * (1.0 - label_smoothing) + label_smoothing / num_classes

    if weight is not None:
        weighted_log_probs = log_probs * weight
        loss = -jnp.sum(target_smoothed * weighted_log_probs, axis=-1)
        curr_weight = weight[safe_target]
        total_weight = jnp.sum(jnp.where(mask, curr_weight, 0.0))
    else:
        loss = -jnp.sum(target_smoothed * log_probs, axis=-1)
        total_weight = jnp.sum(mask.astype(jnp.float32))

    loss = jnp.where(mask, loss, 0.0)
    if reduction == "none":
        return loss
    elif reduction == "sum":
        return jnp.sum(loss)
    elif reduction == "mean":
        return jnp.sum(loss) / (total_weight + 1e-12)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def cross_entropy(
    input: jnp.ndarray,
    target: jnp.ndarray,
    weight: Optional[jnp.ndarray] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> jnp.ndarray:
    # input: (N, C, ...)
    # target: (N, ...) if class indices, or (N, C, ...) if probabilities

    # Move class dimension to the end
    if input.ndim > 1:
        # If input is (C,), ndim=1, no moveaxis
        # If input is (N, C, ...), moveaxis 1 to -1
        if input.ndim >= 2:
            # PyTorch: (N, C) or (N, C, d1, d2, ...)
            input = jnp.moveaxis(input, 1, -1)

    log_probs = jax.nn.log_softmax(input, axis=-1)

    if target.shape == input.shape:
        # Probabilities as target
        # Cross entropy with label smoothing is already handled by probabilities
        loss = -jnp.sum(target * log_probs, axis=-1)
        if weight is not None:
            # This is a bit tricky if weights are per class and target is probs
            # PyTorch handles this by weight[class] * target[class] * log_probs[class]
            weighted_log_probs = log_probs * weight
            loss = -jnp.sum(target * weighted_log_probs, axis=-1)
        return _apply_reduction(loss, reduction)
    else:
        # Class indices as target
        if label_smoothing > 0.0:
            return _cross_entropy_with_label_smoothing(
                log_probs, target, input.shape[-1], label_smoothing, weight, ignore_index, reduction
            )
        else:
            # No label smoothing, use nll_loss (input to nll_loss should be log_probs)
            # But wait, nll_loss expects class dimension at index 1 for > 2D
            # My nll_loss implementation moved it to the end, then we passed moved log_probs
            # So we should be careful.

            # Let's restore input shape for nll_loss if it was modified
            if log_probs.ndim >= 2:
                log_probs = jnp.moveaxis(log_probs, -1, 1)

            return nll_loss(log_probs, target, weight=weight, ignore_index=ignore_index, reduction=reduction)


def kl_div(input: jnp.ndarray, target: jnp.ndarray, reduction: str = "mean", log_target: bool = False) -> jnp.ndarray:
    if log_target:
        loss = jnp.exp(target) * (target - input)
    else:
        loss = target * (jnp.log(target + 1e-12) - input)
        # Handle target == 0
        loss = jnp.where(target > 0, loss, 0.0)

    if reduction == "mean":
        # KLDiv mean is special in PyTorch: it's mean over elements, but historically
        # it was mean over batch if it was 2D.
        # Actually in recent PyTorch, 'mean' is 'batchmean' / batch_size ? No.
        # "mean": the sum of the output will be divided by the number of elements in the output.
        # "batchmean": the sum of the output will be divided by the batchsize.
        return jnp.mean(loss)
    elif reduction == "batchmean":
        return jnp.sum(loss) / input.shape[0]
    else:
        return _apply_reduction(loss, reduction)


def soft_margin_loss(input: jnp.ndarray, target: jnp.ndarray, reduction: str = "mean") -> jnp.ndarray:
    loss = jnp.log(1 + jnp.exp(-input * target))
    return _apply_reduction(loss, reduction)


def multi_label_soft_margin_loss(
    input: jnp.ndarray, target: jnp.ndarray, weight: Optional[jnp.ndarray] = None, reduction: str = "mean"
) -> jnp.ndarray:
    # input: (N, C), target: (N, C)
    # loss = - (target * log(sigmoid(input)) + (1-target) * log(1-sigmoid(input)))
    # This is basically BCEWithLogitsLoss averaged over the class dimension
    loss = binary_cross_entropy_with_logits(input, target, reduction="none")
    loss = jnp.mean(loss, axis=-1)

    if weight is not None:
        loss = loss * weight

    return _apply_reduction(loss, reduction)


def multi_margin_loss(
    input: jnp.ndarray,
    target: jnp.ndarray,
    p: int = 1,
    margin: float = 1.0,
    weight: Optional[jnp.ndarray] = None,
    reduction: str = "mean",
) -> jnp.ndarray:
    # input: (N, C), target: (N)
    num_classes = input.shape[-1]
    target_one_hot = jax.nn.one_hot(target, num_classes)

    # input_y = input[n, target[n]]
    input_y = jnp.take_along_axis(input, jnp.expand_dims(target, axis=-1), axis=-1)

    # margin - input_y + input_i
    diffs = margin - input_y + input
    if p == 1:
        loss_all = jnp.maximum(0, diffs)
    elif p == 2:
        loss_all = jnp.maximum(0, diffs) ** 2
    else:
        raise ValueError("p must be 1 or 2")

    # Remove the term where i == y
    loss_all = loss_all * (1.0 - target_one_hot)

    loss = jnp.sum(loss_all, axis=-1) / num_classes

    if weight is not None:
        loss = loss * weight[target]

    return _apply_reduction(loss, reduction)


def multi_label_margin_loss(input: jnp.ndarray, target: jnp.ndarray, reduction: str = "mean") -> jnp.ndarray:
    # input: (N, C), target: (N, C) containing indices, padded with -1
    num_classes = input.shape[-1]
    mask = target >= 0
    safe_target = jnp.where(mask, target, 0)

    # Convert indices to multi-hot mask
    target_multi_hot = jnp.sum(jax.nn.one_hot(safe_target, num_classes) * jnp.expand_dims(mask, -1), axis=1)
    target_multi_hot = jnp.minimum(target_multi_hot, 1.0)

    return multi_label_margin_loss_multihot(input, target_multi_hot, reduction=reduction)


def multi_label_margin_loss_multihot(input: jnp.ndarray, target_mask: jnp.ndarray, reduction: str = "mean") -> jnp.ndarray:
    # Vectorized multi-label margin loss using multi-hot target mask
    # loss = sum_{i in labels, j not in labels} max(0, 1 - (input[i] - input[j])) / C

    pos = jnp.expand_dims(input, axis=-1) * jnp.expand_dims(target_mask, axis=-1)
    neg = jnp.expand_dims(input, axis=-2) * jnp.expand_dims(1.0 - target_mask, axis=-2)

    diff = 1.0 - (pos - neg)
    loss_matrix = jnp.maximum(0, diff) * jnp.expand_dims(target_mask, axis=-1) * jnp.expand_dims(1.0 - target_mask, axis=-2)

    loss = jnp.sum(loss_matrix, axis=(-1, -2)) / input.shape[-1]
    return _apply_reduction(loss, reduction)
