# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
import math
import time
from typing import Literal

import flatdict
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from optax import GradientTransformation
from orbax.checkpoint import CheckpointManager
from tqdm import tqdm

from energnn.graph import Graph
from energnn.model import GNN
from energnn.problem import ProblemBatch, ProblemLoader
from energnn.tracker import Tracker
from .utils import TaskLogger

# Types
GraphBatch = Graph

logger = logging.getLogger(__name__)


def _cast_cotangent_to_primal_dtype(cotangent_pytree, primal_pytree):
    """
    Cast each leaf in `cotangent_pytree` to the dtype of the corresponding leaf in `primal_pytree`.
    Leaves that don't appear to have a .dtype are returned unchanged.
    """

    def _cast_leaf(c, p):
        try:
            target_dtype = p.dtype
        except Exception:
            # Keep the original cotangent leaf if we cannot read dtype
            return c
        return jnp.asarray(c, dtype=target_dtype)

    return jax.tree.map(_cast_leaf, cotangent_pytree, primal_pytree)


def _setup_ckpt_mngr(checkpoint_manager: CheckpointManager, optim_mode: Literal["minimize", "maximize"]):
    checkpoint_manager._options.best_fn = lambda x: x["score"]
    if optim_mode == "minimize":
        checkpoint_manager._options.best_mode = "min"
    elif optim_mode == "maximize":
        checkpoint_manager._options.best_mode = "max"


class Trainer:
    r"""
    Trainer implementation.

    This basic trainer relies on the training of a permutation-equivariant
    Graph Neural Network :math:`\hat{y}_\theta` over a dataset of problem instances.
    For a fixed problem instance with objective function :math:`f`
    and context :math:`x`, the parameter :math:`\theta` is updated according to the following gradient descent step,

    .. math::
        \theta \gets \theta - \alpha . J_\theta[\hat{y}_\theta](x)^\top .
        \nabla_y f (\hat{y}_\theta(x);x),

    where :math:`J_\theta[\hat{y}_\theta]` is the Jacobian matrix of the GNN
    :math:`\hat{y}_\theta`, and :math:`\nabla_y f` is the gradient of the
    objective function :math:`f` *w.r.t* the decision :math:`y`.
    For the sake of readability, a basic gradient descent is used --
    with a learning rate :math:`\alpha` --
    but more complex optimizers are possible.

    After every training epoch, the current trainer is checkpointed.

    :param model: Core Graph Neural Network model.
    :type model: GNN
    :param gradient_transformation: Optax gradient transformation.
    :type gradient_transformation: optax.GradientTransformation
    :param profile: If true, synchronize the device after each stage of the training step and log
        per-stage timings. Synchronizations prevent asynchronous dispatch, so this slows training
        down and should only be enabled to investigate performance.
    :type profile: bool
    """

    def __init__(
        self,
        *,
        model: GNN,
        gradient_transformation: GradientTransformation,
        profile: bool = False,
    ):
        self.model: GNN = model
        self.optimizer = nnx.Optimizer(self.model, gradient_transformation, wrt=nnx.Param)
        self.train_step: int = 0
        self.best_score: float = float("nan")
        self.profile = profile

        # Cache JIT-compiled wrappers to avoid NNX re-tracing overhead each step.
        # `step_with_metrics` is static because downstream code branches on its concrete value.
        self._jit_forward_vjp = nnx.jit(self._forward_vjp, static_argnames=("step_with_metrics",))
        # The vjp residuals are dead after the backward: donating them lets XLA reuse
        # their buffers during the backward pass instead of allocating new ones.
        self._jit_backward_update = nnx.jit(self._backward_update, donate_argnums=(2,))
        self._jit_eval_forward = nnx.jit(self._eval_forward)

    @staticmethod
    def _forward_vjp(graphdef, params, rest, jax_context, step_with_metrics):
        """Forward pass + VJP setup, designed to be JIT-compiled once and reused.

        The returned ``vjp_fn`` is a pytree (residual arrays + a stable treedef), so it can be
        passed to the compiled `_backward_update` without triggering a re-trace.
        """

        def f_forward(p, r):
            model = nnx.merge(graphdef, p, r)
            decision, _ = model.forward_batch(graph=jax_context, step_with_metrics=step_with_metrics)
            _, _, r_updated = nnx.split(model, nnx.Param, ...)
            return decision, r_updated

        (jax_decision, rest_updated), vjp_fn = jax.vjp(f_forward, params, rest)
        return jax_decision, rest_updated, vjp_fn

    @staticmethod
    def _backward_update(optimizer: nnx.Optimizer, model: GNN, vjp_fn, jax_cotangent, jax_decision, rest_updated):
        """Backward pass, optimizer update and non-param state update in a single compiled call.

        Calling ``vjp_fn`` here instead of eagerly lets XLA compile and fuse the whole backward.
        The zero cotangent on ``rest_updated`` is a compile-time constant, and the cotangent
        w.r.t. the non-param state is discarded: XLA prunes the corresponding computations.
        """
        cotangent = _cast_cotangent_to_primal_dtype(jax_cotangent, jax_decision)
        (grads_params, _) = vjp_fn((cotangent, jax.tree.map(jnp.zeros_like, rest_updated)))
        optimizer.update(model, grads_params)
        nnx.update(model, rest_updated)

    def _log_stage(self, name: str, value, t_start: float) -> None:
        """In profile mode, wait for `value` and log the elapsed time of a training-step stage."""
        if self.profile:
            jax.block_until_ready(value)
            logger.info(f"[training_step {self.train_step}] {name}: {(time.perf_counter() - t_start) * 1000:.3f} ms")

    @staticmethod
    def _eval_forward(model, context):
        """Forward pass for evaluation, designed to be JIT-compiled once and reused."""
        decision, metrics = model.forward_batch(graph=context, step_with_metrics=True)
        _, _, r_updated = nnx.split(model, nnx.Param, ...)
        return decision, metrics, r_updated

    def train(
        self,
        *,
        train_loader: ProblemLoader,
        val_loader: ProblemLoader | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        n_epochs: int,
        tracker: Tracker | None = None,
        log_period: int | None = None,
        eval_period: int | None = None,
        eval_before_training: bool = False,
        eval_after_epoch: bool = True,
        progress_bar: bool = True,
        optim_mode: Literal["minimize", "maximize"] = "minimize",
    ) -> float:
        r"""
        Trains the model over the train loader, periodically validates the model, tracks metrics, and checkpoints the model.

        :param train_loader: Problem loader used for training.
        :param val_loader: Problem loader used for validation.
        :param checkpoint_manager: Checkpoint manager for saving checkpoints.
        :param n_epochs: Number of training epochs to perform.
        :param tracker: Experiment tracker.
        :param log_period: Number of training iterations between two logs, None for no logs. Logged steps are run
            with `step_with_metrics=True`, so components built with `return_metrics=True` report their metrics.
        :param eval_period: Number of training epochs between two evaluations, None for no evaluations.
        :param eval_before_training: If true, evaluate metrics over the full validation loader before training.
        :param eval_after_epoch: If true, evaluate metrics over the full validation loader after each epoch.
        :param progress_bar: If true, display a progress bar during training.
        :param optim_mode: Optimization mode, either "minimize" or "maximize". Overrides the checkpoint manager's `best_mode`.
        :return: Best average score obtained on the validation loader.
        """
        if checkpoint_manager is not None:
            _setup_ckpt_mngr(checkpoint_manager, optim_mode=optim_mode)

        # Evaluation over the full validation loader before training.
        if eval_before_training and (val_loader is not None):
            _ = self.run_evaluation(
                val_loader=val_loader,
                progress_bar=progress_bar,
                tracker=tracker,
                checkpoint_manager=checkpoint_manager,
                optim_mode=optim_mode,
                position=0,
            )

        for epoch in range(1, n_epochs + 1):

            pbar = tqdm(
                train_loader, desc=f"Epoch {epoch}/{n_epochs}", leave=True, unit="batch", disable=not progress_bar, position=0
            )
            for problem_batch in pbar:

                # Perform one training step
                if (log_period is not None) and (self.train_step % log_period == 0) and (tracker is not None):
                    metrics = self.training_step(problem_batch, step_with_metrics=True)
                    tracker.run_append(metrics={"train": metrics}, step=self.train_step)
                else:
                    _ = self.training_step(problem_batch, step_with_metrics=False)

                # If True, run evaluation
                if (eval_period is not None) and (self.train_step % eval_period == 0) and (val_loader is not None):
                    _ = self.run_evaluation(
                        val_loader=val_loader,
                        progress_bar=progress_bar,
                        tracker=tracker,
                        checkpoint_manager=checkpoint_manager,
                        optim_mode=optim_mode,
                        position=0,
                    )

                self.train_step += 1

            # At the end of each epoch, save latest model and perform an evaluation, unless evaluation was just run.
            if (eval_period is not None) and (self.train_step % eval_period == 0):
                continue
            elif eval_after_epoch and (val_loader is not None):
                _ = self.run_evaluation(
                    val_loader=val_loader,
                    progress_bar=progress_bar,
                    tracker=tracker,
                    checkpoint_manager=checkpoint_manager,
                    optim_mode=optim_mode,
                    position=0,
                )

        if checkpoint_manager is not None:
            checkpoint_manager.wait_until_finished()
        return self.best_score

    def run_evaluation(
        self,
        *,
        val_loader,
        progress_bar: bool = True,
        tracker: Tracker | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        optim_mode: Literal["minimize", "maximize"] = "minimize",
        position: int = 0,
    ) -> float:
        """
        Runs an evaluation and checkpoints.

        :param val_loader: Validation data loader.
        :param progress_bar: If true, display a progress bar during evaluation.
        :param tracker: Experiment tracker.
        :param checkpoint_manager: Checkpoint manager for saving checkpoints.
        :param position: Position of the progress bar if shown.
        :return: Average score obtained on the validation set.
        """
        self.model.eval()  # Set model to eval mode

        mean_score, metrics = self.eval(val_loader, progress_bar=progress_bar, position=position)
        if math.isnan(self.best_score):
            self.best_score = mean_score
        else:
            if (optim_mode == "minimize") and (mean_score < self.best_score):
                self.best_score = mean_score
            elif (optim_mode == "maximize") and (mean_score > self.best_score):
                self.best_score = mean_score

        if tracker is not None:
            tracker.run_append(metrics={"eval": metrics}, step=self.train_step)

        if checkpoint_manager is not None:
            self.save_checkpoint(checkpoint_manager=checkpoint_manager, score=mean_score)

        return mean_score

    def save_checkpoint(self, *, checkpoint_manager: CheckpointManager, score: float) -> None:
        """Saves the current model and optimizer state as a checkpoint.

        :param checkpoint_manager: Checkpoint manager to use for saving the checkpoint.
        :param score: Mean score obtained on the validation set.

        Returns:
            str | None: Local path to the saved checkpoint directory,
             or None if the checkpoint manager did not save the checkpoint.
        """
        _, model_state = nnx.split(self.model)
        _, opt_state = nnx.split(self.optimizer)
        checkpoint_data = {
            "model": model_state,
            "optimizer": opt_state,
            "step": self.train_step,
            "score": score,
        }
        checkpoint_manager.save(self.train_step, args=ocp.args.StandardSave(checkpoint_data))

    def load_checkpoint(self, checkpoint_manager: CheckpointManager, step: int | None = None, best: bool = False) -> None:
        """Loads a checkpoint from the checkpoint manager.

        :param checkpoint_manager: Checkpoint manager to use for loading the checkpoint.
        :param step: Step of the checkpoint to load. If None, load the latest checkpoint.
        :param best: If true, load the best checkpoint.
        """
        if best:
            step = checkpoint_manager.best_step()
        elif step is None:
            step = checkpoint_manager.latest_step()

        _, model_state = nnx.split(self.model)
        _, opt_state = nnx.split(self.optimizer)
        abstract_checkpoint_data = {"model": model_state, "optimizer": opt_state, "step": self.train_step, "score": 0.0}
        restored = checkpoint_manager.restore(step, args=ocp.args.StandardRestore(abstract_checkpoint_data))
        nnx.update(self.model, restored["model"])
        nnx.update(self.optimizer, restored["optimizer"])
        self.train_step = restored["step"]

    def eval(self, loader: ProblemLoader, progress_bar: bool = False, position: int = 0) -> tuple[float, dict]:
        """
        Evaluates the amortizer over a problem loader by averaging the score scalar.

        :param loader: Problem loader over which the amortizer is evaluated.
        :param progress_bar: If true, display a progress bar during evaluation.
        :param position: Position of the progress bar if shown.
        :return: Average score obtained over the problem loader.
        """
        score_list, metrics_list = [], []
        pbar = tqdm(loader, desc="Validation", unit="batch", leave=True, disable=not progress_bar, position=position)
        for step, problem_batch in enumerate(pbar):
            score_batch, metrics_batch = self.eval_step(step, problem_batch)
            score_list.append(score_batch)
            metrics_list.append(metrics_batch)
            if progress_bar:
                pbar.set_postfix(score=f"{np.nanmean(np.concatenate(score_list)):.4e}")

        mean_score = np.nanmean(np.concatenate(score_list)).astype(float)

        # Concatenate all metrics together.
        keys = set.union(*[set(metrics_batch.keys()) for metrics_batch in metrics_list])
        metrics = {}
        for k in keys:
            vals = [metrics.get(k, np.array([])) for metrics in metrics_list]
            if any(np.ndim(v) == 0 for v in vals):
                metrics[k] = np.stack(vals)
            else:
                metrics[k] = np.concatenate(vals)
        metrics["score"] = mean_score

        return mean_score, metrics

    def training_step(self, problem_batch: ProblemBatch, step_with_metrics: bool) -> dict:
        """
        Performs a training step to update model parameters.

        :param problem_batch: A batch of problems for training.
        :param step_with_metrics: Whether this step collects metrics. Components only return metrics on such steps,
            and only if they were built with `return_metrics=True`.
        :return: A flat dictionary of metrics about the training step (empty entries when not collected).
        """
        with TaskLogger(logger, f"Training step {self.train_step}"):

            self.model.train()  # Set model to train mode

            metrics = {}
            t_start = time.perf_counter()
            jax_context, metrics["1_context"] = problem_batch.get_context(
                step_with_metrics=step_with_metrics, step=self.train_step
            )
            self._log_stage("get_context", jax_context, t_start)

            t_start = time.perf_counter()
            graphdef, params, *rest = nnx.split(self.model, nnx.Param, ...)
            jax_decision, rest_updated, vjp_fn = self._jit_forward_vjp(graphdef, params, rest, jax_context, step_with_metrics)
            self._log_stage("forward", jax_decision, t_start)

            t_start = time.perf_counter()
            jax_gradient, metrics["3_gradient"] = problem_batch.get_gradient(
                decision=jax_decision, step_with_metrics=step_with_metrics, step=self.train_step
            )
            self._log_stage("get_gradient", jax_gradient, t_start)

            t_start = time.perf_counter()
            self._jit_backward_update(self.optimizer, self.model, vjp_fn, jax_gradient, jax_decision, rest_updated)
            self._log_stage("backward_update", nnx.state(self.model), t_start)

            metrics["4_update"] = {}

        # Flatten and numpify metrics
        flattened_metrics = flatdict.FlatDict(metrics, delimiter="/")
        result_metrics = {k: np.array(v) for k, v in flattened_metrics.items()}

        return result_metrics

    def eval_step(self, eval_step: int, problem_batch: ProblemBatch) -> tuple[list[float], dict]:
        """Evaluates the current gnn over a batch of problems.

        :param eval_step: Index of the current evaluation step.
        :param problem_batch: A problem batch.
        :return: A batch of scores and a dictionary of batched metrics.
        """
        with TaskLogger(logger, f"Eval step {eval_step}"):
            metrics = {}

            jax_context, metrics["1_context"] = problem_batch.get_context(step_with_metrics=True, step=self.train_step)

            jax_decision, metrics["2_forward"], rest_updated = self._jit_eval_forward(model=self.model, context=jax_context)

            score, metrics["3_score"] = problem_batch.get_score(
                decision=jax_decision, step_with_metrics=True, step=self.train_step
            )

        # Flatten and numpify metrics
        flattened_metrics = flatdict.FlatDict(metrics, delimiter="/")
        result_metrics = {k: np.array(v) for k, v in flattened_metrics.items()}

        return score, result_metrics