# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging

import flatdict
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from optax import GradientTransformation
from orbax.checkpoint import CheckpointManager
from tqdm import tqdm

from energnn.amortizer.utils import TaskLogger
from energnn.graph import Graph
from energnn.model import SimpleGNN
from energnn.problem import ProblemBatch, ProblemLoader
from energnn.storage import Storage
from energnn.tracker import Tracker

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


def _update_params(optimizer: nnx.Optimizer, model: SimpleGNN, gradient: nnx.State, get_info: bool) -> dict:
    r"""
    Updates the model weights using the gradient.

    :param optimizer: Optimizer instance.
    :param model: Core Graph Neural Network model.
    :param gradient: Gradient for model parameters.
    :param get_info: If True, return diagnostic info on grads and updates.
    :returns: Dictionary of diagnostic information.
    """

    def update_params(optimizer, model, gradient):
        optimizer.update(model, gradient)

    nnx.jit(update_params)(optimizer, model, gradient)

    if get_info:
        infos = {
            # "grads/l2_norm": optax.tree_utils.tree_l2_norm(grads),
        }
    else:
        infos = {}

    return infos


class SimpleTrainer:
    r"""
    Simple trainer implementation.

    This basic trainer relies on the training of a permutation-equivariant
    Graph Neural Network :math:`\hat{y}_\theta` over a dataset of problem instances.
    For a fixed problem instance with objective function :math:`f`
    and context :math:`x`, the parameter :math:`\theta` is updated as follows,

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
    :param gradient_transformation: Optax gradient transformation (e.g. Adam).
    """

    def __init__(
        self,
        *,
        model: SimpleGNN,
        gradient_transformation: GradientTransformation,
    ):
        self.model: SimpleGNN = model
        self.optimizer = nnx.Optimizer(self.model, gradient_transformation, wrt=nnx.Param)
        self.train_step: int = 0
        self.best_metrics: float = float("inf")

    def train(
        self,
        *,
        train_loader: ProblemLoader,
        val_loader: ProblemLoader,
        storage: Storage | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        n_epochs: int,
        tracker: Tracker | None = None,
        log_period: int | None = 1,
        eval_period: int | None = 1,
        eval_before_training: bool = False,
        eval_after_epoch: bool = True,
        progress_bar: bool = True,
    ) -> float:
        r"""
        Trains the model over the train loader, monitors metrics, and checkpoints the model.

        :param train_loader: Problem loader used for training.
        :param val_loader: Problem loader used for validation.
        :param storage: Remote storage manager for saving checkpoints.
        :param checkpoint_manager: Checkpoint manager for saving checkpoints.
        :param n_epochs: Number of training epochs to perform.
        :param tracker: Experiment tracker.
        :param log_period: Number of training iterations between two logs, None for no logs.
        :param eval_period: Number of training epochs between two evaluations, None for no evaluations.
        :param eval_before_training: If true, evaluate metrics over the full validation loader before training.
        :param eval_after_epoch: If true, evaluate metrics over the full validation loader after each epoch.
        :param progress_bar: If true, display a progress bar during training.
        :return: Best average metrics obtained on the validation loader.
        """

        # Evaluation over the full validation loader before training.
        if eval_before_training:
            _ = self.run_evaluation(
                val_loader=val_loader,
                progress_bar=progress_bar,
                tracker=tracker,
                checkpoint_manager=checkpoint_manager,
                storage=storage,
            )

        for _ in tqdm(range(1, n_epochs + 1), desc="Training", unit="epoch", disable=not progress_bar):

            for problem_batch in tqdm(train_loader, desc="Current epoch", unit="batch", disable=not progress_bar):
                # for problem_batch in tqdm(train_loader, desc="Current epoch", leave=False, unit="batch", disable=not progress_bar):

                # Perform one training step
                if (log_period is not None) and (self.train_step % log_period == 0) and (tracker is not None):
                    infos = self.training_step(problem_batch, get_info=True)
                    tracker.run_append(infos={"train": infos}, step=self.train_step)
                else:
                    _ = self.training_step(problem_batch, get_info=False)

                # If True, run evaluation
                if (eval_period is not None) and (self.train_step % eval_period == 0):
                    _ = self.run_evaluation(
                        val_loader=val_loader,
                        progress_bar=progress_bar,
                        tracker=tracker,
                        checkpoint_manager=checkpoint_manager,
                        storage=storage,
                    )

                self.train_step += 1

            # At the end of each epoch, save latest model and perform an evaluation, unless evaluation was just run.
            if (eval_period is not None) and (self.train_step % eval_period == 0):
                continue
            elif eval_after_epoch:
                _ = self.run_evaluation(
                    val_loader=val_loader,
                    progress_bar=progress_bar,
                    tracker=tracker,
                    checkpoint_manager=checkpoint_manager,
                    storage=storage,
                )

        if checkpoint_manager is not None:
            checkpoint_manager.wait_until_finished()
        return self.best_metrics

    def run_evaluation(
        self,
        *,
        val_loader,
        progress_bar: bool = True,
        tracker: Tracker = None,
        storage: Storage | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> float:
        """
        Runs an evaluation and checkpoints if needed.

        :param val_loader: Validation data loader.
        :param progress_bar: If true, display a progress bar during evaluation.
        :param tracker: Experiment tracker.
        :param storage: Remote storage manager for saving checkpoints.
        :param checkpoint_manager: Checkpoint manager for saving checkpoints.
        :return: Average metrics obtained on the validation set.
        """
        self.model.eval()  # Set model to eval mode

        metrics, infos = self.eval(val_loader, progress_bar=progress_bar)
        if metrics < self.best_metrics:
            self.best_metrics = metrics

        if tracker is not None:
            tracker.run_append(infos={"eval": infos}, step=self.train_step)

        if checkpoint_manager is not None:
            local_ckpt_path = self.save_checkpoint(checkpoint_manager=checkpoint_manager, metrics=metrics)
        else:
            local_ckpt_path = None

        if local_ckpt_path is not None and storage is not None:
            storage.upload(source_path=local_ckpt_path, target_path="Hugo?")

        return metrics

    def save_checkpoint(self, *, checkpoint_manager: CheckpointManager, metrics: float) -> str | None:
        """Saves the current model and optimizer state as a checkpoint.

        :param checkpoint_manager: Checkpoint manager to use for saving the checkpoint.
        :param metrics: Metrics obtained on the validation set.

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
            "metrics": metrics,
        }
        saved = checkpoint_manager.save(
            self.train_step, args=ocp.args.Composite(default=ocp.args.StandardSave(checkpoint_data))
        )
        if saved:
            local_path = checkpoint_manager.directory / str(self.train_step)
        else:
            local_path = None
        return str(local_path) if local_path is not None else None

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
        abstract_checkpoint_data = {"model": model_state, "optimizer": opt_state, "step": self.train_step, "metrics": 0.0}
        restored = checkpoint_manager.restore(
            step, args=ocp.args.Composite(default=ocp.args.StandardRestore(abstract_checkpoint_data))
        )
        restored = restored["default"]
        nnx.update(self.model, restored["model"])
        nnx.update(self.optimizer, restored["optimizer"])
        self.train_step = restored["step"]

    def eval(self, loader: ProblemLoader, progress_bar: bool = False) -> tuple[float, dict]:
        """
        Evaluates the amortizer over a problem loader by averaging the metrics scalar.

        :param loader: Problem loader over which the amortizer is evaluated.
        :param progress_bar: If true, display a progress bar during evaluation.
        :return: Average metrics obtained over the problem loader.
        """
        metrics_list, infos_list = [], []
        for step, problem_batch in enumerate(
            tqdm(loader, desc="Validation", unit="batch", leave=False, disable=not progress_bar)
        ):
            metrics_batch, info_batch = self.eval_step(step, problem_batch)
            metrics_list.append(metrics_batch)
            infos_list.append(info_batch)

        metrics = np.nanmean(np.concatenate(metrics_list)).astype(float)

        # Concatenate all infos together.
        keys = set.union(*[set(info_batch.keys()) for info_batch in infos_list])
        infos = {}
        for k in keys:
            vals = [infos.get(k, np.array([])) for infos in infos_list]
            if any(np.ndim(v) == 0 for v in vals):
                infos[k] = np.stack(vals)
            else:
                infos[k] = np.concatenate(vals)
        infos["metrics"] = metrics

        return metrics, infos

    def training_step(self, problem_batch: ProblemBatch, get_info: bool) -> dict:
        """
        Performs a training step to update model parameters.

        :param problem_batch: A batch of problems for training.
        :param get_info: Whether to compute information or not.
        :return: A dictionary of information about the training step, or list of dictionaries.
        """
        with TaskLogger(logger, f"Training step {self.train_step}"):

            self.model.train()  # Set model to train mode

            infos = {}
            jax_context, infos["1_context"] = problem_batch.get_context(get_info=get_info)

            def apply(params, rest, jax_context):
                def f_forward(p, r):
                    model = nnx.merge(graphdef, p, r)
                    decision, _ = model.forward_batch(graph=jax_context, get_info=get_info)
                    _, _, r_updated = nnx.split(model, nnx.Param, ...)
                    return decision, r_updated

                (jax_decision, rest_updated), vjp_fn = jax.vjp(f_forward, params, rest)
                return jax_decision, rest_updated, vjp_fn

            graphdef, params, rest = nnx.split(self.model, nnx.Param, ...)
            jax_decision, rest_updated, vjp_fn = nnx.jit(apply)(params, rest, jax_context)

            def model_vjp(cotangent):
                # We only care about the cotangent of the decision, not the rest
                rest_cotangent = jax.tree.map(jnp.zeros_like, rest_updated)
                (grads_params, _) = vjp_fn((cotangent, rest_cotangent))
                return (grads_params,)

            nnx.update(self.model, rest_updated)
            jax_gradient, infos["3_gradient"] = problem_batch.get_gradient(decision=jax_decision, get_info=get_info)
            jax_cotangent = _cast_cotangent_to_primal_dtype(jax_gradient, jax_decision)
            (grads_params,) = model_vjp(jax_cotangent)
            infos["4_update"] = _update_params(
                optimizer=self.optimizer,
                model=self.model,
                gradient=grads_params,
                get_info=get_info,
            )

        # Flatten and numpify infos
        infos = flatdict.FlatDict(infos, delimiter="/")
        infos = {k: np.array(v) for k, v in infos.items()}

        return infos

    def eval_step(self, eval_step: int, problem_batch: ProblemBatch) -> tuple[list[float], dict]:
        """Evaluates the current gnn over a batch of problems.

        :param eval_step: Index of the current evaluation step.
        :param problem_batch: A problem batch.
        :return: A batch of metrics and a dictionary of batched information.
        """
        with TaskLogger(logger, f"Eval step {eval_step}"):
            infos = {}

            jax_context, infos["1_context"] = problem_batch.get_context(get_info=True)

            def f(model, context):
                decision, info = model.forward_batch(graph=context, get_info=True)
                _, _, r_updated = nnx.split(model, nnx.Param, ...)
                return decision, info, r_updated

            jax_decision, infos["2_forward"], rest_updated = nnx.jit(f)(model=self.model, context=jax_context)

            metrics, infos["3_metrics"] = problem_batch.get_metrics(decision=jax_decision, get_info=True)

        # Flatten and numpify infos
        infos = flatdict.FlatDict(infos, delimiter="/")
        infos = {k: np.array(v) for k, v in infos.items()}

        return metrics, infos
