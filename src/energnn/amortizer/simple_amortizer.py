#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from __future__ import annotations
import logging
import os
from functools import partial
from typing import Any

import cloudpickle
import flatdict
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, vmap
from omegaconf import DictConfig
from optax import GradientTransformation
from tqdm import tqdm

from energnn.amortizer.utils import TaskLogger
from energnn.gnn import EquivariantGNN
from energnn.graph import Graph, separate_graphs
from energnn.graph.jax import JaxGraph
from energnn.normalizer import Postprocessor, Preprocessor
from energnn.problem import ProblemBatch, ProblemLoader
from energnn.storage import Storage
from energnn.tracker import Tracker

# Types
GraphBatch = Graph

logger = logging.getLogger(__name__)


class SimpleAmortizer:
    r"""
    Simple amortizer implementation.

    This basic amortizer relies on the training of a permutation-equivariant
    Graph Neural Network :math:`\hat{y}_\theta` over a dataset of problem instances.
    For a fixed problem instance with objective function :math:`f`
    and context :math:`x`, the parameter :math:`\theta` is updated as follows,

    .. math::
        \theta \gets \theta - \alpha . D_\theta[\hat{y}_\theta](x)^\top .
        \nabla_y f (\hat{y}_\theta(x);x),

    where :math:`D_\theta[\hat{y}_\theta]` is the Jacobian matrix of the GNN
    :math:`\hat{y}_\theta`, and :math:`\nabla_y f` is the gradient of the
    objective function :math:`f` *w.r.t* the decision :math:`y`.
    For the sake of readability, a basic gradient descent is used --
    with a learning rate :math:`\alpha` --
    but more complex optimizers are possible.

    The GNN :math:`\hat{y}_\theta` involves pre-processing and post-processing
    layers to improve the training stability.

    After every training epoch, the current amortizer is saved.
    The amortizer is also frequently evaluated against the validation set,
    and saved only if the average metrics has improved.

    :param gnn: Core Graph Neural Network model.
    :param preprocessor: Pre-processing layer applied over the GNN input.
    :param postprocessor: Post-processing layer applied over the GNN output.
    :param optimizer: Gradient descent method for updating parameter :math:`\theta`.
    :param progress_bar: Whether to display tqdm progress bars during training and evaluation.
    """

    def __init__(
        self,
        *,
        gnn: EquivariantGNN,
        preprocessor: Preprocessor,
        postprocessor: Postprocessor,
        optimizer: GradientTransformation,
        progress_bar: bool = True,
    ):
        self.gnn: EquivariantGNN = gnn
        self.preprocessor: Preprocessor = preprocessor
        self.postprocessor: Postprocessor = postprocessor
        self.optimizer: GradientTransformation = optimizer
        self.progress_bar = progress_bar
        self.params: dict
        self.best_metrics: float = 1e9
        self.initialized: bool = False
        self.opt_state: Any = None
        self.train_step: int = 0

    def init(self, *, rngs: jax.Array, loader: ProblemLoader, problem_cfg: DictConfig) -> None:
        """
        Initializes the GNN weights and the optimizer state, and fits normalization layers on the loader.


        :param rngs: Jax random key.
        :param loader: Problem loader used to fit matrix dimensions, and normalization layers.
        :param problem_cfg: Problem configuration.
        """
        self.preprocessor.fit_problem_loader(problem_loader=loader, refit=False, progress_bar=self.progress_bar)
        self.postprocessor.fit_problem_loader(
            problem_loader=loader, problem_cfg=problem_cfg, refit=False, progress_bar=self.progress_bar
        )

        pb_batch = next(iter(loader))
        context, _ = pb_batch.get_context()
        context = separate_graphs(context)[0]
        decision_structure = pb_batch.get_decision_structure()
        jax_context = JaxGraph.from_numpy_graph(context)
        self.params = self.gnn.init(rngs=rngs, context=jax_context, out_structure=decision_structure)
        self.opt_state = self.optimizer.init(self.params)
        self.train_step = 0
        self.best_metrics = 1e9
        self.initialized = True

    def train(
        self,
        *,
        train_loader: ProblemLoader,
        val_loader: ProblemLoader,
        problem_cfg: DictConfig,
        n_epochs: int,
        out_dir: str,
        last_id: str,
        best_id: str,
        storage: Storage,
        tracker: Tracker,
        log_period: int | None = 1,
        save_period: int | None = 1,
        eval_period: int | None = 1,
        eval_before_training: bool = False,
    ) -> float:
        r"""
        Trains the GNN over the train loader, monitors metrics and saves the best gnn on the validation set.

        :param train_loader: Problem loader used for training.
        :param val_loader: Problem loader used for validation.
        :param problem_cfg: Problem configuration.
        :param n_epochs: Number of training epochs to perform.
        :param out_dir: Path to the local output directory.
        :param last_id: Unique ID associated with the current last gnn.
        :param best_id: Unique ID associated with the current best gnn.
        :param storage: Remote storage manager.
        :param tracker: Experiment tracker.
        :param log_period: Number of training iterations between two logs, None for no logs.
        :param save_period: Number of training iterations between two saves, None for no saves.
        :param eval_period: Number of training epochs between two evaluations, None for no evaluations.
        :param eval_before_training: If true, evaluate metrics over the full validation set before training.
        :return: Best average metrics obtained on the validation set.
        :raises RuntimeError: If called before `init()` or with uninitialized parameters.
        """
        if not self.initialized:
            raise RuntimeError("Amortizer must be initialized by `init()` before training.")

        # Evaluation over the full validation set before training.
        if eval_before_training:
            self.run_evaluation(
                val_loader=val_loader,
                cfg=problem_cfg,
                tracker=tracker,
                storage=storage,
                out_dir=out_dir,
                best_id=best_id,
            )

        for _ in tqdm(range(1, n_epochs + 1), desc="Training", unit="epoch", disable=not self.progress_bar):

            for problem_batch in tqdm(
                train_loader, desc="Current epoch", leave=False, unit="batch", disable=not self.progress_bar
            ):

                # Perform one training step
                if (log_period is not None) and (self.train_step % log_period == 0):
                    infos = self.training_step(problem_batch, cfg=problem_cfg, get_info=True)
                    tracker.run_append(infos={"train": infos}, step=self.train_step)
                else:
                    _ = self.training_step(problem_batch, cfg=problem_cfg, get_info=False)

                # If True, save latest model
                if (save_period is not None) and (self.train_step % save_period == 0):
                    self.save_latest(out_dir=out_dir, last_id=last_id, storage=storage)

                # If True, run evaluation
                if (eval_period is not None) and (self.train_step % eval_period == 0):
                    self.run_evaluation(
                        val_loader=val_loader,
                        cfg=problem_cfg,
                        tracker=tracker,
                        storage=storage,
                        out_dir=out_dir,
                        best_id=best_id,
                    )

                self.train_step += 1

            # At the end of each epoch, save latest model and perform an evaluation.
            self.save_latest(out_dir=out_dir, last_id=last_id, storage=storage)
            self.run_evaluation(
                val_loader=val_loader,
                cfg=problem_cfg,
                tracker=tracker,
                storage=storage,
                out_dir=out_dir,
                best_id=best_id,
            )

        return self.best_metrics

    def run_evaluation(self, *, val_loader, cfg: DictConfig, tracker: Tracker, storage: Storage, out_dir: str, best_id: str):
        """
        Runs an evaluation and saves the model if it returns better metrics than the best one.

        :param val_loader: Validation data loader.
        :param cfg: Problem configuration for evaluation.
        :param tracker: Tracker for logging experiment's metrics and infos.
        :param storage: Remote storage manager for uploading the best gnn.
        :param out_dir: Directory to store local checkpoint.
        :param best_id: Unique ID associated with the current best gnn.
        """
        metrics, infos = self.eval(val_loader, cfg=cfg)
        tracker.run_append(infos={"eval": infos}, step=self.train_step)
        if metrics < self.best_metrics:
            self.save(name="best", directory=out_dir)
            storage.upload(source_path=os.path.join(out_dir, "best"), target_path="amortizers/" + best_id)
            self.best_metrics = metrics

    def save_latest(self, *, out_dir: str, last_id: str, storage: Storage):
        """
        Save and upload the most recent model checkpoint.

        :param out_dir: Local directory for saving checkpoint.
        :param last_id: Unique ID associated with the current last gnn.
        :param storage: Remote storage manager.
        """
        self.save(name="last", directory=out_dir)
        storage.upload(source_path=os.path.join(out_dir, "last"), target_path="amortizers/" + last_id)

    def eval(self, loader: ProblemLoader, cfg: DictConfig) -> tuple[float, dict]:
        """
        Evaluates the amortizer over a problem loader, by averaging the metrics scalar.

        :param loader: Problem loader over which the amortizer is evaluated.
        :param cfg: Problem configuration.
        :return: Average metrics obtained on the problem loader.
        """
        metrics_list, infos_list = [], []
        for eval_step, problem_batch in enumerate(
            tqdm(loader, desc="Validation", unit="batch", leave=False, disable=not self.progress_bar)
        ):
            metrics_batch, info_batch = self.eval_step(eval_step, problem_batch, cfg)
            metrics_list.append(metrics_batch)
            infos_list.append(info_batch)

        metrics = np.nanmean(np.concatenate(metrics_list)).astype(float)

        # Concatenate all infos together.
        keys = set.union(*[set(info_batch.keys()) for info_batch in infos_list])
        infos = {k: np.concatenate([infos.get(k, np.array([])) for infos in infos_list]) for k in keys}
        infos["metrics"] = metrics

        return metrics, infos

    def training_step(self, problem_batch: ProblemBatch, cfg: DictConfig, get_info: bool) -> dict:
        """
        Performs a training step to update gnn parameters.

        :param problem_batch: a batch of problems for training.
        :param cfg: Problem configuration.
        :param get_info: whether to compute information or not.
        :return: a dictionary of information about the training step, or list of dictionaries.
        """
        with TaskLogger(logger, f"Training step {self.train_step}"):
            infos = {}
            context, infos["1_context"] = problem_batch.get_context(get_info=get_info)
            jax_context = JaxGraph.from_numpy_graph(context)
            jax_decision, infos["2_forward"] = self.forward_batch(params=self.params, context=jax_context, get_info=get_info)
            decision = jax_decision.to_numpy_graph()
            gradient, infos["3_gradient"] = problem_batch.get_gradient(decision=decision, cfg=cfg, get_info=get_info)
            jax_gradient = JaxGraph.from_numpy_graph(gradient)
            self.params, self.opt_state, infos["4_update"] = self.update_params(
                params=self.params,
                opt_state=self.opt_state,
                context=jax_context,
                gradient=jax_gradient,
                get_info=get_info,
            )

        # Flatten and numpify infos
        infos = flatdict.FlatDict(infos, delimiter="/")
        infos = {k: np.array(v) for k, v in infos.items()}

        return infos

    def eval_step(self, eval_step: int, problem_batch: ProblemBatch, cfg: DictConfig) -> tuple[list[float], dict]:
        """Evaluates the current gnn over a batch of problems.

        :param eval_step: Index of the current evaluation step.
        :param problem_batch: A problem batch.
        :param cfg: Problem configuration.
        :return: A batch of metrics and a dictionary of batched information.
        """
        with TaskLogger(logger, f"Eval step {eval_step}"):
            infos = {}
            context, infos["1_context"] = problem_batch.get_context(get_info=True)
            jax_context = JaxGraph.from_numpy_graph(context)
            jax_decision, infos["2_forward"] = self.forward_batch(params=self.params, context=jax_context, get_info=True)
            decision = jax_decision.to_numpy_graph()
            metrics, infos["3_metrics"] = problem_batch.get_metrics(decision=decision, cfg=cfg, get_info=True)

        # Flatten and numpify infos
        infos = flatdict.FlatDict(infos, delimiter="/")
        infos = {k: np.array(v) for k, v in infos.items()}

        return metrics, infos

    @partial(jit, static_argnums=(0, 3))
    def forward_batch(self, params: dict, context: JaxGraph, get_info: bool) -> tuple[JaxGraph, dict]:
        """
        Vectorized forward pass over a batch of context graphs.

        Preprocesses the input batch of graphs, applies the GNN, and postprocesses the batch.

        :param params: Parameter dictionary.
        :param context: A batch context graph
        :param get_info: Whether to compute information or not.
        :returns: A tuple of batched decision graphs and info dictionary.
        """
        return vmap(self.forward, in_axes=(None, 0, None), out_axes=(0, 0))(params, context, get_info)

    @partial(jit, static_argnums=(0, 2))
    def infer(self, context: JaxGraph, get_info: bool) -> tuple[JaxGraph, dict]:
        r"""
        Infers a decision graph :math:`\hat{y}` based on a context graph :math:`x`.

        :param context: Context graph :math:`x`.
        :param get_info: If true, returns an information dictionary for monitoring purposes.
        :return: Decision graph :math:`\hat{y}` and information dictionary.
        """
        return self.forward(params=self.params, context=context, get_info=get_info)

    @partial(jit, static_argnums=(0, 2))
    def infer_batch(self, context: JaxGraph, get_info: bool) -> tuple[JaxGraph, dict]:
        r"""
        Infers a batch of decision graphs :math:`\hat{y}` based on a batch of context graphs :math:`x`.

        :param context: Batch of context graphs :math:`x`.
        :param get_info: If true, returns an information dictionary for monitoring purposes.
        :return: Batch of decision graphs :math:`\hat{y}` and information dictionary.
        """
        return vmap(self.forward, in_axes=(None, 0, None))(self.params, context, get_info)

    def _apply_model(self, params: dict, context: JaxGraph, get_info: bool) -> tuple[JaxGraph, dict]:
        """
        Computes gnn.apply without keyword arguments for vmap usage.

        :param params: Parameter dictionary.
        :param context: Input graph context
        :param get_info: Whether to return intermediate diagnostics or not.
        """
        return self.gnn.apply(params=params, context=context, get_info=get_info)

    def forward(self, params: dict, context: JaxGraph, get_info: bool) -> tuple[JaxGraph, dict]:
        """
        Preprocesses the input graph, applies the GNN, and postprocesses the output.

        :param params: Parameter dictionary.
        :param context: Input graph context
        :param get_info: Whether to return intermediate diagnostics or not.
        """
        infos = {}
        norm_context, infos["preprocess"] = self.preprocessor.preprocess(context, get_info)
        norm_decision, infos["gnn"] = self._apply_model(params, norm_context, get_info)
        decision, infos["postprocess"] = self.postprocessor.postprocess(norm_decision, get_info)
        return decision, infos

    @partial(jit, static_argnames=("self", "get_info"))
    def update_params(
        self, params: dict, opt_state: dict, context: JaxGraph, gradient: JaxGraph, get_info: bool
    ) -> tuple[dict, dict, dict]:
        r"""
        Updates the gnn weights based on the preconditioned gradient.

        The loss is defined as the inner product between model output features
        and preconditioned gradient features, averaged over batch.

        :param params: Parameters dictionary
        param opt_state: Current optimizer state.
        :param context: Batch of context graphs.
        :param gradient: Batch of raw gradient graphs.
        :param get_info: If True, return diagnostic info on grads and updates.
        :returns: Tuple of (new_params, new_opt_state, infos_dict).
        """

        def loss_fn(_params: dict, _norm_context: JaxGraph, _gradient: JaxGraph) -> jax.Array:
            _norm_decision, _ = jax.vmap(self._apply_model, in_axes=(None, 0, None), out_axes=0)(_params, _norm_context, False)
            _prec_gradient, _ = self.postprocessor.precondition_gradient_batch(_norm_decision, _gradient)
            return jnp.nanmean(_norm_decision.feature_flat_array * _prec_gradient.feature_flat_array)

        norm_context, _ = self.preprocessor.preprocess_batch(context)
        loss, grads = jax.value_and_grad(loss_fn)(params, norm_context, gradient)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if get_info:
            infos = {
                "loss": loss,
                "grads/l2_norm": optax.tree_utils.tree_l2_norm(grads),
                "grads/0th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=0), grads),
                "grads/10th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=10), grads),
                "grads/25th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=25), grads),
                "grads/50th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=50), grads),
                "grads/75th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=75), grads),
                "grads/90th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=90), grads),
                "grads/100th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=100), grads),
                "updates/l2_norm": optax.tree_utils.tree_l2_norm(updates),
                "updates/0th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=0), updates),
                "updates/10th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=10), updates),
                "updates/25th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=25), updates),
                "updates/50th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=50), updates),
                "updates/75th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=75), updates),
                "updates/90th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=90), updates),
                "updates/100th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=100), updates),
                "params/l2_norm": optax.tree_utils.tree_l2_norm(params),
                "params/0th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=0), params),
                "params/10th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=10), params),
                "params/25th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=25), params),
                "params/50th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=50), params),
                "params/75th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=75), params),
                "params/90th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=90), params),
                "params/100th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=100), params),
            }
        else:
            infos = {}

        return params, opt_state, infos

    def save(self, *, name: str, directory: str) -> None:
        """Saves an amortizer as a .pkl file.

        :param name: Name of the .pkl file.
        :param directory: Directory where the amortizer should be stored.
        """
        path = os.path.join(directory, name)
        with open(path, "wb") as handle:
            cloudpickle.dump(self, handle)

    @classmethod
    def load(cls, path: str) -> SimpleAmortizer:
        """Loads one amortizer instance.

        :param path: File path to a pickled amortizer.
        :returns: Deserialized SimpleAmortizer instance.
        """
        with open(path, "rb") as handle:
            normalizer = cloudpickle.load(handle)
        return normalizer
