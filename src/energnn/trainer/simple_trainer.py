# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations

import logging
import os
from typing import Any

import cloudpickle
import flatdict
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import DictConfig
from optax import GradientTransformation
from tqdm import tqdm

from energnn.amortizer.utils import TaskLogger
from energnn.graph import Graph, separate_graphs
from energnn.graph.jax import JaxGraph
from energnn.model import SimpleGNN
from energnn.problem import ProblemBatch, ProblemLoader
from energnn.storage import Storage
from energnn.tracker import Tracker

# Types
GraphBatch = Graph

logger = logging.getLogger(__name__)


class SimpleTrainer:
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
        model: SimpleGNN,
        optimizer: GradientTransformation,
        progress_bar: bool = True,
    ):
        self.model: SimpleGNN = model
        self.optimizer = optimizer
        self.nnx_optimizer = None
        self.progress_bar = progress_bar
        self.params: dict
        self.best_metrics: float = 1e9
        self.opt_state: Any = None
        self.train_step: int = 0
        self.initialized: bool = False

    def _initialize_model_and_optimizer(self, loader: ProblemLoader) -> None:
        """Initializes the model and optimizer, based on the first sample of the loader."""
        pb_batch = next(iter(loader))
        context_batch, _ = pb_batch.get_context()
        context = separate_graphs(context_batch)[0]
        jax_context = JaxGraph.from_numpy_graph(context)
        _, _ = self.model(graph=jax_context, get_info=False)
        self.nnx_optimizer = nnx.Optimizer(self.model, self.optimizer, wrt=nnx.Param)
        self.initialized = True

    def train(
        self,
        *,
        train_loader: ProblemLoader,
        val_loader: ProblemLoader,
        n_epochs: int,
        out_dir: str,
        last_id: str,
        best_id: str,
        storage: Storage,
        tracker: Tracker,
        problem_cfg: DictConfig = None,
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
            self._initialize_model_and_optimizer(loader=train_loader)

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

            def apply(model: SimpleGNN, jax_context: JaxGraph):
                # Split model into graphdef and state for differentiation
                # We separate Param (differentiable) from the rest (non-differentiable like normalizer stats, RNG keys)
                graphdef, params, rest = nnx.split(model, nnx.Param, ...)

                def f(params):
                    model = nnx.merge(graphdef, params, rest)
                    decision, _ = self.forward_batch(model=model, context=jax_context, get_info=get_info)
                    return decision

                jax_decision, model_vjp = jax.vjp(f, params)
                return jax_decision, model_vjp

            jax_decision, model_vjp = nnx.jit(apply)(model=self.model, jax_context=jax_context)

            decision = jax_decision.to_numpy_graph()
            gradient, infos["3_gradient"] = problem_batch.get_gradient(decision=decision, cfg=cfg, get_info=get_info)
            jax_gradient = JaxGraph.from_numpy_graph(gradient)

            jax_cotangent = self._cast_cotangent_to_primal_dtype(jax_gradient, jax_decision)

            infos["4_update"] = self.update_params(
                nnx_optimizer=self.nnx_optimizer,
                model=self.model,
                model_vjp=model_vjp,
                gradient=jax_cotangent,
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

            def f(model, context):
                return self.forward_batch(model=model, context=context, get_info=True)

            jax_decision, infos["2_forward"] = jax.jit(f)(model=self.model, context=jax_context)

            # jax_decision, infos["2_forward"] = self.forward_batch(model=self.model, context=jax_context, get_info=True)
            decision = jax_decision.to_numpy_graph()
            metrics, infos["3_metrics"] = problem_batch.get_metrics(decision=decision, cfg=cfg, get_info=True)

        # Flatten and numpify infos
        infos = flatdict.FlatDict(infos, delimiter="/")
        infos = {k: np.array(v) for k, v in infos.items()}

        return metrics, infos

    # @partial(jit, static_argnums=(0, 3))
    def forward_batch(self, model: SimpleGNN, context: JaxGraph, get_info: bool) -> tuple[JaxGraph, dict]:
        """
        Vectorized forward pass over a batch of context graphs.

        Preprocesses the input batch of graphs, applies the GNN, and postprocesses the batch.

        :param params: Parameter dictionary.
        :param context: A batch context graph
        :param get_info: Whether to compute information or not.
        :returns: A tuple of batched decision graphs and info dictionary.
        """

        def apply(model, context, get_info):
            return model(context, get_info=get_info)

        return jax.vmap(apply, in_axes=[None, 0, None], out_axes=0)(model, context, get_info)

    def forward(self, model: SimpleGNN, context: JaxGraph, get_info: bool) -> tuple[JaxGraph, dict]:
        """
        Preprocesses the input graph, applies the GNN, and postprocesses the output.

        :param params: Parameter dictionary.
        :param context: Input graph context
        :param get_info: Whether to return intermediate diagnostics or not.
        """
        return model(context, get_info=get_info)

    def _cast_cotangent_to_primal_dtype(self, cotangent_pytree, primal_pytree):
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

    # @partial(jit, static_argnames=("self", "get_info"))
    def update_params(
        self,
        nnx_optimizer: nnx.Optimizer,
        model: SimpleGNN,
        model_vjp: jax.extend.linear_util.Callable,
        gradient: JaxGraph,
        get_info: bool,
    ) -> dict:
        r"""
        Updates the model weights using the problem gradient.

        :param f_vjp: Vector Jacobian Product of the forward function.
        :param gradient: Batch of raw gradient graphs.
        :param get_info: If True, return diagnostic info on grads and updates.
        :returns: Tuple of (new_params, new_opt_state, infos_dict).
        """

        # model_vjp returns a tuple with gradients for each input (only params in this case)
        # (grads_params,) = nnx.jit(model_vjp)(gradient)  # ça devrait pouvoir se jit.

        # Update model parameters using the gradients
        # As of Flax 0.11.0, update() requires both model and grads

        def update_params(nnx_optimizer, model, gradient, model_vjp):
            (grads_params,) = model_vjp(gradient)
            return nnx_optimizer.update(model, grads_params)

        jit_update_params = nnx.jit(update_params)
        jit_update_params(nnx_optimizer, model, gradient, model_vjp)

        # nnx_optimizer.update(model=model, grads=grads_params)

        if get_info:
            infos = {
                # "grads/l2_norm": optax.tree_utils.tree_l2_norm(grads),
                # "grads/0th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=0), grads),
                # "grads/10th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=10), grads),
                # "grads/25th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=25), grads),
                # "grads/50th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=50), grads),
                # "grads/75th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=75), grads),
                # "grads/90th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=90), grads),
                # "grads/100th-percentile": jax.tree.map(lambda x: jnp.nanpercentile(x, q=100), grads),
            }
        else:
            infos = {}

        return infos

    def __getstate__(self):
        """
        Return a picklable state. Replace JAX PRNG keys by a serializable marker:
        {'__jax_prng_key__': numpy_array_of_key_bits}.
        This avoids cloudpickle trying to pickle PRNGKeyArray/KeyArray objects.
        Only traverse common container types (dict, list, tuple, set).
        For other objects, keep them as-is (cloudpickle will try to pickle them).
        """

        def _serialize(obj):
            # dict
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            # list
            if isinstance(obj, list):
                return [_serialize(v) for v in obj]
            # tuple (covers namedtuple as well)
            if isinstance(obj, tuple):
                seq = [_serialize(v) for v in obj]
                # For tuple subclasses (namedtuple), call constructor with unpacked args.
                try:
                    return type(obj)(*seq)
                except Exception:
                    # Fallback to plain tuple if reconstruction fails
                    return tuple(seq)
            # set
            if isinstance(obj, set):
                return {_serialize(v) for v in obj}
            # Try to treat as a JAX PRNG key: key_data raises if not a key
            try:
                key_bits = jax.random.key_data(obj)
            except Exception:
                # Not a PRNG key: return as-is (let pickle handle it)
                return obj
            else:
                # convert to numpy array (serializable) and mark
                return {"__jax_prng_key__": np.asarray(key_bits)}

        raw_state = self.__dict__.copy()
        return _serialize(raw_state)

    def __setstate__(self, state):
        """
        Reconstruct the object state from the serialized form produced by __getstate__.
        Tries to reconstruct PRNG key objects with jax.random.wrap_key_data() when possible.
        If wrap_key_data is unavailable, leaves the numpy representation in place.
        """
        logger = logging.getLogger(__name__)

        def _deserialize(obj):
            # dict that encodes a PRNG key marker?
            if isinstance(obj, dict):
                if "__jax_prng_key__" in obj and len(obj) == 1:
                    bits = np.asarray(obj["__jax_prng_key__"])
                    try:
                        key = jax.random.wrap_key_data(bits)
                        return key
                    except Exception:
                        logger.debug(
                            "jax.random.wrap_key_data unavailable/failed while unpickling PRNG key; "
                            "restoring numpy array of key bits instead."
                        )
                        return bits
                # general dict: recurse
                return {k: _deserialize(v) for k, v in obj.items()}
            # list/tuple
            if isinstance(obj, list):
                return [_deserialize(v) for v in obj]
            if isinstance(obj, tuple):
                seq = [_deserialize(v) for v in obj]
                try:
                    return type(obj)(*seq)
                except Exception:
                    return tuple(seq)
            # set
            if isinstance(obj, set):
                return {_deserialize(v) for v in obj}
            # leaf
            return obj

        recovered = _deserialize(state)
        self.__dict__.update(recovered)

    def save(self, *, name: str, directory: str) -> None:
        """Saves an amortizer as a .pkl file.

        :param name: Name of the .pkl file.
        :param directory: Directory where the amortizer should be stored.
        """
        path = os.path.join(directory, name)
        with open(path, "wb") as handle:
            cloudpickle.dump(self, handle)

    @classmethod
    def load(cls, path: str) -> SimpleTrainer:
        """Loads one amortizer instance.

        :param path: File path to a pickled amortizer.
        :returns: Deserialized SimpleAmortizer instance.
        """
        with open(path, "rb") as handle:
            normalizer = cloudpickle.load(handle)
        return normalizer
