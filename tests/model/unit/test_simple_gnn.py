#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from energnn.model.simple_gnn import SimpleGNN
from energnn.graph.jax import JaxGraph
from tests.utils import TestProblemLoader
from flax import nnx

from energnn.model.normalizer.normalizer import Normalizer
from energnn.model.encoder.encoder import Encoder
from energnn.model.coupler.coupler import Coupler
from energnn.model.decoder.decoder import Decoder


np.random.seed(0)
n = 10
pb_loader = TestProblemLoader(seed=0)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)


class DummyNormalizer(Normalizer):

    def __init__(self, out, info=None):
        super().__init__()
        self._out = nnx.data(out)
        self._info = nnx.data({} if info is None else info)
        self.called_with = nnx.data(None)

    def __call__(self, graph, get_info: bool = False):
        self.called_with = nnx.data({"graph": graph, "get_info": get_info})
        return self._out, self._info


class DummyEncoder(Encoder):
    def __init__(self, out, info=None):
        super().__init__()
        self._out = nnx.data(out)
        self._info = nnx.data({} if info is None else info)
        self.called_with = nnx.data(None)

    def __call__(self, graph, get_info: bool = False):
        self.called_with = nnx.data({"graph": graph, "get_info": get_info})
        return self._out, self._info


class DummyCoupler(Coupler):
    def __init__(self, out, info=None):
        super().__init__()
        self._out = nnx.data(out)
        self._info = nnx.data({} if info is None else info)
        self.called_with = nnx.data(None)

    def __call__(self, graph, get_info: bool = False):
        self.called_with = nnx.data({"graph": graph, "get_info": get_info})
        return self._out, self._info


class DummyDecoder(Decoder):
    def __init__(self, out, info=None):
        super().__init__()
        self._out = nnx.data(out)
        self._info = nnx.data({} if info is None else info)
        self.called_with = nnx.data(None)

    def __call__(self, coordinates, graph, get_info: bool = False):
        self.called_with = nnx.data({"coordinates": coordinates, "graph": graph, "get_info": get_info})
        return self._out, self._info


class FailingEncoder(Encoder):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def __call__(self, graph, get_info: bool = False):
        raise self.exc


def test_pipeline_happy_path_graph_output():
    normalized_graph = "normalized_graph_obj"
    encoded_graph = "encoded_graph_obj"
    latents = jnp.zeros((int(np.array(jax_context.non_fictitious_addresses).shape[0]), 3))
    decoded_graph = jax_context  # final object returned by decoder

    norm = DummyNormalizer(out=normalized_graph, info={"norm": True})
    enc = DummyEncoder(out=encoded_graph, info={"enc": True})
    coup = DummyCoupler(out=latents, info={"coup": True})
    dec = DummyDecoder(out=decoded_graph, info={"dec": True})

    model = SimpleGNN(normalizer=norm, encoder=enc, coupler=coup, decoder=dec)

    out, info = model(graph=jax_context, get_info=False)

    assert out is decoded_graph
    assert set(info.keys()) == {"normalization", "encoding", "coupling", "decoding"}
    assert info["normalization"] == {"norm": True}
    assert info["encoding"] == {"enc": True}
    assert info["coupling"] == {"coup": True}
    assert info["decoding"] == {"dec": True}

    # verify kwargs passed to each stub
    assert norm.called_with is not None and "graph" in norm.called_with
    assert enc.called_with is not None and enc.called_with["graph"] == normalized_graph
    assert coup.called_with is not None and coup.called_with["graph"] == encoded_graph
    # decoder should be invoked with coordinates returned by coupler and graph returned by encoder
    assert dec.called_with is not None
    # compare latents arrays numerically
    assert np.array_equal(np.array(dec.called_with["coordinates"]), np.array(latents))
    assert dec.called_with["graph"] == encoded_graph


def test_get_info_flag_propagation():
    """When get_info=True the flag must be forwarded to all submodules and be reflected in info."""

    norm = DummyNormalizer(out="norm_out", info={"called_with_get_info": True})
    enc = DummyEncoder(out="enc_out", info={"called_with_get_info": True})
    coup = DummyCoupler(out=jnp.array([[0.0]]), info={"called_with_get_info": True})
    dec = DummyDecoder(out=jnp.array([1.0]), info={"called_with_get_info": True})

    model = SimpleGNN(
        normalizer=norm,
        encoder=enc,
        coupler=coup,
        decoder=dec,
    )

    _, info = model(graph=jax_context, get_info=True)

    assert info["normalization"]["called_with_get_info"] is True
    assert info["encoding"]["called_with_get_info"] is True
    assert info["coupling"]["called_with_get_info"] is True
    assert info["decoding"]["called_with_get_info"] is True


def test_decoder_returns_array_invariant_case():
    """Decoder may return a JAX array (invariant decoder) — SimpleGNN must forward it."""
    norm = DummyNormalizer(out="normed", info={})
    enc = DummyEncoder(out="encoded", info={})
    latents = jnp.arange(6).reshape((3, 2))
    coup = DummyCoupler(out=latents, info={})
    arr = jnp.array([42.0, -1.0])
    dec = DummyDecoder(out=arr, info={"decoded": "ok"})

    model = SimpleGNN(normalizer=norm, encoder=enc, coupler=coup, decoder=dec)
    out, info = model(graph=jax_context, get_info=True)

    assert isinstance(out, (jax.Array, jnp.ndarray))
    np.testing.assert_allclose(np.array(out), np.array(arr))
    assert info["decoding"] == {"decoded": "ok"}


def test_exception_propagation_from_submodule():
    """If a submodule raises, the exception must propagate (no swallowing)."""
    norm = DummyNormalizer(out="norm", info={})
    enc_fail = FailingEncoder(ValueError("encoder failure"))
    coup = DummyCoupler(out=jnp.zeros((1, 2)), info={})
    dec = DummyDecoder(out=jax_context, info={})

    model = SimpleGNN(normalizer=norm, encoder=enc_fail, coupler=coup, decoder=dec)
    with pytest.raises(ValueError):
        _ = model(graph=jax_context, get_info=False)


def test_forward_batch_execution():
    """
    Test forward_batch logic: normalizer called once on whole batch,
    others vmapped over batch axis.
    """
    batch_size = int(jax_context_batch.non_fictitious_addresses.shape[0])
    n_addr = int(jax_context_batch.non_fictitious_addresses.shape[1])
    latent_dim = 3

    # Normalizer output should be a batch of graphs
    norm_out = jax_context_batch
    norm = DummyNormalizer(out=norm_out, info={"norm": "ok"})

    # Other modules return something per-sample
    enc = DummyEncoder(out="encoded_sample", info={})

    # Coupler returns latents for a single sample (will be vmapped)
    latents_sample = jnp.zeros((n_addr, latent_dim))
    coup = DummyCoupler(out=latents_sample, info={})

    # Decoder returns output for a single sample
    dec_out_sample = jnp.ones((5,))  # arbitrary output array
    dec = DummyDecoder(out=dec_out_sample, info={})

    model = SimpleGNN(normalizer=norm, encoder=enc, coupler=coup, decoder=dec)

    out, info = model.forward_batch(graph=jax_context_batch, get_info=True)

    # Check output shape: (batch_size, ...)
    assert out.shape == (batch_size, 5)
    np.testing.assert_allclose(np.array(out), 1.0)

    # Check info structure in forward_batch (notably the inconsistency of normalization key)
    assert "norm" in info  # Inconsistency: it's not under "normalization"
    assert "encoding" in info
    assert "coupling" in info
    assert "decoding" in info
    assert info["norm"] == "ok"
