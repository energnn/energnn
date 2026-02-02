from abc import ABC, abstractmethod
from energnn.graph import JaxGraph
import jax
from flax import nnx


class Coupler(nnx.Module, ABC):
    """Interface for a coupler.

    It should take as input a graph and return latent coordinates for each address.
    Graph information should be injected into the latent coordinates in a permutation-equivariant manner.
    """

    def __init__(self, *, seed: int = 0):
        self.rngs = nnx.Rngs(seed)

    @abstractmethod
    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[jax.Array, dict]:
        raise NotImplementedError
