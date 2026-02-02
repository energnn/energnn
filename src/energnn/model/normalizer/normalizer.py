from abc import ABC, abstractmethod

from flax import nnx

from energnn.graph import JaxGraph


class Normalizer(nnx.Module, ABC):
    """Interface for a normalizer.

    It should take as input a graph and return a normalized version of it.
    """

    @abstractmethod
    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        raise NotImplementedError
