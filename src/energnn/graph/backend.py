from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar("T")


class Backend(ABC):
    @property
    @abstractmethod
    def np(self) -> Any:
        """Access the underlying array library (numpy or jax.numpy)."""
        pass

    @abstractmethod
    def array(self, x: Any, dtype: Any = None) -> Any:
        pass

    @abstractmethod
    def ones(self, shape: Any, dtype: Any = None) -> Any:
        pass

    @abstractmethod
    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        pass

    @abstractmethod
    def concatenate(self, arrays: list[Any], axis: int = 0) -> Any:
        pass

    @abstractmethod
    def stack(self, arrays: list[Any], axis: int = 0) -> Any:
        pass

    @abstractmethod
    def unstack(self, x: Any, axis: int = 0) -> list[Any]:
        pass

    @abstractmethod
    def any(self, x: Any) -> bool:
        pass

    @abstractmethod
    def all(self, x: Any) -> bool:
        pass

    @abstractmethod
    def isnan(self, x: Any) -> Any:
        pass

    @abstractmethod
    def shape(self, x: Any) -> tuple[int, ...]:
        pass

    @abstractmethod
    def maximum(self, x: Any, y: Any) -> Any:
        pass


class NumpyBackend(Backend):
    def __eq__(self, other):
        return isinstance(other, NumpyBackend)

    def __hash__(self):
        return hash(NumpyBackend)

    @property
    def np(self) -> Any:
        return np

    def array(self, x: Any, dtype: Any = None) -> Any:
        return np.array(x, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = None) -> Any:
        return np.ones(shape, dtype=dtype)

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        return np.zeros(shape, dtype=dtype)

    def concatenate(self, arrays: list[Any], axis: int = 0) -> Any:
        return np.concatenate(arrays, axis=axis)

    def stack(self, arrays: list[Any], axis: int = 0) -> Any:
        return np.stack(arrays, axis=axis)

    def unstack(self, x: Any, axis: int = 0) -> list[Any]:
        return [np.squeeze(res, axis=axis) for res in np.split(x, x.shape[axis], axis=axis)]

    def any(self, x: Any) -> bool:
        return np.any(x)

    def all(self, x: Any) -> bool:
        return np.all(x)

    def isnan(self, x: Any) -> Any:
        return np.isnan(x)

    def shape(self, x: Any) -> tuple[int, ...]:
        return np.shape(x)

    def maximum(self, x: Any, y: Any) -> Any:
        return np.maximum(x, y)


class JaxBackend(Backend):
    def __eq__(self, other):
        return isinstance(other, JaxBackend)

    def __hash__(self):
        return hash(JaxBackend)

    @property
    def np(self) -> Any:
        return jnp

    def array(self, x: Any, dtype: Any = None) -> Any:
        return jnp.array(x, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = None) -> Any:
        return jnp.ones(shape, dtype=dtype)

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        return jnp.zeros(shape, dtype=dtype)

    def concatenate(self, arrays: list[Any], axis: int = 0) -> Any:
        return jnp.concatenate(arrays, axis=axis)

    def stack(self, arrays: list[Any], axis: int = 0) -> Any:
        return jnp.stack(arrays, axis=axis)

    def unstack(self, x: Any, axis: int = 0) -> list[Any]:
        return jnp.unstack(x, axis=axis)

    def any(self, x: Any) -> bool:
        return jnp.any(x)

    def all(self, x: Any) -> bool:
        return jnp.all(x)

    def isnan(self, x: Any) -> Any:
        return jnp.isnan(x)

    def shape(self, x: Any) -> tuple[int, ...]:
        return jnp.shape(x)

    def maximum(self, x: Any, y: Any) -> Any:
        return jnp.maximum(x, y)
