"""
Probability distribution wrappers for stochastic simulations.

Each distribution exposes a ``sample()`` method (also callable via ``()``).
These are thin wrappers around ``random`` that keep simulation code
declarative::

    service = Exponential(mean=2.0)
    next_service_time = service()
"""
import math
import random
from typing import Callable


class Distribution:
    """Abstract base for all distributions."""

    def sample(self) -> float:
        raise NotImplementedError

    def __call__(self) -> float:
        return self.sample()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Deterministic(Distribution):
    """Always returns the same constant value."""

    def __init__(self, value: float):
        self.value = value

    def sample(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Deterministic({self.value})"


class Exponential(Distribution):
    """Exponential distribution with given *mean*."""

    def __init__(self, mean: float):
        self.mean = mean

    def sample(self) -> float:
        return random.expovariate(1.0 / self.mean)

    def __repr__(self) -> str:
        return f"Exponential(mean={self.mean})"


class Erlang(Distribution):
    """Erlang-*k* distribution with given *mean*.

    An Erlang-k(mean) is the sum of k independent Exp(mean/k) variables.
    """

    def __init__(self, k: int, mean: float):
        self.k = k
        self.mean = mean
        self._rate = k / mean  # rate of each exponential phase

    def sample(self) -> float:
        return sum(random.expovariate(self._rate) for _ in range(self.k))

    def __repr__(self) -> str:
        return f"Erlang(k={self.k}, mean={self.mean})"


class Uniform(Distribution):
    """Continuous uniform distribution on [*low*, *high*]."""

    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.low = low
        self.high = high

    def sample(self) -> float:
        return random.uniform(self.low, self.high)

    def __repr__(self) -> str:
        return f"Uniform({self.low}, {self.high})"


class Normal(Distribution):
    """Normal (Gaussian) distribution with given *mean* and *std*."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def sample(self) -> float:
        return random.gauss(self.mean, self.std)

    def __repr__(self) -> str:
        return f"Normal(mean={self.mean}, std={self.std})"


class Sequence(Distribution):
    """Deterministic sequence driven by a function of *n*.

    ``func(n)`` is called with *n* = 0, 1, 2, … on successive samples.
    Call ``reset()`` to restart from *n* = 0.
    """

    def __init__(self, func: Callable[[int], float]):
        self.func = func
        self.n: int = 0

    def sample(self) -> float:
        value = self.func(self.n)
        self.n += 1
        return value

    def reset(self) -> None:
        self.n = 0

    def __repr__(self) -> str:
        return f"Sequence(n={self.n})"
