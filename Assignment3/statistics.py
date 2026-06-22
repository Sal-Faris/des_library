"""
Statistics collection utilities for DES.

* TimeWeightedStatistic — tracks a time-weighted average of a value
  that changes at discrete points (e.g. queue length).
* SampleStatistic — tracks running statistics over a stream of samples
  (e.g. waiting times), with confidence-interval support.
* Counter — simple event counter with rate computation.
"""
import math
from typing import Tuple


# ---------------------------------------------------------------------------
# t-distribution critical value (avoids scipy dependency)
# ---------------------------------------------------------------------------

def _t_critical(confidence: float, df: int) -> float:
    """Approximate the two-tailed t critical value.

    Uses the Abramowitz-Stegun normal approximation refined with
    a Cornish-Fisher expansion for small degrees of freedom.
    """
    p = 1 - (1 - confidence) / 2  # upper tail probability

    # Normal quantile via rational approximation (A&S 26.2.23)
    t = 1 - p if p > 0.5 else p
    w = math.sqrt(-2 * math.log(t))
    z = w - (2.515517 + 0.802853 * w + 0.010328 * w ** 2) / (
        1 + 1.432788 * w + 0.189269 * w ** 2 + 0.001308 * w ** 3
    )
    if p < 0.5:
        z = -z

    if df >= 120:
        return z

    # Cornish-Fisher refinement
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


# ---------------------------------------------------------------------------
# TimeWeightedStatistic
# ---------------------------------------------------------------------------

class TimeWeightedStatistic:
    """Accumulates the time-weighted integral of a piecewise-constant signal.

    Call ``update(time, new_value)`` every time the tracked quantity changes.
    The integral of *old_value × (time − last_time)* is added automatically.

    ``mean(current_time)`` returns the integral divided by total elapsed time.
    """

    def __init__(self, initial_value: float = 0.0, start_time: float = 0.0):
        self._accumulated: float = 0.0
        self._last_time: float = start_time
        self._last_value: float = initial_value

    def update(self, current_time: float, new_value: float) -> None:
        self._accumulated += self._last_value * (current_time - self._last_time)
        self._last_time = current_time
        self._last_value = new_value

    def mean(self, current_time: float) -> float:
        if current_time <= 0:
            return 0.0
        total = self._accumulated + self._last_value * (current_time - self._last_time)
        return total / current_time

    def accumulated(self, current_time: float) -> float:
        return self._accumulated + self._last_value * (current_time - self._last_time)

    def reset(self, time: float = 0.0, value: float = 0.0) -> None:
        self._accumulated = 0.0
        self._last_time = time
        self._last_value = value


# ---------------------------------------------------------------------------
# SampleStatistic
# ---------------------------------------------------------------------------

class SampleStatistic:
    """Running statistics over a stream of scalar samples.

    Records count, sum, sum-of-squares, min, max and derives mean,
    sample variance, standard deviation, and confidence intervals.
    """

    def __init__(self):
        self._n: int = 0
        self._sum: float = 0.0
        self._sum_sq: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")

    def record(self, value: float) -> None:
        self._n += 1
        self._sum += value
        self._sum_sq += value * value
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

    @property
    def count(self) -> int:
        return self._n

    @property
    def total(self) -> float:
        return self._sum

    def mean(self) -> float:
        return self._sum / self._n if self._n > 0 else 0.0

    def variance(self) -> float:
        if self._n < 2:
            return 0.0
        return (self._sum_sq - self._sum ** 2 / self._n) / (self._n - 1)

    def std(self) -> float:
        return math.sqrt(self.variance())

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Return (lower, upper) bounds of a *confidence*-level CI."""
        if self._n < 2:
            return (float("-inf"), float("inf"))
        m = self.mean()
        se = math.sqrt(self.variance() / self._n)
        z = _t_critical(confidence, self._n - 1)
        return (m - z * se, m + z * se)

    @property
    def minimum(self) -> float:
        return self._min if self._n > 0 else 0.0

    @property
    def maximum(self) -> float:
        return self._max if self._n > 0 else 0.0

    def reset(self) -> None:
        self._n = 0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = float("inf")
        self._max = float("-inf")


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------

class Counter:
    """Simple counter with rate computation."""

    def __init__(self):
        self._count: int = 0

    def increment(self, n: int = 1) -> None:
        self._count += n

    @property
    def value(self) -> int:
        return self._count

    def rate(self, elapsed_time: float) -> float:
        return self._count / elapsed_time if elapsed_time > 0 else 0.0

    def fraction(self, total: int) -> float:
        return self._count / total if total > 0 else 0.0

    def reset(self) -> None:
        self._count = 0
