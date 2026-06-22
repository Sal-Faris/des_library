"""
des_library — A reusable Discrete Event Simulation toolkit.

Core classes
    Simulation          Event-driven simulation engine (no global state).
    Event               Base class for all events.
    StopSimulation      Event that terminates the run.

Statistics
    TimeWeightedStatistic   Time-weighted average (e.g. queue length).
    SampleStatistic         Running mean / variance / CI over samples.
    Counter                 Simple counter with rate helpers.

Distributions
    Deterministic, Exponential, Erlang, Uniform, Normal, Sequence
"""

from .core import Simulation, Event, StopSimulation
from .statistics import TimeWeightedStatistic, SampleStatistic, Counter
from .distributions import (
    Distribution,
    Deterministic,
    Exponential,
    Erlang,
    Uniform,
    Normal,
    Sequence,
)

__all__ = [
    "Simulation",
    "Event",
    "StopSimulation",
    "TimeWeightedStatistic",
    "SampleStatistic",
    "Counter",
    "Distribution",
    "Deterministic",
    "Exponential",
    "Erlang",
    "Uniform",
    "Normal",
    "Sequence",
]
