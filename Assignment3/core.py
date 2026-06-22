"""
Discrete Event Simulation Engine.

Provides a fully encapsulated simulation engine with no global state.
All simulation state lives inside a Simulation instance; events receive
the simulation reference when executed and can schedule new events,
read the clock, or stop the run.
"""
import heapq
from typing import Callable, List, Optional


class Event:
    """Base class for simulation events.

    Subclass and override ``execute(sim)`` to define behaviour.
    Events are ordered by (time, sequence_number) so that ties
    are broken in FIFO order.
    """

    def __init__(self, time: float):
        self.time: float = time
        self.cancelled: bool = False
        self._seq: int = 0  # set by Simulation.schedule()

    def execute(self, sim: "Simulation") -> None:
        raise NotImplementedError

    def cancel(self) -> None:
        self.cancelled = True

    @property
    def active(self) -> bool:
        return not self.cancelled

    def __lt__(self, other: "Event") -> bool:
        if self.time != other.time:
            return self.time < other.time
        return self._seq < other._seq

    def __repr__(self) -> str:
        tag = "" if self.active else " [X]"
        return f"{self.__class__.__name__}(t={self.time:.4f}){tag}"


class StopSimulation(Event):
    """Immediately stops the simulation when executed."""

    def execute(self, sim: "Simulation") -> None:
        sim.stop()


class Simulation:
    """Discrete Event Simulation engine.

    Usage::

        sim = Simulation()
        sim.schedule(MyEvent(0.0))
        sim.run()

    The engine pops events from a heap, skips cancelled ones, calls
    optional hooks, then invokes ``event.execute(sim)``.
    """

    def __init__(self):
        self._event_list: List[Event] = []
        self._event_counter: int = 0
        self.current_time: float = 0.0
        self.previous_time: float = 0.0
        self._running: bool = False
        self._before_hooks: List[Callable] = []
        self._after_hooks: List[Callable] = []

    # -- scheduling ----------------------------------------------------------

    def schedule(self, event: Event) -> Event:
        """Insert *event* into the event list and return it."""
        self._event_counter += 1
        event._seq = self._event_counter
        heapq.heappush(self._event_list, event)
        return event

    def cancel(self, event: Event) -> None:
        """Mark *event* as cancelled; it will be skipped during the run."""
        event.cancel()

    # -- hooks ---------------------------------------------------------------

    def on_before_event(self, hook: Callable) -> None:
        """Register a callback ``hook(sim, event)`` called before each event."""
        self._before_hooks.append(hook)

    def on_after_event(self, hook: Callable) -> None:
        """Register a callback ``hook(sim, event)`` called after each event."""
        self._after_hooks.append(hook)

    # -- execution -----------------------------------------------------------

    def run(self, stop_condition: Optional[Callable[["Simulation"], bool]] = None) -> None:
        """Run the simulation until the event list is empty or stopped.

        Parameters
        ----------
        stop_condition : callable, optional
            ``stop_condition(sim)`` is checked after every event; the run
            ends when it returns ``True``.
        """
        self._running = True
        while self._running and self._event_list:
            event = heapq.heappop(self._event_list)
            if event.cancelled:
                continue
            self.previous_time = self.current_time
            self.current_time = event.time
            for hook in self._before_hooks:
                hook(self, event)
            event.execute(self)
            for hook in self._after_hooks:
                hook(self, event)
            if stop_condition is not None and stop_condition(self):
                self._running = False

    def stop(self) -> None:
        """Signal the engine to stop after the current event."""
        self._running = False

    # -- reset / inspection --------------------------------------------------

    def reset(self) -> None:
        """Clear the event list and reset the clock."""
        self._event_list.clear()
        self._event_counter = 0
        self.current_time = 0.0
        self.previous_time = 0.0
        self._running = False

    @property
    def pending_event_count(self) -> int:
        return len(self._event_list)

    def peek_next_time(self) -> float:
        """Return the time of the next non-cancelled event (inf if none)."""
        for ev in self._event_list:
            if not ev.cancelled:
                return ev.time
        return float("inf")
