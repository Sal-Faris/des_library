"""
M/G/1 Processor Sharing example built on des_library.

Structure:
- `MG1PSModel` owns the queue, timing state, event handles, and statistics.
- `Customer` stores per-job remaining service and arrival timestamp.
- `Arrival` and `EndService` events maintain processor-sharing dynamics by
  updating all remaining times and rescheduling completions when needed.

Why this design:
- Encapsulates all mutable state in one model object, which avoids hidden coupling.
- Keeps event handlers focused on state transitions tied to simulation time.
- Demonstrates cancellation/rescheduling patterns needed in preemptive DES models.
"""

from __future__ import annotations

import bisect
import random
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event, TimeWeightedStatistic, SampleStatistic


class Customer:
    def __init__(self, service_time: float, arrival_time: float):
        self.remaining = service_time
        self.arrival_time = arrival_time

    def decrease(self, amount: float) -> None:
        self.remaining -= amount


class MG1PSModel:
    def __init__(self, arrival_rate: float = 0.9, service_rate: float = 1.0,
                 end_time: float = 1000.0, seed: int = 42):
        random.seed(seed)
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.end_time = end_time

        self.sim = Simulation()
        self.queue: list[Customer] = []
        self.last_update_time = 0.0
        self.next_end_service: EndService | None = None

        self.queue_length = TimeWeightedStatistic()
        self.sojourn_time = SampleStatistic()

    def insert_customer(self, cust: Customer) -> None:
        keys = [c.remaining for c in self.queue]
        idx = bisect.bisect_right(keys, cust.remaining)
        self.queue.insert(idx, cust)

    def update_all_remaining_times(self, now: float) -> bool:
        n = len(self.queue)
        if n > 0:
            delta = (now - self.last_update_time) / n
            for c in self.queue:
                c.decrease(delta)
        self.last_update_time = now
        return n > 0

    def start_service(self, now: float) -> None:
        self.next_end_service = EndService(now + self.queue[0].remaining * len(self.queue), self)
        self.sim.schedule(self.next_end_service)

    def run(self) -> None:
        self.sim.schedule(Arrival(0.0, self))
        self.sim.run(stop_condition=lambda sim: sim.current_time > self.end_time)
        self.queue_length.update(self.sim.current_time, len(self.queue))

    def report(self) -> None:
        t = self.sim.current_time
        print("M/G/1-PS Example")
        print(f"  Horizon time: {t:.2f}")
        print(f"  Avg queue length: {self.queue_length.mean(t):.4f}")
        print(f"  Avg sojourn time (completed): {self.sojourn_time.mean():.4f}")


class Arrival(Event):
    def __init__(self, time: float, model: MG1PSModel):
        super().__init__(time)
        self.model = model

    def execute(self, sim: Simulation) -> None:
        m = self.model
        m.queue_length.update(self.time, len(m.queue))
        if m.update_all_remaining_times(self.time) and m.next_end_service is not None:
            sim.cancel(m.next_end_service)

        service_time = random.expovariate(m.service_rate)
        m.insert_customer(Customer(service_time, self.time))
        m.queue_length.update(self.time, len(m.queue))
        m.start_service(self.time)
        sim.schedule(Arrival(self.time + random.expovariate(m.arrival_rate), m))


class EndService(Event):
    def __init__(self, time: float, model: MG1PSModel):
        super().__init__(time)
        self.model = model

    def execute(self, sim: Simulation) -> None:
        m = self.model
        if self.cancelled:
            return
        m.queue_length.update(self.time, len(m.queue))
        m.update_all_remaining_times(self.time)
        done = m.queue.pop(0)
        m.sojourn_time.record(self.time - done.arrival_time)
        m.queue_length.update(self.time, len(m.queue))
        if m.queue:
            m.start_service(self.time)


if __name__ == "__main__":
    model = MG1PSModel()
    model.run()
    model.report()
