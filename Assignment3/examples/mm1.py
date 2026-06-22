"""
M/M/1 example built on des_library.

Structure:
- `MM1Model` owns parameters, simulation state, and statistics collectors.
- `Arrival` and `EndService` events encode all system dynamics.
- `run()` sets the initial event and delegates progression to the DES engine.

Why this design:
- Keeps model state centralized and explicit, so event logic stays easy to reason about.
- Separates concerns between domain behavior (events) and infrastructure (simulation engine).
- Makes the example reusable as a template for queueing models with different policies.
"""

from __future__ import annotations

import random
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event, TimeWeightedStatistic, SampleStatistic


class MM1Model:
    def __init__(self, arrival_rate: float = 0.9, service_rate: float = 1.0,
                 end_time: float = 1000.0, seed: int = 42):
        random.seed(seed)
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.end_time = end_time

        self.sim = Simulation()
        self.num_customers = 0
        self.queue_length = TimeWeightedStatistic()
        self.waiting_time = SampleStatistic()

    def start_service(self, now: float) -> None:
        service_duration = random.expovariate(self.service_rate)
        self.sim.schedule(EndService(now + service_duration, self))

    def run(self) -> None:
        self.sim.schedule(Arrival(0.0, self))
        self.sim.run(stop_condition=lambda sim: sim.current_time > self.end_time)
        self.queue_length.update(self.sim.current_time, self.num_customers)

    def report(self) -> None:
        t = self.sim.current_time
        print("M/M/1 Example")
        print(f"  Horizon time: {t:.2f}")
        print(f"  Avg number in system: {self.queue_length.mean(t):.4f}")
        print(f"  Avg waiting time (approx, arrivals seeing system): {self.waiting_time.mean():.4f}")


class Arrival(Event):
    def __init__(self, time: float, model: MM1Model):
        super().__init__(time)
        self.model = model

    def execute(self, sim: Simulation) -> None:
        m = self.model
        m.queue_length.update(self.time, m.num_customers)
        # This sample tracks queueing delay proxy when customer arrives.
        m.waiting_time.record(m.num_customers / m.service_rate if m.service_rate > 0 else 0.0)
        m.num_customers += 1
        m.queue_length.update(self.time, m.num_customers)

        if m.num_customers == 1:
            m.start_service(self.time)

        sim.schedule(Arrival(self.time + random.expovariate(m.arrival_rate), m))


class EndService(Event):
    def __init__(self, time: float, model: MM1Model):
        super().__init__(time)
        self.model = model

    def execute(self, sim: Simulation) -> None:
        m = self.model
        m.queue_length.update(self.time, m.num_customers)
        m.num_customers -= 1
        m.queue_length.update(self.time, m.num_customers)
        if m.num_customers > 0:
            m.start_service(self.time)


if __name__ == "__main__":
    model = MM1Model()
    model.run()
    model.report()
