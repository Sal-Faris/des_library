import math, random
from dataclasses import dataclass

from core import Event, Simulation
from statistics import SampleStatistic, TimeWeightedStatistic, Counter
from distributions import Exponential, Distribution


class LogNormal(Distribution):

    def __init__(self, mean: float, sigma: float):
        self.mean = mean
        self.sigma = sigma

    def sample(self) -> float:
        return random.lognormvariate(self.mean, self.sigma)

    def __repr__(self) -> str:
        return f"LogNormal(mean={self.mean}, sigma={self.sigma})"


@dataclass
# for one transaction
class PendingTx:
    id: int
    submitted_at: float
    gas_demand: float
    max_fee: float
    tip: float
    expiry_event: Event = None
    is_confirmed: bool = False
    is_expired: bool = False


# defining the state of the simulation
class MempoolState:
    def __init__(self):
        self.pending = []

        # base fee starts at 10 Gwei
        self.base_fee = 10.0

        # gas constants given in the description
        self.gas_target = 15_000_000
        self.gas_limit = 30_000_000
        self.base_fee_floor = 1.0

        # distribution parameters for the comparison experiment
        self.gas_demand_dist = LogNormal(mean=10.69, sigma=0.5)
        self.max_fee_dist = LogNormal(mean=4.56, sigma=0.3)
        self.tip_dist = Exponential(mean=2.0)

        # arrival and block distributions
        self.arrival_gap = Exponential(mean=1 / 8)
        self.block_gap = Exponential(mean=12)

        # statistics
        self.num_confirmed = 0
        self.num_expired = Counter()
        self.total_submitted = Counter()

        self.time_to_confirm = SampleStatistic()
        self.mempool_size_over_time = TimeWeightedStatistic()
        self.ineligible_fraction_over_time = TimeWeightedStatistic()
        self.block_gas_utilisation = SampleStatistic()

        # base fee time series
        self.base_fee_times = []
        self.base_fee_values = []

    def ineligible_fraction(self):
        if len(self.pending) == 0:
            return 0.0

        num_ineligible = 0
        for tx in self.pending:
            if tx.max_fee < self.base_fee:
                num_ineligible += 1

        return num_ineligible / len(self.pending)

    def update_time_weighted_statistics(self, current_time):
        self.mempool_size_over_time.update(current_time, len(self.pending))
        self.ineligible_fraction_over_time.update(current_time, self.ineligible_fraction())


# defining a transaction arrival event
class TxArrival(Event):
    def __init__(self, time, seq, state):
        super().__init__(time) # when the arrival happens
        self.seq = seq # transaction number
        self.state = state # shared mempool state

    def execute(self, sim):
        n = self.seq

        # update the time weighted statistics before the mempool changes
        self.state.update_time_weighted_statistics(sim.current_time)

        # sample transaction properties given in the description
        gas_demand = self.state.gas_demand_dist()
        max_fee = self.state.max_fee_dist()
        tip = self.state.tip_dist()

        # create the transaction object
        tx = PendingTx(n, sim.current_time, gas_demand, max_fee, tip)

        # add it to the mempool, now it is waiting to be included in a block
        self.state.pending.append(tx)
        self.state.total_submitted.increment()

        # schedule expiry after 300 seconds
        expiry_event = TxExpiry(sim.current_time + 300, tx, self.state)
        tx.expiry_event = sim.schedule(expiry_event)

        # update the time weighted statistics after the mempool changes
        self.state.update_time_weighted_statistics(sim.current_time)

        # schedule the next arrival
        gap = self.state.arrival_gap()
        sim.schedule(TxArrival(sim.current_time + gap, n + 1, self.state))


# defining a block production event
class BlockFound(Event):
    def __init__(self, time, state):
        super().__init__(time)
        self.state = state

    def execute(self, sim):
        pending = self.state.pending

        # update the time weighted statistics before the mempool and base fee change
        self.state.update_time_weighted_statistics(sim.current_time)

        # only transactions with max fee at least the current base fee are eligible
        eligible = []
        for tx in pending:
            if tx.max_fee >= self.state.base_fee:
                eligible.append(tx)

        # choose eligible transactions with the highest tip first
        eligible.sort(key=lambda tx: (-tx.tip, tx.id))

        # fill the block greedily under the gas limit
        gas_used = 0
        included = []

        for tx in eligible:
            if gas_used + tx.gas_demand > self.state.gas_limit:
                continue

            gas_used += tx.gas_demand
            included.append(tx)

        # process all the transactions that are included
        for tx in included:
            tx.is_confirmed = True

            if tx.expiry_event is not None:
                sim.cancel(tx.expiry_event)

            pending.remove(tx)
            self.state.num_confirmed += 1

            wait = sim.current_time - tx.submitted_at
            self.state.time_to_confirm.record(wait)

        # record gas utilisation for this block
        self.state.block_gas_utilisation.record(gas_used / self.state.gas_limit)

        # update the base fee according to the EIP-1559 rule
        multiplier = 1 + (1 / 8) * ((gas_used - self.state.gas_target) / self.state.gas_target)
        self.state.base_fee = self.state.base_fee * multiplier

        # enforce the minimum base fee
        self.state.base_fee = max(self.state.base_fee, self.state.base_fee_floor)

        # record the base fee that applies to the next block
        self.state.base_fee_times.append(sim.current_time)
        self.state.base_fee_values.append(self.state.base_fee)

        # update the time weighted statistics after the mempool and base fee change
        self.state.update_time_weighted_statistics(sim.current_time)

        # schedule the next block
        gap = self.state.block_gap()
        sim.schedule(BlockFound(sim.current_time + gap, self.state))


# defining a transaction expiry event
class TxExpiry(Event):
    def __init__(self, time, tx, state):
        super().__init__(time)
        self.tx = tx
        self.state = state

    def execute(self, sim):
        # only expire the transaction if it has not already been confirmed
        if self.tx.is_confirmed or self.tx.is_expired:
            return

        # update the time weighted statistics before the mempool changes
        self.state.update_time_weighted_statistics(sim.current_time)

        self.tx.is_expired = True

        if self.tx in self.state.pending:
            self.state.pending.remove(self.tx)

        self.state.num_expired.increment()

        # update the time weighted statistics after the mempool changes
        self.state.update_time_weighted_statistics(sim.current_time)


if __name__ == "__main__":
    sim = Simulation()
    state = MempoolState()

    sim.schedule(TxArrival(0, 1, state))
    sim.schedule(BlockFound(12, state))

    sim.run(stop_condition=lambda sim: state.num_confirmed >= 2000)

    T = sim.current_time

    print("Average confirmation time:")
    print(state.time_to_confirm.mean())

    print("\nAverage mempool size:")
    print(state.mempool_size_over_time.mean(T))

    print("\nAverage ineligible fraction:")
    print(state.ineligible_fraction_over_time.mean(T))

    print("\nAverage block gas utilisation:")
    print(state.block_gas_utilisation.mean())

    print("\nExpiry rate:")
    print(state.num_expired.rate(T))

    print("\nFinal base fee:")
    print(state.base_fee)