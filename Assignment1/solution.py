import math
from dataclasses import dataclass

from core import Event, Simulation
from statistics import TimeWeightedStatistic, SampleStatistic, Counter


@dataclass
# for one transaction
class PendingTx:
    id: int
    submitted_at: float
    byte_size: float
    fee_sats: float
    sat_per_byte: float
    is_confirmed: bool = False
    was_bumped: bool = False

# defining the state fo the simulation
class MempoolState:
    def __init__(self):
        self.pending = []
        self.num_confirmed = 0
        self.time_to_confirm = SampleStatistic()
        self.time_to_confirm_bumped = SampleStatistic()
        self.time_to_confirm_normal = SampleStatistic()
        self.pending_size_over_time = TimeWeightedStatistic()
        self.block_fill_rate = SampleStatistic()
        # self.total_submitted = Counter()

# defining a transaction arrival event
class TxArrival(Event):
    def __init__(self, time, seq, state):
        super().__init__(time) # when the arrival happens
        self.seq = seq # transaction number
        self.state = state # shared mempool state

    def execute(self, sim):
        n = self.seq
        # given in the description
        byte_size = 200 + 100 * abs(math.cos(n * math.pi / 7))
        fee_sats = 500 + 300 * abs(math.sin(n * math.e / 2))
        # create the transaction object
        tx = PendingTx(n, sim.current_time, byte_size, fee_sats, fee_sats / byte_size)
        # add it to the mempool, now its waiting to be mined
        self.state.pending.append(tx)
        # count the arrival and update the time weighted mempool size
        self.state.total_submitted.increment()
        self.state.pending_size_over_time.update(sim.current_time, len(self.state.pending))
        # schedule the next arrival (gap given in description)
        gap = 8 * (2 + math.sin(n * math.pi / 5))
        sim.schedule(TxArrival(sim.current_time + gap, n + 1, self.state))
        sim.schedule(FeeBump(sim.current_time + 180, tx, self.state))

# defining a block mining event
class BlockFound(Event):
    MAX_BLOCK_BYTES = 1_000_000

    def __init__(self, time, state):
        super().__init__(time)
        self.state = state

    def execute(self, sim):
        pending = self.state.pending
        # need to choose the transactions with the highest fee rate first (descending sort)
        pending.sort(key=lambda tx: (-tx.sat_per_byte, tx.id))
        # fill in the block greedily
        bytes_used = 0
        included = []
        for tx in pending:
            if bytes_used + tx.byte_size > self.MAX_BLOCK_BYTES:
                break
            bytes_used += tx.byte_size
            included.append(tx)
        # process all the transactions that are included
        for tx in included:
            tx.is_confirmed = True
            pending.remove(tx)
            self.state.num_confirmed += 1
            wait = sim.current_time - tx.submitted_at
            self.state.time_to_confirm.record(wait)
            if tx.was_bumped:
                self.state.time_to_confirm_bumped.record(wait)
            else:
                self.state.time_to_confirm_normal.record(wait)
        # statistics
        self.state.block_fill_rate.record(bytes_used / self.MAX_BLOCK_BYTES)
        self.state.pending_size_over_time.update(sim.current_time, len(pending))
        # produce a block every 600 seconds (given)
        sim.schedule(BlockFound(sim.current_time + 600, self.state))

# define replace by fee event. 
class FeeBump(Event):
    def __init__(self, time, tx, state):
        super().__init__(time)
        self.tx = tx
        self.state = state

    def execute(self, sim):
        # Only runs if necessary
        if self.tx.is_confirmed or self.tx.was_bumped:
            return
        self.tx.fee_sats *= 1.5
        self.tx.sat_per_byte = self.tx.fee_sats / self.tx.byte_size
        self.tx.was_bumped = True


if __name__ == "__main__":
    sim = Simulation()
    state = MempoolState()
    sim.schedule(TxArrival(0, 1, state))
    sim.schedule(BlockFound(600, state))
    sim.run(stop_condition=lambda sim: state.num_confirmed >= 2000)
    T = sim.current_time

    print("Average confirmation time (all):")
    print(state.time_to_confirm.mean())
    print("\nAverage confirmation time (fee-bumped):")
    print(state.time_to_confirm_bumped.mean())
    print("\nAverage confirmation time (normal):")
    print(state.time_to_confirm_normal.mean())
    print("\nAverage mempool depth:")
    print(state.pending_size_over_time.mean(T))
    print("\nAverage block fill rate:")
    print(state.block_fill_rate.mean())