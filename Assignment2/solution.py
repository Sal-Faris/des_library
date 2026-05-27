import math
import random
from dataclasses import dataclass

from core import Event, Simulation
from statistics import SampleStatistic, TimeWeightedStatistic, Counter, _t_critical
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
    def __init__(self, arrival_rate):
        self.pending = []

        # logs for batch means and regenerative method
        self.confirmation_log = []
        self.regen_indices = []

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
        self.arrival_rate = arrival_rate
        self.arrival_gap = Exponential(mean=1 / arrival_rate)
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
        super().__init__(time)  # when the arrival happens
        self.seq = seq  # transaction number
        self.state = state  # shared mempool state

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
            self.state.confirmation_log.append(wait)

        # possible regeneration point: a block leaves the mempool empty
        if len(pending) == 0 and len(included) > 0:
            self.state.regen_indices.append(len(self.state.confirmation_log))

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


# simple run for preliminary comparison
def run_scenario(arrival_rate):
    sim = Simulation()
    state = MempoolState(arrival_rate)

    # schedule first events
    sim.schedule(TxArrival(0, 1, state))
    sim.schedule(BlockFound(state.block_gap(), state))

    # run the simulation
    sim.run(stop_condition=lambda sim: state.num_confirmed >= 2000)

    T = sim.current_time

    print("\n===================================")
    print(f"Results for lambda = {arrival_rate} tx/s")
    print("===================================")

    print("Simulation time:", T)
    print("Total submitted:", state.total_submitted.value)
    print("Total confirmed:", state.num_confirmed)
    print("Total expired:", state.num_expired.value)

    print("Average confirmation time:", state.time_to_confirm.mean())
    print("Average mempool size:", state.mempool_size_over_time.mean(T))
    print("Average ineligible fraction:", state.ineligible_fraction_over_time.mean(T))
    print("Average block gas utilisation:", state.block_gas_utilisation.mean())
    print("Expiry rate:", state.num_expired.rate(T))
    print("Final base fee:", state.base_fee)


# one replication used for warm-up, batch means, and regenerative method
def run_replication(observations, arrival_rate, D=0):
    sim = Simulation()
    state = MempoolState(arrival_rate)

    sim.schedule(TxArrival(0.0, 1, state))
    sim.schedule(BlockFound(state.block_gap(), state))

    t0 = 0.0

    # warm-up phase
    if D > 0:
        sim.run(stop_condition=lambda s: len(state.confirmation_log) >= D)
        t0 = sim.current_time

        state.confirmation_log.clear()
        state.regen_indices.clear()
        state.time_to_confirm.reset()
        state.block_gas_utilisation.reset()
        state.num_expired.reset()

        state.mempool_size_over_time.reset(
            time=sim.current_time,
            value=len(state.pending)
        )

        state.ineligible_fraction_over_time.reset(
            time=sim.current_time,
            value=state.ineligible_fraction()
        )

    # main measurement phase
    sim.run(stop_condition=lambda s: len(state.confirmation_log) >= observations)

    T = sim.current_time - t0

    # after warm-up, use accumulated / elapsed time, not mean(sim.current_time)
    mempool_mean = state.mempool_size_over_time.accumulated(sim.current_time) / T if T > 0 else 0.0
    ineligible_mean = state.ineligible_fraction_over_time.accumulated(sim.current_time) / T if T > 0 else 0.0

    return {
        "confirmation_log": state.confirmation_log[:observations],
        "mempool_size_mean": mempool_mean,
        "regen_indices": state.regen_indices,
        "ineligible_mean": ineligible_mean,
        "block_util_mean": state.block_gas_utilisation.mean(),
        "expiry_rate": state.num_expired.rate(T) if T > 0 else 0.0,
        "final_base_fee": state.base_fee,
        "base_fee_times": state.base_fee_times,
        "base_fee_values": state.base_fee_values,
    }


def find_warmup_length(
    num_replications=20,
    observations=10000,
    threshold=0.05,
    min_D=100,
    arrival_rate=8.0,
):
    runs = []

    for n in range(num_replications):
        res = run_replication(
            observations=observations,
            arrival_rate=arrival_rate,
            D=0,
        )

        runs.append(res["confirmation_log"])

    w_avg = []

    for k in range(observations):
        total = 0.0

        for n in range(num_replications):
            total += runs[n][k]

        w_avg.append(total / num_replications)

    max_D = observations // 2

    for D in range(min_D, max_D):
        m1 = sum(w_avg[:D]) / D
        m2 = sum(w_avg[:2 * D]) / (2 * D)

        if m1 == 0:
            continue

        if abs((m2 / m1) - 1.0) <= threshold:
            return D

    return min_D


def batch_means(obs, num_batches=30, confidence=0.95):
    n = len(obs)
    b = n // num_batches

    if b < 1:
        raise ValueError(f"Too few observations ({n}) for {num_batches} batches.")

    batch_avg = []

    for k in range(num_batches):
        start = k * b
        end = start + b
        batch = obs[start:end]

        batch_avg.append(sum(batch) / b)

    stat = SampleStatistic()

    for x in batch_avg:
        stat.record(x)

    est = stat.mean()
    low, high = stat.confidence_interval(confidence=confidence)
    hw = (high - low) / 2

    return {
        "point_estimate": est,
        "ci_lower": low,
        "ci_upper": high,
        "half_width": hw,
        "rel_precision": hw / est if est > 0 else float("inf"),
        "batch_size": b,
        "num_batches": num_batches,
    }


def regenerative_estimate(confirmation_log, regen_indices, confidence=0.95):
    if len(regen_indices) < 3:
        return None

    W = []
    M = []

    for k in range(len(regen_indices) - 1):
        start = regen_indices[k]
        end = regen_indices[k + 1]

        cycle = confirmation_log[start:end]

        if len(cycle) == 0:
            continue

        W.append(sum(cycle))
        M.append(len(cycle))

    n = len(W)

    if n < 2:
        return None

    avg_W = sum(W) / n
    avg_M = sum(M) / n

    est = avg_W / avg_M

    V = []

    for k in range(n):
        V.append(W[k] - est * M[k])

    stat = SampleStatistic()

    for x in V:
        stat.record(x)

    var = stat.variance()
    t = _t_critical(confidence, n - 1)

    hw = t * math.sqrt(var / n) / avg_M

    return {
        "point_estimate": est,
        "ci_lower": est - hw,
        "ci_upper": est + hw,
        "half_width": hw,
        "rel_precision": hw / est if est > 0 else float("inf"),
        "num_cycles": n,
    }


def run_steady_state_analysis(arrival_rate):
    print("\n===================================")
    print(f"Steady-state analysis for lambda = {arrival_rate} tx/s")
    print("===================================")

    print("\nEstimating warm-up length (Welch procedure)...")

    D = find_warmup_length(
        num_replications=20,
        observations=4000,
        arrival_rate=arrival_rate,
    )

    print(f"Warm-up length D = {D}")

    r = 45
    b = 4 * D
    observations = b * r

    print("\nRunning main simulation:")
    print(f"  warm-up D     = {D}")
    print(f"  batch size b  = {b}  (= 4D)")
    print(f"  num batches r = {r}")
    print(f"  observations  = {observations}  (= b * r)")

    res = run_replication(
        observations=observations,
        arrival_rate=arrival_rate,
        D=D,
    )

    bm = batch_means(res["confirmation_log"], num_batches=r)

    print("\n--- Confirmation time (batch means) ---")
    print(f"  Point estimate:     {bm['point_estimate']:.3f} s")
    print(f"  95% CI:             [{bm['ci_lower']:.3f}, {bm['ci_upper']:.3f}]")
    print(f"  Half-width:         {bm['half_width']:.3f}")
    print(f"  Relative precision: {bm['rel_precision']:.3f}")

    if bm["rel_precision"] <= 0.10:
        print("  Precision target met (<= 0.10).")
    else:
        print("  Precision target NOT met. Increase observations and re-run.")

    print("\n--- Other metrics (run-averaged) ---")
    print(f"  Mempool size mean:     {res['mempool_size_mean']:.3f}")
    print(f"  Ineligible fraction:   {res['ineligible_mean']:.3f}")
    print(f"  Block gas utilisation: {res['block_util_mean']:.4f}")
    print(f"  Expiry rate:           {res['expiry_rate']:.4f} tx/s")
    print(f"  Final base fee:        {res['final_base_fee']:.3f} Gwei")

    rm = regenerative_estimate(res["confirmation_log"], res["regen_indices"])

    print("\n--- Confirmation time (regenerative) ---")

    if rm is None:
        print(f"  Too few regeneration points ({len(res['regen_indices'])}).")
        print("  Method not applicable; use batch means instead.")
    else:
        print(f"  Point estimate:     {rm['point_estimate']:.3f} s")
        print(f"  95% CI:             [{rm['ci_lower']:.3f}, {rm['ci_upper']:.3f}]")
        print(f"  Half-width:         {rm['half_width']:.3f}")
        print(f"  Relative precision: {rm['rel_precision']:.3f}")
        print(f"  Number of cycles:   {rm['num_cycles']}")


if __name__ == "__main__":
    # quick preliminary comparison
    run_scenario(8)
    run_scenario(12)

    # longer steady-state analysis
    run_steady_state_analysis(8)
    run_steady_state_analysis(12)