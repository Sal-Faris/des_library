import math
import random
from dataclasses import dataclass

from core import Event, Simulation
from statistics import SampleStatistic, TimeWeightedStatistic, Counter
from distributions import Exponential, Distribution


class Uniform(Distribution):

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample(self) -> float:
        return random.uniform(self.low, self.high)

    def __repr__(self) -> str:
        return f"Uniform(low={self.low}, high={self.high})"


# time is measured in hours
# time 0 = Monday 00:00
WEEK = 168


def day_of_week(time):
    # Monday = 0, Tuesday = 1, ..., Sunday = 6
    return int((time % WEEK) // 24)


def hour_of_day(time):
    return time % 24


def is_weekday(time):
    return day_of_week(time) <= 4


def is_office_time(time):
    h = hour_of_day(time)
    return is_weekday(time) and 8 <= h < 16


def next_capacity_change_after(time):
    # office capacity changes at 08:00 and 16:00 on weekdays
    current_day = int(math.floor(time / 24))

    for d in range(current_day, current_day + 14):
        if d % 7 <= 4:
            for h in [8, 16]:
                candidate = d * 24 + h

                if candidate > time:
                    return candidate

    return time + WEEK


def next_friday_scheduling_after(time):
    # Friday 16:00
    current_week = int(math.floor(time / WEEK))
    candidate = current_week * WEEK + 4 * 24 + 16

    if candidate <= time:
        candidate += WEEK

    return candidate


@dataclass
# for one patient
class Patient:
    id: int
    patient_type: str

    request_time: float
    arrival_time: float = None
    appointment_time: float = None

    scan_start_time: float = None
    scan_end_time: float = None

    office_request: bool = False
    same_day_deadline: float = None


# defining the state of the simulation
class CTState:

    def __init__(
        self,
        emergency_rate,
        inpatient_rate,
        outpatient_request_rate
    ):
        # queues
        self.emergency_queue = []
        self.normal_queue = []
        self.outpatient_waiting_list = []

        # scanner 0 is always available
        # scanner 1 is only available during office hours
        self.scanners = [None, None]

        # waiting room has 3 chairs
        self.num_chairs = 3

        # outpatient appointment calendar
        # key = hour start time
        # value = number of outpatients already booked in that hour
        self.appointment_counts = {}

        # arrival distributions
        self.emergency_arrival_gap = Exponential(mean=1 / emergency_rate)
        self.inpatient_arrival_gap = Exponential(mean=1 / inpatient_rate)
        self.outpatient_request_gap = Exponential(mean=1 / outpatient_request_rate)

        # scan duration in hours
        # Uniform[10, 19] minutes
        self.scan_time = Uniform(10 / 60, 19 / 60)

        # counters
        self.total_completed = Counter()

        self.total_waiting_room_arrivals = Counter()
        self.num_waited_outside = Counter()

        self.inpatient_office_requests = Counter()
        self.inpatient_missed_same_day = Counter()

        # sample statistics
        self.emergency_waiting_time = SampleStatistic()
        self.outpatient_waiting_time = SampleStatistic()
        self.outpatient_access_time = SampleStatistic()

        # time weighted statistics
        self.waiting_room_size_over_time = TimeWeightedStatistic()

        # utilization statistics
        self.last_utilisation_update_time = 0.0

        self.office_busy_time = 0.0
        self.office_available_time = 0.0

        self.outside_busy_time = 0.0
        self.outside_available_time = 0.0

    def num_busy_scanners(self):
        count = 0

        for patient in self.scanners:
            if patient is not None:
                count += 1

        return count

    def num_waiting(self):
        return len(self.emergency_queue) + len(self.normal_queue)

    def free_scanner_indices(self, current_time):
        free = []

        if is_office_time(current_time):
            # during office hours, both scanners can be used
            for i in range(len(self.scanners)):
                if self.scanners[i] is None:
                    free.append(i)

            return free

        # outside office hours, only one scanner may be used
        # if any scanner is already busy, do not start another scan
        if self.num_busy_scanners() > 0:
            return []

        # if no scanner is busy, use scanner 0
        if self.scanners[0] is None:
            return [0]

        return []

    def update_utilisation_statistics(self, current_time):
        t = self.last_utilisation_update_time

        while t < current_time:
            next_change = min(next_capacity_change_after(t), current_time)
            duration = next_change - t

            busy = self.num_busy_scanners()

            if is_office_time(t):
                # during office hours, 2 scanners are available
                self.office_busy_time += min(busy, 2) * duration
                self.office_available_time += 2 * duration
            else:
                # outside office hours, 1 scanner is available
                self.outside_busy_time += min(busy, 1) * duration
                self.outside_available_time += 1 * duration

            t = next_change

        self.last_utilisation_update_time = current_time

    def update_time_weighted_statistics(self, current_time):
        self.update_utilisation_statistics(current_time)
        self.waiting_room_size_over_time.update(current_time, self.num_waiting())

    def record_waiting_room_arrival(self):
        self.total_waiting_room_arrivals.increment()

        # if 3 patients are already waiting, this patient waits outside
        if self.num_waiting() >= self.num_chairs:
            self.num_waited_outside.increment()

    def find_earliest_outpatient_slot(self, first_day, last_day):
        # first_day and last_day are absolute day numbers
        # day 0 = Monday of week 0

        for day in range(first_day, last_day + 1):

            # only weekdays
            if day % 7 > 4:
                continue

            for hour in range(8, 16):
                hour_start = day * 24 + hour

                if hour < 12:
                    capacity = 4
                else:
                    capacity = 3

                used = self.appointment_counts.get(hour_start, 0)

                if used < capacity:
                    self.appointment_counts[hour_start] = used + 1

                    # spread appointments inside the hour
                    appointment_time = hour_start + used / capacity

                    return appointment_time

        return None

    def schedule_outpatient_this_week(self, current_time):
        current_day = int(math.floor(current_time / 24))
        current_week = current_day // 7

        first_possible_day = current_day + 1
        last_day_this_week = current_week * 7 + 4

        if first_possible_day > last_day_this_week:
            return None

        return self.find_earliest_outpatient_slot(
            first_possible_day,
            last_day_this_week
        )

    def schedule_outpatient_next_week(self, current_time):
        current_day = int(math.floor(current_time / 24))
        current_week = current_day // 7

        first_day_next_week = (current_week + 1) * 7
        last_day_next_week = first_day_next_week + 4

        return self.find_earliest_outpatient_slot(
            first_day_next_week,
            last_day_next_week
        )

    def fraction_waited_outside(self):
        if self.total_waiting_room_arrivals.value == 0:
            return 0.0

        return self.num_waited_outside.value / self.total_waiting_room_arrivals.value

    def fraction_inpatients_missed_same_day(self):
        if self.inpatient_office_requests.value == 0:
            return 0.0

        return self.inpatient_missed_same_day.value / self.inpatient_office_requests.value

    def office_utilisation(self):
        if self.office_available_time == 0:
            return 0.0

        return self.office_busy_time / self.office_available_time

    def outside_utilisation(self):
        if self.outside_available_time == 0:
            return 0.0

        return self.outside_busy_time / self.outside_available_time


def try_start_scan(sim, state):
    # keep starting scans while a scanner is free and someone is waiting

    while True:
        free_scanners = state.free_scanner_indices(sim.current_time)

        if len(free_scanners) == 0:
            return

        if len(state.emergency_queue) == 0 and len(state.normal_queue) == 0:
            return

        # update before changing queues/scanners
        state.update_time_weighted_statistics(sim.current_time)

        # emergency patients go first
        if len(state.emergency_queue) > 0:
            patient = state.emergency_queue.pop(0)
        else:
            patient = state.normal_queue.pop(0)

        scanner_index = free_scanners[0]

        patient.scan_start_time = sim.current_time
        state.scanners[scanner_index] = patient

        # calculate waiting time
        wait = patient.scan_start_time - patient.arrival_time

        if patient.patient_type == "emergency":
            state.emergency_waiting_time.record(wait)

        if patient.patient_type == "outpatient":
            state.outpatient_waiting_time.record(wait)

        if patient.patient_type == "inpatient" and patient.office_request:
            if patient.scan_start_time >= patient.same_day_deadline:
                state.inpatient_missed_same_day.increment()

        # sample scan duration
        duration = state.scan_time()

        # schedule scan finish
        sim.schedule(
            ScanFinished(
                sim.current_time + duration,
                scanner_index,
                patient,
                state
            )
        )

        # update after changing queues/scanners
        state.update_time_weighted_statistics(sim.current_time)


# defining an emergency arrival event
class EmergencyArrival(Event):

    def __init__(self, time, seq, state):
        super().__init__(time)
        self.seq = seq
        self.state = state

    def execute(self, sim):
        n = self.seq

        self.state.update_time_weighted_statistics(sim.current_time)

        patient = Patient(
            id=n,
            patient_type="emergency",
            request_time=sim.current_time,
            arrival_time=sim.current_time
        )

        self.state.record_waiting_room_arrival()
        self.state.emergency_queue.append(patient)

        try_start_scan(sim, self.state)

        self.state.update_time_weighted_statistics(sim.current_time)

        # schedule next emergency arrival
        gap = self.state.emergency_arrival_gap()
        sim.schedule(EmergencyArrival(sim.current_time + gap, n + 1, self.state))


# defining an inpatient arrival event
class InpatientArrival(Event):

    def __init__(self, time, seq, state):
        super().__init__(time)
        self.seq = seq
        self.state = state

    def execute(self, sim):
        n = self.seq

        self.state.update_time_weighted_statistics(sim.current_time)

        office_request = is_office_time(sim.current_time)

        same_day_deadline = None

        if office_request:
            current_day = int(math.floor(sim.current_time / 24))
            same_day_deadline = current_day * 24 + 16
            self.state.inpatient_office_requests.increment()

        patient = Patient(
            id=n,
            patient_type="inpatient",
            request_time=sim.current_time,
            arrival_time=sim.current_time,
            office_request=office_request,
            same_day_deadline=same_day_deadline
        )

        self.state.record_waiting_room_arrival()
        self.state.normal_queue.append(patient)

        try_start_scan(sim, self.state)

        self.state.update_time_weighted_statistics(sim.current_time)

        # schedule next inpatient arrival
        gap = self.state.inpatient_arrival_gap()
        sim.schedule(InpatientArrival(sim.current_time + gap, n + 1, self.state))


# defining an outpatient request event
class OutpatientRequest(Event):

    def __init__(self, time, seq, state):
        super().__init__(time)
        self.seq = seq
        self.state = state

    def execute(self, sim):
        n = self.seq

        patient = Patient(
            id=n,
            patient_type="outpatient",
            request_time=sim.current_time
        )

        appointment_time = self.state.schedule_outpatient_this_week(sim.current_time)

        if appointment_time is None:
            # no slot available in the same week
            # patient is not in the CT department yet
            self.state.outpatient_waiting_list.append(patient)

        else:
            patient.appointment_time = appointment_time

            # access time is measured in days
            access_time = (patient.appointment_time - patient.request_time) / 24
            self.state.outpatient_access_time.record(access_time)

            sim.schedule(
                OutpatientAppointmentArrival(
                    patient.appointment_time,
                    patient,
                    self.state
                )
            )

        # schedule next outpatient request
        gap = self.state.outpatient_request_gap()
        sim.schedule(OutpatientRequest(sim.current_time + gap, n + 1, self.state))


# defining an outpatient physical arrival event
class OutpatientAppointmentArrival(Event):

    def __init__(self, time, patient, state):
        super().__init__(time)
        self.patient = patient
        self.state = state

    def execute(self, sim):
        self.state.update_time_weighted_statistics(sim.current_time)

        self.patient.arrival_time = sim.current_time

        self.state.record_waiting_room_arrival()
        self.state.normal_queue.append(self.patient)

        try_start_scan(sim, self.state)

        self.state.update_time_weighted_statistics(sim.current_time)


# defining a scan finish event
class ScanFinished(Event):

    def __init__(self, time, scanner_index, patient, state):
        super().__init__(time)
        self.scanner_index = scanner_index
        self.patient = patient
        self.state = state

    def execute(self, sim):
        self.state.update_time_weighted_statistics(sim.current_time)

        self.patient.scan_end_time = sim.current_time

        self.state.scanners[self.scanner_index] = None
        self.state.total_completed.increment()

        try_start_scan(sim, self.state)

        self.state.update_time_weighted_statistics(sim.current_time)


# defining the Friday evening waiting-list scheduling event
class FridayScheduling(Event):

    def __init__(self, time, state):
        super().__init__(time)
        self.state = state

    def execute(self, sim):
        waiting_list = self.state.outpatient_waiting_list
        self.state.outpatient_waiting_list = []

        for patient in waiting_list:
            appointment_time = self.state.schedule_outpatient_next_week(sim.current_time)

            if appointment_time is None:
                # if next week is also full, keep patient waiting
                self.state.outpatient_waiting_list.append(patient)

            else:
                patient.appointment_time = appointment_time

                # access time is measured in days
                access_time = (patient.appointment_time - patient.request_time) / 24
                self.state.outpatient_access_time.record(access_time)

                sim.schedule(
                    OutpatientAppointmentArrival(
                        patient.appointment_time,
                        patient,
                        self.state
                    )
                )

        # schedule next Friday scheduling event
        sim.schedule(FridayScheduling(sim.current_time + WEEK, self.state))


# defining a capacity change event
class CapacityChange(Event):

    def __init__(self, time, state):
        super().__init__(time)
        self.state = state

    def execute(self, sim):
        self.state.update_time_weighted_statistics(sim.current_time)

        # at 08:00 scanner 1 opens
        # at 16:00 scanner 1 closes for new scans
        try_start_scan(sim, self.state)

        self.state.update_time_weighted_statistics(sim.current_time)

        # schedule next capacity change
        next_change = next_capacity_change_after(sim.current_time)
        sim.schedule(CapacityChange(next_change, self.state))


def run_scenario(
    emergency_rate,
    inpatient_rate,
    outpatient_request_rate,
    num_weeks
):
    sim = Simulation()

    state = CTState(
        emergency_rate=emergency_rate,
        inpatient_rate=inpatient_rate,
        outpatient_request_rate=outpatient_request_rate
    )

    # schedule first arrivals
    sim.schedule(EmergencyArrival(state.emergency_arrival_gap(), 1, state))
    sim.schedule(InpatientArrival(state.inpatient_arrival_gap(), 1, state))
    sim.schedule(OutpatientRequest(state.outpatient_request_gap(), 1, state))

    # schedule first capacity change
    sim.schedule(CapacityChange(next_capacity_change_after(0), state))

    # schedule first Friday scheduling event
    sim.schedule(FridayScheduling(next_friday_scheduling_after(0), state))

    # run simulation
    end_time = num_weeks * WEEK

    sim.run(stop_condition=lambda sim: sim.current_time >= end_time)

    T = sim.current_time
    state.update_time_weighted_statistics(T)

    print("\n===================================")
    print("Results for current CT simulation")
    print("===================================")

    print("Simulation time:", T)
    print("Completed patients:", state.total_completed.value)

    print("Office-hour scanner utilisation:", state.office_utilisation())
    print("Outside-office scanner utilisation:", state.outside_utilisation())

    print("Average emergency waiting time:", state.emergency_waiting_time.mean())
    print("Average outpatient waiting time:", state.outpatient_waiting_time.mean())
    print("Average outpatient access time in days:", state.outpatient_access_time.mean())

    print("Fraction waited outside:", state.fraction_waited_outside())

    print(
        "Fraction office-hour inpatients missed same-day target:",
        state.fraction_inpatients_missed_same_day()
    )

    print("Average number waiting:", state.waiting_room_size_over_time.mean(T))

    return state


if __name__ == "__main__":

    # example rates per hour
    # replace these with the project values

    run_scenario(
        emergency_rate=1.0,
        inpatient_rate=0.3,
        outpatient_request_rate=1.0,
        num_weeks=10
    )