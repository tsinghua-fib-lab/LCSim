import math
from typing import Dict, List, Tuple

from ..gen.agent import trip_pb2


class Schedule:
    """
    Agent's trip schedule.
    """

    base: List[trip_pb2.Schedule] = []  # list of schedules
    schedule_index: int = 0  # index of current schedule
    trip_index: int = 0  # index of current trip of current schedule
    loop_count: int = 0  # loop counter of current schedule
    last_trip_end_time: float = 0  # end time of last trip

    def __init__(self, pbs: List[trip_pb2.Schedule]):
        # pbs: trip_pb2.Agent.schedules
        self.base = pbs

    def next_trip(self, time: float) -> bool:
        """
        Switch to next trip, return False if no more trips.
        """
        if len(self.base) == 0:
            return False
        schedule = self.base[self.schedule_index]
        self.last_trip_end_time = time
        self.trip_index += 1
        if self.trip_index >= len(schedule.trips):
            self.trip_index = 0
            self.loop_count += 1
            if schedule.loop_count > 0 and self.loop_count >= schedule.loop_count:
                self.loop_count = 0
                self.schedule_index += 1
                if self.schedule_index >= len(self.base):
                    self.schedule_index = 0
                    self.base = []
                    return False
        return True

    def get_trip(self) -> trip_pb2.Trip:
        """
        Get current trip.
        """
        return self.base[self.schedule_index].trips[self.trip_index]

    def get_departure_time(self) -> float:
        """
        Get departure time of current trip.
        """
        if len(self.base) == 0:
            # return infinite if no schedule
            return math.inf
        trip = self.get_trip()
        schedule = self.base[self.schedule_index]
        if trip.HasField("departure_time"):
            return trip.departure_time
        if schedule.HasField("departure_time"):
            return schedule.departure_time
        if trip.HasField("wait_time"):
            return self.last_trip_end_time + trip.wait_time
        return self.last_trip_end_time
