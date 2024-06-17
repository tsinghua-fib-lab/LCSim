from typing import List, Optional

from ..gen.agent import trip_pb2
from ..gen.geo import geo_pb2
from ..gen.routing import routing_pb2
from ..lane import lane
from ..road import road


class Route:
    """
    Route for a trip.
    """

    class JunctionCandidate:
        lanes: List[lane.Lane]  # list of candidate lanes in junction
        pre_lanes: List[lane.Lane]  # list of lanes into junction

        def __init__(self, lanes: List[lane.Lane], pre_lanes: List[lane.Lane]):
            self.lanes = lanes
            self.pre_lanes = pre_lanes

    class LC:
        """
        Route info for lane change.
        """

        in_candidate: bool  # if current lane is in candidate
        # true, no need to change lane
        neighbors_cnt: List[int] = [
            0,
            0,
        ]  # cnt of neighbor lanes in candidate (left, right)
        # false, need to change lane
        side: int = 0  # 0: left, 1: right
        count: int = 0  # count of lane change
        force_lc_length: float = 0.0  # length of lane change

    LC_FACTOR = 3  # lane change length factor(s)
    MIN_FORCE_LC_LENGTH = 10.0  # min length for force lane change
    MAX_FORCE_LC_LENGTH = 50.0  # max length for force lane change

    base: List[routing_pb2.Journey]  # list of routes
    start: geo_pb2.Position
    start_lane: lane.Lane
    end: geo_pb2.Position
    end_lane: lane.Lane
    end_s: float
    valid: bool  # if the route is valid

    roads: List[road.Road]  # list of roads
    junc_lane_groups: List[JunctionCandidate]  # list of junction candidates
    at_road: bool  # if the current route position is on road

    def __init__(
        self,
        pbs: List[routing_pb2.Journey],
        start: geo_pb2.Position,
        end: geo_pb2.Position,
        engine,
    ):
        # pbs: routing_pb2.Trip.routes
        self.base = pbs
        if len(self.base) == 0:
            self.valid = False
            return
        self.start = start
        self.start_lane = engine.lane_manager.get_lane_by_id(
            start.lane_position.lane_id
        )
        self.end = end
        self.end_lane = engine.lane_manager.get_lane_by_id(end.lane_position.lane_id)
        self.end_s = end.lane_position.s
        assert len(self.base) > 0
        # process routes into roads and junctions
        road_ids = self.base[0].driving.road_ids
        self.roads = [
            engine.road_manager.get_road_by_id(road_id) for road_id in road_ids
        ]
        assert (
            engine.lane_manager.get_lane_by_id(end.lane_position.lane_id).parent_road
            == self.roads[-1]
        )
        self.junc_lane_groups = []
        for i in range(len(self.roads) - 1):
            in_road, out_road = self.roads[i], self.roads[i + 1]
            in_lane: lane.Lane = engine.lane_manager.get_lane_by_id(in_road.lane_ids[0])
            junc = list(in_lane.successors.values())[0].parent_junction
            assert junc is not None
            lg = junc.driving_lane_group(in_road.id, out_road.id)
            assert lg is not None
            lanes = [engine.lane_manager.get_lane_by_id(lid) for lid in lg.lane_ids]
            pre_lanes = []
            for l in lanes:
                pl = list(l.predecessors.values())[0]
                assert pl.parent_road == in_road
                pre_lanes.append(pl)
            self.junc_lane_groups.append(self.JunctionCandidate(lanes, pre_lanes))
        self.at_road = True
        self.valid = True

    def next(
        self, cur_lane: lane.Lane, cur_s: float, cur_v: float
    ) -> Optional[lane.Lane]:
        """
        Get next lane of the route, return None if reach the end.
        """
        if self.at_road:
            # on road, lane change or into junction
            if len(self.junc_lane_groups) == 0:
                return None
            lc = self.get_lc(cur_lane, cur_s, cur_v)
            group = self.junc_lane_groups[0]
            if lc.in_candidate:
                # in candidate, no need to change lane
                next_lane = self.get_junc_lane_by_pre_lane(cur_lane, 0)
            else:
                if lc.side == 0:
                    # left lane change, nearest is the right lane in group
                    next_lane = group.lanes[-1]
                else:
                    # right lane change, nearest is the left lane in group
                    next_lane = group.lanes[0]
            self.roads.pop(0)
        else:
            # in junction, out lane is unique
            next_lane = cur_lane.unique_successor()
            self.junc_lane_groups.pop(0)
        self.at_road = not self.at_road
        return next_lane

    def get_lc(self, cur_lane: lane.Lane, cur_s: float, cur_v: float) -> LC:
        """
        Get lane change info.
        """
        lc = self.LC()
        assert self.at_road  # no lane change in junction
        road = self.roads[0]
        assert cur_lane.parent_road == road
        cur_lane_offset = cur_lane.offset_in_road
        if len(self.junc_lane_groups) == 0:
            # reach the end
            delta = self.end_lane.offset_in_road - cur_lane_offset
            if delta == 0:
                lc.in_candidate = True
                lc.neighbors_cnt = [0, 0]
            elif delta < 0:
                lc.in_candidate = False
                lc.side = 0
                lc.count = -delta
            else:
                lc.in_candidate = False
                lc.side = 1
                lc.count = delta
            return lc
        # look into the next junction
        jlg = self.junc_lane_groups[0]
        offset_l, offset_r = (
            jlg.pre_lanes[0].offset_in_road,
            jlg.pre_lanes[-1].offset_in_road,
        )
        delta_l, delta_r = offset_l - cur_lane_offset, offset_r - cur_lane_offset
        if delta_l > 0 or delta_r < 0:
            # need lane change
            lc.in_candidate = False
            pre_lanes = jlg.pre_lanes
            min_delta = 1e6
            cnt = 0
            for pl in pre_lanes:
                delta = pl.offset_in_road - cur_lane_offset
                if abs(delta) < abs(min_delta):
                    min_delta = delta
                    cnt = abs(delta)
            assert cnt > 0
            lc.count = cnt
            lc.side = 0 if min_delta < 0 else 1
            force_lc_length = cur_lane.max_speed * self.LC_FACTOR
            lc.force_lc_length = min(
                max(force_lc_length, self.MIN_FORCE_LC_LENGTH),
                self.MAX_FORCE_LC_LENGTH,
            )
            return lc
        # no need to change lane
        lc.in_candidate = True
        lc.neighbors_cnt = [-delta_l, delta_r]
        return lc

    def get_junc_lane_by_pre_lane(
        self, pre_lane: lane.Lane, junc_index: int
    ) -> Optional[lane.Lane]:
        """
        Get junction lane by pre lane.
        """
        if junc_index >= len(self.junc_lane_groups):
            return None
        jlg = self.junc_lane_groups[junc_index]
        min_delta = 1e6
        for i, pl in enumerate(jlg.pre_lanes):
            delta = abs(pre_lane.offset_in_road - pl.offset_in_road)
            if delta < min_delta:
                min_delta = delta
                index = i
        return jlg.lanes[index]
