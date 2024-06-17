from typing import Dict, List, Optional, Tuple

import numpy as np

from ..gen.map import map_pb2
from . import lane


class LaneManager:
    """
    Manager for all lanes in a map.
    """

    lanes: Dict[int, "lane.Lane"]  # lane_id -> Lane

    def __init__(self, pbs: List[map_pb2.Lane]):
        self.lanes = {}
        # pbs: map_pb2.Map.lanes
        for pb in pbs:
            self.lanes[pb.id] = lane.Lane(pb)
        for l in self.lanes.values():
            l.init_with_manager(self)

    def init_polyline_emb(self, motion_diffuser):
        for lane in self.lanes.values():
            lane.init_polyline_emb(motion_diffuser)

    def get_lane_by_id(self, lane_id: int) -> Optional["lane.Lane"]:
        return self.lanes.get(lane_id)

    def prepare(self):
        for l in self.lanes.values():
            l.prepare()

    def reset(self):
        for l in self.lanes.values():
            l.reset()
