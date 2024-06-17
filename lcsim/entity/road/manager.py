from typing import Dict, List, Optional, Tuple

import numpy as np

from ..gen.map import map_pb2
from .road import Road


class RoadManager:
    """
    Manager for all roads in a map.
    """

    roads: Dict[int, Road]  # road_id -> Road

    def __init__(self, pbs: List[map_pb2.Road], lane_manager):
        self.roads = {}
        # pbs: map_pb2.Map.roads
        for pb in pbs:
            self.roads[pb.id] = Road(pb, lane_manager)

    def init_polyline_emb(self, motion_diffuser):
        for road in self.roads.values():
            road.init_polyline_emb(motion_diffuser)

    def get_road_by_id(self, road_id: int) -> Optional[Road]:
        return self.roads.get(road_id)
