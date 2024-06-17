from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sortedcontainers import SortedDict

from ..gen.map import map_pb2
from ..junction.junction import Junction
from ..road.road import Road
from ..utils.polygon import (
    generate_batch_polylines_from_map,
    get_polyline_dir,
    interp_polyline_by_fixed_waypt_interval,
)
from . import manager


class Lane:
    """
    A single lane in a road/junction.
    """

    CENTERLINE_INTERP_INTERVAL: float = 0.5
    CENTERLINE_TYPE = map_pb2.CENTER_LINE_TYPE_SURFACE_STREET

    # static features
    pb: map_pb2.Lane

    # polygon features (centerline)
    centerline: np.ndarray  # (N,2) original centerline
    centerline_interp: np.ndarray  # (M,2) centerline interpolated with fixed interval

    # embedding for deep models
    polyline_emb: np.ndarray = None
    poly_center_points: np.ndarray = None
    poly_center_point_s: List[float]  # s on lane of each center point
    roadgraph_points: dict = None

    # topological relations
    predecessors: Dict[int, "Lane"]  # predecessor lane_id -> Lane
    successors: Dict[int, "Lane"]  # successor lane_id -> Lane
    side_lanes: List[List["Lane"]]  # [[left lanes], [right lanes]] nearest first
    line_lengths: List[float]  # length of each centerline segment
    line_directions: List[float]  # direction of each centerline segment
    length: float = 0.0  # total length of centerline
    parent_road: Road = None
    offset_in_road: int = 0  # index of lane in road from left to right
    parent_junction: Junction = None
    parent_id: int = 0  # road/junction id

    # sim features
    agents: SortedDict[float, int]  # {s on centerline: agent_id}
    new_agents: SortedDict[float, int]

    def __init__(self, pb: map_pb2.Lane):
        # pb: map_pb2.Lane
        self.pb = pb
        # polygon features
        self.centerline = np.array([[p.x, p.y] for p in pb.center_line.nodes])
        self.centerline_interp, _ = interp_polyline_by_fixed_waypt_interval(
            polyline=self.centerline,
            waypt_interval=self.CENTERLINE_INTERP_INTERVAL,
        )  # (N,2)
        self.line_lengths, self.line_directions = [], []
        length = 0.0
        self.line_lengths.append(length)
        for i in range(1, len(self.centerline)):
            length += np.linalg.norm(self.centerline[i] - self.centerline[i - 1])
            self.line_lengths.append(length)
            self.line_directions.append(
                np.arctan2(
                    self.centerline[i][1] - self.centerline[i - 1][1],
                    self.centerline[i][0] - self.centerline[i - 1][0],
                )
            )
        self.length = length
        self.agents = SortedDict()
        self.new_agents = SortedDict()

    @property
    def id(self) -> int:
        return self.pb.id

    @property
    def max_speed(self) -> float:
        return self.pb.max_speed

    @property
    def width(self) -> float:
        return self.pb.width

    def init_with_manager(self, manager: "manager.LaneManager"):
        # init topological relations with manager
        self.predecessors, self.successors, self.side_lanes = {}, {}, [[], []]
        for conn in self.pb.predecessors:
            l = manager.get_lane_by_id(conn.id)
            self.predecessors[conn.id] = l
        for conn in self.pb.successors:
            l = manager.get_lane_by_id(conn.id)
            self.successors[conn.id] = l
        for lid in self.pb.left_lane_ids[::-1]:
            l = manager.get_lane_by_id(lid)
            self.side_lanes[0].append(l)
        for rid in self.pb.right_lane_ids:
            l = manager.get_lane_by_id(rid)
            self.side_lanes[1].append(l)

    def init_polyline_emb(self, motion_diffuser):
        # add zero for z
        centerline_with_z = np.concatenate(
            [self.centerline_interp, np.zeros((len(self.centerline_interp), 1))], axis=1
        )
        center_dir = get_polyline_dir(centerline_with_z)
        center_type = np.array([self.CENTERLINE_TYPE] * len(self.centerline_interp))
        polylines = np.concatenate(
            [centerline_with_z, center_dir, center_type[:, None]], axis=1
        )  # (N,7) [x, y, z, dir_x, dir_y, dir_z, type]
        poly_points, poly_points_mask, poly_center_points = (
            generate_batch_polylines_from_map(
                polylines=polylines,
                ids=np.array([0] * len(self.centerline_interp)),
                num_points_each_polyline=10 if self.in_junction() else 20,
            )
        )
        self.roadgraph_points = {
            "xyz": torch.from_numpy(centerline_with_z).to(torch.float32),
            "dir": torch.from_numpy(center_dir).to(torch.float32),
            "type": torch.from_numpy(center_type).to(torch.int32),
        }
        # compute embs
        device = next(motion_diffuser.parameters()).device
        self.polyline_emb = (
            motion_diffuser.map_encoder.forward(
                {
                    "roadgraph": {
                        "poly_points": torch.from_numpy(poly_points)
                        .to(torch.float32)
                        .to(device),
                        "poly_points_mask": torch.from_numpy(poly_points_mask)
                        .to(torch.bool)
                        .to(device),
                        "poly_center_points": torch.from_numpy(poly_center_points)
                        .to(torch.float32)
                        .to(device),
                    }
                }
            )
            .detach()
            .cpu()
            .numpy()
        )
        self.poly_center_point_s = []
        origin = self.centerline_interp[0]
        s = 0.0
        for cp in poly_center_points:
            cp_xy = cp[:2]
            s += np.linalg.norm(cp_xy - origin)
            self.poly_center_point_s.append(s)
            origin = cp_xy
        self.poly_center_points = poly_center_points

    def set_parent_road_when_init(self, road: Road, offset: int):
        self.parent_road = road
        self.offset_in_road = offset
        self.parent_id = road.pb.id

    def set_parent_junction_when_init(self, junction: Junction):
        self.parent_junction = junction
        self.parent_id = junction.pb.id

    def prepare(self):
        self.agents.clear()
        for s, agent_id in self.new_agents.items():
            self.agents[s] = agent_id
        self.new_agents.clear()

    def reset(self):
        self.agents.clear()
        self.new_agents.clear()

    def get_centerline(self) -> np.ndarray:
        return self.centerline

    def get_polyline_emb(self) -> np.ndarray:
        # (num_polylines, hidden_dim)
        return self.polyline_emb

    def in_junction(self) -> bool:
        return self.parent_junction is not None

    def in_road(self) -> bool:
        return self.parent_road is not None

    def unique_successor(self) -> Optional["Lane"]:
        assert self.in_junction()
        return list(self.successors.values())[0]

    def neighbor_lane(self, side: int) -> Optional["Lane"]:
        if len(self.side_lanes[side]) == 0:
            return None
        return self.side_lanes[side][0]

    def left_neighbor(self) -> Optional["Lane"]:
        return self.neighbor_lane(0)

    def right_neighbor(self) -> Optional["Lane"]:
        return self.neighbor_lane(1)

    def add_agent(self, agent_id: int, s: float):
        self.new_agents[s] = agent_id

    def get_next_agent_by_s(self, s: float) -> Optional[Tuple[float, int]]:
        """
        Get next agent by s, return None if not found.
        """
        index = self.agents.bisect_right(s)
        if index == len(self.agents):
            return None
        return self.agents.peekitem(index)

    def get_pre_agent_by_s(self, s: float) -> Optional[Tuple[float, int]]:
        """
        Get previous agent by s, return None if not found.
        """
        index = self.agents.bisect_left(s)
        if index == 0:
            return None
        return self.agents.peekitem(index - 1)

    def get_first_agent(self) -> Optional[Tuple[float, int]]:
        """
        Get first agent on lane(with smallest s), return None if no agent.
        """
        if len(self.agents) == 0:
            return None
        return self.agents.peekitem(0)

    def get_agents_by_range(self, start_s: float, end_s: float) -> List[int]:
        """
        Get agents in range [start_s, end_s].
        """
        start_s, end_s = max(0.0, start_s), min(self.length, end_s)
        assert start_s <= end_s
        s_idx, e_idx = self.agents.bisect_left(start_s), self.agents.bisect_right(end_s)
        return self.agents.values()[s_idx:e_idx]

    def get_agent_ids(self) -> List[int]:
        return list(self.agents.values())

    def get_polyline_feature_by_range(
        self, start_s: float, end_s: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get polyline feature in range [start_s, end_s] -> (emb, center_points).
        """
        start_s, end_s = max(0.0, start_s), min(self.poly_center_point_s[-1], end_s)
        assert start_s <= end_s
        s_idx, e_idx = 0, 0
        for i, s in enumerate(self.poly_center_point_s):
            if s >= start_s:
                s_idx = i
                break
        for i, s in enumerate(self.poly_center_point_s):
            if s >= end_s:
                e_idx = i + 1
                break
        return self.polyline_emb[s_idx:e_idx], self.poly_center_points[s_idx:e_idx]

    def get_position_by_s(self, s: float) -> np.ndarray:
        """
        Get position on lane by s.
        """
        s = max(0.0, min(s, self.length))
        for i in range(1, len(self.line_lengths)):
            if s <= self.line_lengths[i]:
                break
        p1, p2 = self.centerline[i - 1], self.centerline[i]
        ratio = (s - self.line_lengths[i - 1]) / (
            self.line_lengths[i] - self.line_lengths[i - 1]
        )
        return p1 + ratio * (p2 - p1)

    def get_direction_by_s(self, s: float) -> float:
        """
        Get direction on lane by s.
        """
        s = max(0.0, min(s, self.length))
        for i in range(1, len(self.line_lengths)):
            if s <= self.line_lengths[i]:
                break
        return self.line_directions[i - 1]

    def project_from_lane(self, other: "Lane", other_s: float) -> float:
        """
        Project the s on the lane to the current lane.
        """
        if self.parent_road != other.parent_road:
            raise ValueError(f"lanes {self.id} and {other.id} are not on the same road")
        s = other_s / other.length * self.length
        return max(0.0, min(s, self.length))
