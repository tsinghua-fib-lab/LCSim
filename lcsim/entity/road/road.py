from typing import List, Optional, Tuple

import matplotlib
import matplotlib.patheffects as pe
import numpy as np
import torch

from ..gen.map import map_pb2
from ..utils import plot
from ..utils.polygon import (
    generate_batch_polylines_from_map,
    get_polyline_dir,
    interp_polyline_by_fixed_waypt_interval,
)


class Road:
    """
    A single road in a map.
    """

    POLYLINE_INTERP_INTERVAL: float = 0.5
    ROAD_LINE_TYPE_PREFIX = 5
    ROAD_EDGE_TYPE_PREFIX = 14
    # static features
    pb: map_pb2.Road
    length: float = 0.0
    # polygon features (polygon points, type)
    road_lines: List[Tuple[np.ndarray, int]]
    road_lines_interp: List[Tuple[np.ndarray, int]]
    road_edges: List[Tuple[np.ndarray, int]]
    road_edges_interp: List[Tuple[np.ndarray, int]]
    road_connections: List[Tuple[np.ndarray, int]]
    road_connections_interp: List[Tuple[np.ndarray, int]]
    # embedding for deep models
    polyline_emb: np.ndarray = None
    poly_center_points: np.ndarray = None
    poly_center_point_s: List[float]
    roadgraph_points: dict = None
    # sim features
    lane_manager = None

    def __init__(self, pb: map_pb2.Road, lane_manager):
        # pb: map_pb2.Road
        self.pb = pb
        # polygon features
        self.road_lines = [
            (np.array([[p.x, p.y] for p in line.nodes]), line.type)  # ((N,2),int)
            for line in pb.road_lines
        ]
        self.road_lines_interp = [
            (
                interp_polyline_by_fixed_waypt_interval(
                    polyline=np.array([[p.x, p.y] for p in line.nodes]),
                    waypt_interval=0.5,
                )[0],
                line.type,
            )  # ((N,2),int)
            for line in pb.road_lines
        ]
        self.road_edges = [
            (np.array([[p.x, p.y] for p in edge.nodes]), edge.type)  # ((N,2),int)
            for edge in pb.road_edges
        ]
        self.road_edges_interp = [
            (
                interp_polyline_by_fixed_waypt_interval(
                    polyline=np.array([[p.x, p.y] for p in edge.nodes]),
                    waypt_interval=0.5,
                )[0],
                edge.type,
            )  # ((N,2),int)
            for edge in pb.road_edges
        ]
        self.road_connections = [
            (np.array([[p.x, p.y] for p in conn.nodes]), conn.type)  # ((N,2),int)
            for conn in pb.road_connections
        ]
        self.road_connections_interp = [
            (
                interp_polyline_by_fixed_waypt_interval(
                    polyline=np.array([[p.x, p.y] for p in conn.nodes]),
                    waypt_interval=0.5,
                )[0],
                conn.type,
            )  # ((N,2),int)
            for conn in pb.road_connections
        ]
        # sim features
        for i, lid in enumerate(pb.lane_ids):
            lane = lane_manager.get_lane_by_id(lid)
            lane.set_parent_road_when_init(self, i)
            self.length += lane.length
        self.length /= len(pb.lane_ids)
        self.lane_manager = lane_manager

    @property
    def id(self) -> int:
        return self.pb.id

    @property
    def lane_ids(self) -> List[int]:
        return self.pb.lane_ids

    def init_polyline_emb(self, motion_diffuser):
        self.poly_center_point_s = []
        poly_points_all, poly_points_mask_all, poly_center_points_all = [], [], []
        xyzs, dirs, types, ids = [], [], [], []
        _id = 0
        for rl in self.road_lines_interp:
            rl_with_z = np.concatenate([rl[0], np.zeros((len(rl[0]), 1))], axis=1)
            rl_dir = get_polyline_dir(rl_with_z)
            rl_type = np.array([rl[1] + self.ROAD_LINE_TYPE_PREFIX] * len(rl[0]))
            xyzs.append(rl_with_z)
            dirs.append(rl_dir)
            types.append(rl_type)
            ids.append([_id] * len(rl[0]))
            _id += 1
            polyline = np.concatenate([rl_with_z, rl_dir, rl_type[:, None]], axis=1)
            poly_points, poly_points_mask, poly_center_points = (
                generate_batch_polylines_from_map(
                    polylines=polyline, ids=[0] * len(rl[0])
                )
            )
            origin = rl[0][0]
            s = 0.0
            for cp in poly_center_points:
                cp_xy = cp[:2]
                s += np.linalg.norm(cp_xy - origin)
                self.poly_center_point_s.append(s)
                origin = cp_xy
            poly_points_all.append(poly_points)
            poly_points_mask_all.append(poly_points_mask)
            poly_center_points_all.append(poly_center_points)
        for re in self.road_edges_interp:
            re_with_z = np.concatenate([re[0], np.zeros((len(re[0]), 1))], axis=1)
            re_dir = get_polyline_dir(re_with_z)
            re_type = np.array([re[1] + self.ROAD_EDGE_TYPE_PREFIX] * len(re[0]))
            xyzs.append(re_with_z)
            dirs.append(re_dir)
            types.append(re_type)
            ids.append([_id] * len(re[0]))
            _id += 1
            polyline = np.concatenate([re_with_z, re_dir, re_type[:, None]], axis=1)
            poly_points, poly_points_mask, poly_center_points = (
                generate_batch_polylines_from_map(
                    polylines=polyline, ids=[0] * len(re[0])
                )
            )
            origin = re[0][0]
            s = 0.0
            for cp in poly_center_points:
                cp_xy = cp[:2]
                s += np.linalg.norm(cp_xy - origin)
                self.poly_center_point_s.append(s)
                origin = cp_xy
            poly_points_all.append(poly_points)
            poly_points_mask_all.append(poly_points_mask)
            poly_center_points_all.append(poly_center_points)
        for conn in self.road_connections_interp:
            conn_with_z = np.concatenate([conn[0], np.zeros((len(conn[0]), 1))], axis=1)
            conn_dir = get_polyline_dir(conn_with_z)
            conn_type = np.array([conn[1] + self.ROAD_LINE_TYPE_PREFIX] * len(conn[0]))
            xyzs.append(conn_with_z)
            dirs.append(conn_dir)
            types.append(conn_type)
            ids.append([_id] * len(conn[0]))
            _id += 1
            polyline = np.concatenate(
                [conn_with_z, conn_dir, conn_type[:, None]], axis=1
            )
            poly_points, poly_points_mask, poly_center_points = (
                generate_batch_polylines_from_map(
                    polylines=polyline, ids=[0] * len(conn[0])
                )
            )
            origin = conn[0][0]
            s = 0.0
            for cp in poly_center_points:
                cp_xy = cp[:2]
                s += np.linalg.norm(cp_xy - origin)
                self.poly_center_point_s.append(s)
                origin = cp_xy
            poly_points_all.append(poly_points)
            poly_points_mask_all.append(poly_points_mask)
            poly_center_points_all.append(poly_center_points)
        poly_points = np.concatenate(poly_points_all, axis=0)
        poly_points_mask = np.concatenate(poly_points_mask_all, axis=0)
        poly_center_points = np.concatenate(poly_center_points_all, axis=0)
        xyzs = np.concatenate(xyzs, axis=0)
        dirs = np.concatenate(dirs, axis=0)
        types = np.concatenate(types, axis=0)
        ids = np.concatenate(ids, axis=0)
        self.roadgraph_points = {
            "xyz": torch.from_numpy(xyzs).to(torch.float32),
            "dir": torch.from_numpy(dirs).to(torch.float32),
            "type": torch.from_numpy(types).to(torch.int32),
            "ids": torch.from_numpy(ids).to(torch.int32),
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
        self.poly_center_points = poly_center_points
        # sort by s
        sort_idx = np.argsort(self.poly_center_point_s)
        self.polyline_emb = self.polyline_emb[sort_idx]
        self.poly_center_points = self.poly_center_points[sort_idx]
        self.poly_center_point_s = np.array(self.poly_center_point_s)[sort_idx]

    def get_polyline_feature_by_range(
        self, start_s: float, end_s: float
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get polyline feature in range [start_s, end_s] -> (emb, center_points).
        """
        emb, center_points = [], []
        # road
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
        emb.append(self.polyline_emb[s_idx:e_idx])
        center_points.append(self.poly_center_points[s_idx:e_idx])
        # lanes
        for lid in self.lane_ids:
            lane = self.lane_manager.get_lane_by_id(lid)
            assert lane is not None
            lane_emb, lane_center_points = lane.get_polyline_feature_by_range(
                start_s, end_s
            )
            emb.append(lane_emb)
            center_points.append(lane_center_points)
        return np.concatenate(emb, axis=0), np.concatenate(center_points, axis=0)

    def get_agent_ids(self) -> List[int]:
        """
        Get agent ids in road.
        """
        agent_ids = []
        for lane_id in self.lane_ids:
            agent_ids.extend(self.lane_manager.get_lane_by_id(lane_id).get_agent_ids())
        return agent_ids

    def get_agents_by_range(self, start_s: float, end_s: float) -> List[int]:
        """
        Get agent in range [start_s, end_s].
        """
        start_s, end_s = max(0.0, start_s), min(self.length, end_s)
        assert start_s <= end_s
        agent_ids = []
        for lane_id in self.lane_ids:
            lane = self.lane_manager.get_lane_by_id(lane_id)
            assert lane is not None
            agent_ids.extend(lane.get_agents_by_range(start_s, end_s))
        return agent_ids

    def get_pre_junction_id(self) -> Optional[int]:
        """
        Get pre junction id.
        """
        lane = self.lane_manager.get_lane_by_id(self.lane_ids[0])
        if not lane.predecessors:
            return None
        pre = list(lane.predecessors.values())[0]
        assert pre.in_junction()
        return pre.parent_id

    def get_succ_junction_id(self) -> Optional[int]:
        """
        Get succ junction id.
        """
        lane = self.lane_manager.get_lane_by_id(self.lane_ids[-1])
        if not lane.successors:
            return None
        succ = list(lane.successors.values())[0]
        assert succ.in_junction()
        return succ.parent_id

    def plot_polylines(self, ax: matplotlib.axes.Axes, range_xy: List[int] = None):
        """
        Plot polylines on ax.
        """
        for line in self.road_lines:
            if (line[1] + self.ROAD_LINE_TYPE_PREFIX) in plot.BROKEN_LINE_TYPE:
                plot.plot_dashed_line(
                    ax,
                    polyline=line[0],
                    dashes=[2, 4],
                    color=plot.get_color_from_type(
                        line[1] + self.ROAD_LINE_TYPE_PREFIX
                    ),
                    range_xy=range_xy,
                )
                continue
            ax.plot(
                line[0][:, 0],
                line[0][:, 1],
                color=plot.get_color_from_type(line[1] + self.ROAD_LINE_TYPE_PREFIX),
                ms=2,
            )
        for edge in self.road_edges:
            ax.plot(
                edge[0][:, 0],
                edge[0][:, 1],
                color=plot.get_color_from_type(edge[1] + self.ROAD_EDGE_TYPE_PREFIX),
                ms=2,
            )
        for conn in self.road_connections:
            ax.plot(
                conn[0][:, 0],
                conn[0][:, 1],
                color=plot.get_color_from_type(conn[1] + self.ROAD_LINE_TYPE_PREFIX),
                ms=2,
            )

    def check_offroad(self, pos: np.ndarray) -> bool:
        """check if pos is off road

        Args:
            pos (np.ndarray): (2,)

        Returns:
            bool: True if off road
        """
        edge_points = []
        edge_dir = []
        for edge in self.road_edges:
            edge_points.extend(edge[0])
            edge_dir.extend(get_polyline_dir(edge[0]))
        edge_points = np.array(edge_points)
        edge_dir = np.array(edge_dir)
        nearest_idx = np.argmin(np.linalg.norm(edge_points - pos, axis=1))
        pos2pt = edge_points[nearest_idx] - pos
        pt_dir = edge_dir[nearest_idx]
        cross = np.cross(pos2pt, pt_dir)
        return cross < 0
