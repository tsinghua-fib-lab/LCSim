from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ..gen.map import map_pb2
from ..utils import plot
from ..utils.polygon import (
    generate_batch_polylines_from_map,
    get_polyline_dir,
    interp_polyline_by_fixed_waypt_interval,
)


class Junction:
    """
    A single junction in a map.
    """

    class LaneGroup:
        in_angle: float
        out_angle: float
        lane_ids: List[int]

        def __init__(
            self, in_angle: float = 0, out_angle: float = 0, lane_ids: List[int] = []
        ):
            self.in_angle = in_angle
            self.out_angle = out_angle
            self.lane_ids = lane_ids

    POLYLINE_INTERP_INTERVAL: float = 0.5
    ROAD_LINE_TYPE_PREFIX = 5
    ROAD_EDGE_TYPE_PREFIX = 14
    ROAD_GRAPH_DATA_RANGE = 60  # meters
    # static features
    pb: map_pb2.Junction
    # polygon features (polygon points, type)
    road_lines: List[Tuple[np.ndarray, int]]
    road_edges: List[Tuple[np.ndarray, int]]
    center: np.ndarray
    # embedding for deep models
    polyline_emb: np.ndarray = None
    poly_center_points: np.ndarray = None
    roadgraph_data: dict = None
    roadgraph_points: dict = None
    # sim features
    driving_lane_groups: Dict[Tuple[int, int], LaneGroup]
    in_roads: List[int]
    out_roads: List[int]
    agent_ids: List[int]
    lane_manager = None

    def __init__(self, pb: map_pb2.Junction, lane_manager, engine):
        # pb: map_pb2.Junction
        self.pb = pb
        # polygon features
        self.road_lines = [
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
            (
                interp_polyline_by_fixed_waypt_interval(
                    polyline=np.array([[p.x, p.y] for p in edge.nodes]),
                    waypt_interval=0.5,
                )[0],
                edge.type,
            )  # ((N,2),int)
            for edge in pb.road_edges
        ]
        center = np.array([0.0, 0.0])  # compute average of all road edge points
        minx, maxx, miny, maxy = 1e10, -1e10, 1e10, -1e10
        for edge in self.road_edges:
            center += np.mean(edge[0], axis=0)
            minx = min(minx, np.min(edge[0][:, 0]))
            maxx = max(maxx, np.max(edge[0][:, 0]))
            miny = min(miny, np.min(edge[0][:, 1]))
            maxy = max(maxy, np.max(edge[0][:, 1]))
        center[0] = (minx + maxx) / 2
        center[1] = (miny + maxy) / 2
        self.center = center
        # sim features
        self.driving_lane_groups = {}
        irs = set()
        ors = set()
        for lg in pb.driving_lane_groups:
            self.driving_lane_groups[(lg.in_road_id, lg.out_road_id)] = self.LaneGroup(
                in_angle=lg.in_angle, out_angle=lg.out_angle, lane_ids=lg.lane_ids
            )
            irs.add(lg.in_road_id)
            ors.add(lg.out_road_id)
        self.in_roads = list(irs)
        self.out_roads = list(ors)
        for lid in pb.lane_ids:
            lane = lane_manager.get_lane_by_id(lid)
            lane.set_parent_junction_when_init(self)
        self.agent_ids = []
        self.lane_manager = lane_manager
        self.engine = engine

    @property
    def id(self) -> int:
        return self.pb.id

    @property
    def lane_ids(self) -> List[int]:
        return self.pb.lane_ids

    def init_polyline_emb(self, motion_diffuser):
        polylines = []
        ids = []
        xyzs, dirs, types, ids = [], [], [], []
        _id = 0
        for rl in self.road_lines:
            rl_with_z = np.concatenate([rl[0], np.zeros((len(rl[0]), 1))], axis=1)
            rl_dir = get_polyline_dir(rl_with_z)
            rl_type = np.array([rl[1]] * len(rl[0]))
            xyzs.append(rl_with_z)
            dirs.append(rl_dir)
            types.append(rl_type)
            polylines.append(
                np.concatenate([rl_with_z, rl_dir, rl_type[:, None]], axis=1)
            )
            ids.extend([_id] * len(rl[0]))
            _id += 1
        for re in self.road_edges:
            re_with_z = np.concatenate([re[0], np.zeros((len(re[0]), 1))], axis=1)
            re_dir = get_polyline_dir(re_with_z)
            re_type = np.array([re[1] + self.ROAD_EDGE_TYPE_PREFIX] * len(re[0]))
            xyzs.append(re_with_z)
            dirs.append(re_dir)
            types.append(re_type)
            polylines.append(
                np.concatenate([re_with_z, re_dir, re_type[:, None]], axis=1)
            )
            ids.extend([_id] * len(re[0]))
            _id += 1
        polylines = np.concatenate(polylines, axis=0)
        ids = np.array(ids)
        poly_points, poly_points_mask, poly_center_points = (
            generate_batch_polylines_from_map(
                polylines=polylines, ids=ids, num_points_each_polyline=10
            )
        )
        xyzs = np.concatenate(xyzs, axis=0)
        dirs = np.concatenate(dirs, axis=0)
        types = np.concatenate(types, axis=0)
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

    def init_roadgraph_data(self):
        """
        Init roadgraph data dict for deep models
        """
        self.roadgraph_data = {}
        poly_center_points = []
        poly_emb = []
        xyzs, dirs, types, ids = [], [], [], []
        _id = 0
        # lanes in junction
        for lid in self.lane_ids:
            lane = self.lane_manager.get_lane_by_id(lid)
            assert lane is not None
            poly_center_points.append(lane.poly_center_points)
            poly_emb.append(lane.polyline_emb)
            xyzs.append(lane.roadgraph_points["xyz"])
            dirs.append(lane.roadgraph_points["dir"])
            types.append(lane.roadgraph_points["type"])
            ids.append(torch.tensor([_id] * len(lane.roadgraph_points["xyz"])))
            _id += 1
        # surrounding roads
        for irs in self.in_roads:
            r = self.engine.road_manager.get_road_by_id(irs)
            assert r is not None
            emb, pts = r.get_polyline_feature_by_range(
                r.length - self.ROAD_GRAPH_DATA_RANGE, r.length
            )
            poly_center_points.append(pts)
            poly_emb.append(emb)
            xyzs.append(r.roadgraph_points["xyz"])
            dirs.append(r.roadgraph_points["dir"])
            types.append(r.roadgraph_points["type"])
            ids.append(torch.tensor([_id] * len(r.roadgraph_points["xyz"])))
            _id = ids[-1][-1] + 1
        for ors in self.out_roads:
            r = self.engine.road_manager.get_road_by_id(ors)
            assert r is not None
            emb, pts = r.get_polyline_feature_by_range(0, self.ROAD_GRAPH_DATA_RANGE)
            poly_center_points.append(pts)
            poly_emb.append(emb)
            xyzs.append(r.roadgraph_points["xyz"])
            dirs.append(r.roadgraph_points["dir"])
            types.append(r.roadgraph_points["type"])
            ids.append(torch.tensor([_id] * len(r.roadgraph_points["xyz"])))
            _id = ids[-1][-1] + 1
        self.roadgraph_data["poly_center_points"] = torch.from_numpy(
            np.concatenate(poly_center_points)
        ).to(torch.float32)
        self.roadgraph_data["poly_emb"] = torch.from_numpy(np.concatenate(poly_emb)).to(
            torch.float32
        )
        self.roadgraph_data["num_nodes"] = len(
            self.roadgraph_data["poly_center_points"]
        )
        xyzs.append(self.roadgraph_points["xyz"])
        dirs.append(self.roadgraph_points["dir"])
        types.append(self.roadgraph_points["type"])
        ids.append(self.roadgraph_points["ids"] + _id)
        xyzs = torch.cat(xyzs, dim=0)
        dirs = torch.cat(dirs, dim=0)
        types = torch.cat(types, dim=0)
        ids = torch.cat(ids, dim=0)
        self.roadgraph_points = {
            "xyz": xyzs,
            "dir": dirs,
            "type": types,
            "ids": ids,
            "num_nodes": len(xyzs),
        }

    def driving_lane_group(
        self, in_road_id: int, out_road_id: int
    ) -> Optional[LaneGroup]:
        return self.driving_lane_groups.get((in_road_id, out_road_id), None)

    def get_surrounding_roads(self) -> List[int]:
        return self.in_roads + self.out_roads

    def get_agent_ids(self) -> List[int]:
        if len(self.agent_ids) == 0:
            agent_ids = []
            for lane_id in self.lane_ids:
                agent_ids.extend(
                    self.lane_manager.get_lane_by_id(lane_id).get_agent_ids()
                )
            return agent_ids
        else:
            self_ids = self.agent_ids
            active_ids = [a.id for a in self.engine.agent_manager.active_agents]
            return list(set(self_ids) & set(active_ids))

    def add_agent_id(self, agent_id: int):
        self.agent_ids.append(agent_id)

    def get_hetero_data(self) -> Optional[Tuple[HeteroData, List[int]]]:
        """
        HeteroData for deep models, return None if no agent in range.
        """
        # collect agent data in range around junction
        agent_ids = self.get_agent_ids()
        if len(self.agent_ids) == 0:
            # surrounding roads
            for irs in self.in_roads:
                r = self.engine.road_manager.get_road_by_id(irs)
                assert r is not None
                agent_ids.extend(
                    r.get_agents_by_range(
                        r.length - self.ROAD_GRAPH_DATA_RANGE, r.length
                    )
                )
            for ors in self.out_roads:
                r = self.engine.road_manager.get_road_by_id(ors)
                assert r is not None
                agent_ids.extend(r.get_agents_by_range(0, self.ROAD_GRAPH_DATA_RANGE))
        agent_ids = list(set(agent_ids))
        if len(agent_ids) == 0:
            return None
        # agent data
        xyz, vel, heading, shape, atype, valid = [], [], [], [], [], []
        for aid in agent_ids:
            agent = self.engine.agent_manager.get_agent_by_id(aid)
            assert agent is not None
            xyz.append(agent.history_states.xyz)
            vel.append(agent.history_states.vel)
            heading.append(agent.history_states.heading)
            shape.append(agent.history_states.shape)
            atype.append(agent.history_states.type)
            valid.append(agent.history_states.valid)
        xyz = np.stack(xyz, axis=0)
        vel = np.stack(vel, axis=0)
        heading = np.stack(heading, axis=0)
        shape = np.stack(shape, axis=0)
        atype = np.stack(atype, axis=0).squeeze(-1)
        valid = np.stack(valid, axis=0)
        # hetero data & agent ids
        return (
            HeteroData(
                roadgraph=self.roadgraph_data,
                roadgraph_points=self.roadgraph_points,
                agent={
                    "xyz": torch.from_numpy(xyz).to(torch.float32),
                    "vel": torch.from_numpy(vel).to(torch.float32),
                    "heading": torch.from_numpy(heading).to(torch.float32),
                    "shape": torch.from_numpy(shape).to(torch.float32),
                    "type": torch.from_numpy(atype).to(torch.int32),
                    "valid": torch.from_numpy(valid).to(torch.bool),
                    "num_nodes": len(agent_ids),
                },
            ),
            agent_ids,
        )

    def plot_polylines(self, ax):
        dashes = [2, 4]
        for line in self.road_lines:
            if (line[1] + self.ROAD_LINE_TYPE_PREFIX) in plot.BROKEN_LINE_TYPE:
                plot.plot_dashed_line(
                    ax,
                    polyline=line[0],
                    dashes=dashes,
                    color=plot.get_color_from_type(
                        line[1] + self.ROAD_LINE_TYPE_PREFIX
                    ),
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

    def check_offroads(self, pos: np.ndarray) -> np.array:
        """check if pos is off road

        Args:
            pos (np.ndarray): (N,2)

        Returns:
            np.array: (N,) True if off road
        """
        edge_points = []
        edge_dir = []
        for edge in self.road_edges:
            edge_points.extend(edge[0])
            edge_dir.extend(get_polyline_dir(edge[0]))
        edge_points = np.array(edge_points)
        edge_dir = np.array(edge_dir)
        nearest_idx = np.argmin(
            np.linalg.norm(edge_points[:, None] - pos, axis=2), axis=0
        )
        pos2pt = edge_points[nearest_idx] - pos
        pt_dir = edge_dir[nearest_idx]
        cross = np.cross(pos2pt, pt_dir)
        return cross < 0
