import argparse
import math
from typing import Dict, List

import av2.geometry.polyline_utils as polyline_utils
import numpy as np
import pymongo
from entity.gen.agent import agent_pb2
from entity.gen.geo import geo_pb2
from entity.gen.map import map_pb2
from google.protobuf.json_format import MessageToJson, Parse, ParseDict

ROAD_EDGE_DIST = 3  # distance from the road center to the edge of the road


def parse_args():
    parser = argparse.ArgumentParser(
        description="Converts a scenario from the MOSS format to the lcsim format"
    )
    parser.add_argument("mongo_uri", type=str, help="URI for the MongoDB instance")
    parser.add_argument("db_name", type=str, help="Name of the MongoDB database")
    parser.add_argument(
        "collection_name", type=str, help="Name of the MongoDB collection"
    )
    parser.add_argument("output_file", type=str, help="Name of the output file")
    return parser.parse_args()


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    args = parse_args()
    client = pymongo.MongoClient(args.mongo_uri)
    col = client[args.db_name][args.collection_name]
    lanes_raw = list(col.find({"class": "lane"}))
    lanes = {}
    lane_pbs: Dict[int, map_pb2.Lane] = {}
    for lane in lanes_raw:
        lanes[lane["data"]["id"]] = lane["data"]
        lane_pbs[lane["data"]["id"]] = ParseDict(
            lane["data"], map_pb2.Lane(), ignore_unknown_fields=True
        )
    roads_raw = list(col.find({"class": "road"}))
    roads = {}
    road_pbs: Dict[int, map_pb2.Road] = {}
    for road in roads_raw:
        roads[road["data"]["id"]] = road["data"]
        road_pbs[road["data"]["id"]] = ParseDict(road["data"], map_pb2.Road())
    junctions_raw = list(col.find({"class": "junction"}))
    junctions = {}
    junction_pbs: Dict[int, map_pb2.Junction] = {}
    for junction in junctions_raw:
        junctions[junction["data"]["id"]] = junction["data"]
        junction_pbs[junction["data"]["id"]] = ParseDict(
            junction["data"], map_pb2.Junction()
        )
    print(
        f"Loaded {len(lanes)} lanes, {len(roads)} roads, and {len(junctions)} junctions"
    )

    road_groups = {}
    for road in roads.values():
        l = lanes[road["lane_ids"][0]]
        pre_ids = l["predecessors"]
        junc_from = -1 if len(pre_ids) == 0 else lanes[pre_ids[0]["id"]]["parent_id"]
        succ_ids = l["successors"]
        junc_to = -1 if len(succ_ids) == 0 else lanes[succ_ids[0]["id"]]["parent_id"]
        group_key = (
            (junc_from, junc_to) if junc_from > junc_to else (junc_to, junc_from)
        )
        if group_key in road_groups:
            road_groups[group_key].append(road)
        else:
            road_groups[group_key] = [road]

    # add road connections, road lines, and road edges to the road protobufs
    for juncs, rs in road_groups.items():
        groups = [rs]  # parallel roads with reverse direction
        if len(rs) > 2:
            assert juncs[1] == -1  # boundary of the map
            junc = junctions[juncs[0]]
            # group roads by the angle with the junction
            road_angles = {
                dlg["in_road_id"]: dlg["in_angle"]
                for dlg in junc["driving_lane_groups"]
            } | {
                dlg["out_road_id"]: wrap_angle(dlg["out_angle"] - math.pi)
                for dlg in junc["driving_lane_groups"]
            }
            groups = []
            for road in rs:
                angle = road_angles[road["id"]]
                found = False
                for group in groups:
                    if abs(angle - group[0]) < math.pi / 6:
                        group.append(road)
                        found = True
                        break
                if not found:
                    groups.append([angle, road])
            groups = [g[1:] for g in groups]
        # complete the topology information
        for g in groups:
            assert len(g) > 0 and len(g) <= 2
            # add solid double yellow between the roads and single broken white between the straight lanes
            for r in g:
                pb = road_pbs[r["id"]]
                del pb.road_lines[:], pb.road_edges[:], pb.road_connections[:]
                lids = pb.lane_ids
                # between the straight lanes and the left/right turning lanes, add a single solid white line near the junction
                # and a single broken white line between the straight lanes
                for index, lid in enumerate(lids[:-1]):
                    ll = lane_pbs[lid]
                    rl = lane_pbs[lids[index + 1]]
                    lns, rns = ll.center_line.nodes, rl.center_line.nodes
                    mid_pts, _ = polyline_utils.interp_utils.compute_midpoint_line(
                        left_ln_boundary=np.array([[ln.x, ln.y] for ln in lns]),
                        right_ln_boundary=np.array([[rn.x, rn.y] for rn in rns]),
                        num_interp_pts=max(len(lns), len(rns)),
                    )
                    pb.road_lines.append(
                        map_pb2.RoadLine(
                            type=map_pb2.ROAD_LINE_TYPE_BROKEN_SINGLE_WHITE,
                            nodes=[
                                geo_pb2.XYPosition(x=pt[0], y=pt[1]) for pt in mid_pts
                            ],
                        )
                    )
                # add road edges to the road
                el = lane_pbs[lids[-1]]
                el_nodes = np.array([[ln.x, ln.y] for ln in el.center_line.nodes])
                el_dir = np.zeros_like(el_nodes)
                el_dir[:-1] = el_nodes[1:] - el_nodes[:-1]
                el_dir[-1] = el_dir[-2]
                el_dir = el_dir / np.linalg.norm(el_dir, axis=1, keepdims=True)
                el_dir = np.stack([el_dir[:, 1], -el_dir[:, 0]], axis=1)
                edge_nodes = el_nodes + ROAD_EDGE_DIST * el_dir
                pb.road_edges.append(
                    map_pb2.RoadEdge(
                        type=map_pb2.ROAD_EDGE_TYPE_BOUNDARY,
                        nodes=[
                            geo_pb2.XYPosition(x=pt[0], y=pt[1]) for pt in edge_nodes
                        ],
                    )
                )
            if len(g) == 2:
                # add solid double yellow between the two roads
                pb1 = road_pbs[g[0]["id"]]
                pb2 = road_pbs[g[1]["id"]]
                l1 = lane_pbs[pb1.lane_ids[0]]
                l2 = lane_pbs[pb2.lane_ids[0]]
                l1_nodes = np.array([[ln.x, ln.y] for ln in l1.center_line.nodes])
                l2_nodes = np.array([[ln.x, ln.y] for ln in l2.center_line.nodes])[::-1]
                mid_pts, _ = polyline_utils.interp_utils.compute_midpoint_line(
                    left_ln_boundary=l1_nodes,
                    right_ln_boundary=l2_nodes,
                    num_interp_pts=max(len(l1_nodes), len(l2_nodes)),
                )
                pb1.road_connections.append(
                    map_pb2.RoadConnection(
                        type=map_pb2.ROAD_LINE_TYPE_SOLID_DOUBLE_YELLOW,
                        nodes=[geo_pb2.XYPosition(x=pt[0], y=pt[1]) for pt in mid_pts],
                        road_id=pb2.id,
                    )
                )
                pb2.road_connections.append(
                    map_pb2.RoadConnection(
                        type=map_pb2.ROAD_LINE_TYPE_SOLID_DOUBLE_YELLOW,
                        nodes=[
                            geo_pb2.XYPosition(x=pt[0], y=pt[1]) for pt in mid_pts[::-1]
                        ],
                        road_id=pb1.id,
                    )
                )

    # add road edges to the junction protobufs
    for junc_id, junc_pb in junction_pbs.items():
        del junc_pb.road_edges[:]
        # find the boundary roads, i.e., the right-turn lanes connected to the rightmost lane of the roads entering the junction
        in_roads = set()
        for dlg in junc_pb.driving_lane_groups:
            in_roads.add(dlg.in_road_id)
        boundary_lane_ids = set()
        for ir in in_roads:
            in_road = road_pbs[ir]
            ril = lane_pbs[in_road.lane_ids[-1]]
            for succ in ril.successors:
                succ_lane = lane_pbs[succ.id]
                if succ_lane.turn == map_pb2.LANE_TURN_RIGHT:
                    boundary_lane_ids.add(succ.id)
        # add road edges to the junction
        for lid in boundary_lane_ids:
            lane = lane_pbs[lid]
            nodes = np.array([[ln.x, ln.y] for ln in lane.center_line.nodes])
            nodes_dir = np.zeros_like(nodes)
            nodes_dir[:-1] = nodes[1:] - nodes[:-1]
            nodes_dir[-1] = nodes_dir[-2]
            nodes_dir = nodes_dir / np.linalg.norm(nodes_dir, axis=1, keepdims=True)
            nodes_dir = np.stack([nodes_dir[:, 1], -nodes_dir[:, 0]], axis=1)
            edge_nodes = nodes + ROAD_EDGE_DIST * nodes_dir
            junc_pb.road_edges.append(
                map_pb2.RoadEdge(
                    type=map_pb2.ROAD_EDGE_TYPE_BOUNDARY,
                    nodes=[geo_pb2.XYPosition(x=pt[0], y=pt[1]) for pt in edge_nodes],
                )
            )

    # save the map protobuf file
    map_pb = map_pb2.Map()
    map_pb.lanes.extend(lane_pbs.values())
    map_pb.roads.extend(road_pbs.values())
    map_pb.junctions.extend(junction_pbs.values())
    with open(args.output_file, "w") as f:
        f.write(MessageToJson(map_pb))


if __name__ == "__main__":
    main()
