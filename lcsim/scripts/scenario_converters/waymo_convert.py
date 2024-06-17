import json
import os
import sys
from pathlib import Path
from typing import Tuple

sys.path.append("/root/ads-test")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToJson, Parse, ParseDict
from matplotlib import pyplot as plt
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

from lcsim.entity.gen.agent import agent_pb2, trip_pb2
from lcsim.entity.gen.geo import geo_pb2
from lcsim.entity.gen.map import map_pb2

TIME_INTERVAL = 0.1


def default_attr() -> agent_pb2.AgentAttribute:
    attr = agent_pb2.AgentAttribute()
    attr.max_speed = 40
    attr.max_acceleration = 5
    attr.usual_acceleration = 2
    attr.max_braking_acceleration = -5
    attr.usual_braking_acceleration = -2
    return attr


def convert(scenario: scenario_pb2.Scenario) -> Tuple[map_pb2.Map, agent_pb2.Agents]:
    map_pb = map_pb2.Map()
    # include all lanes in one junction
    junction = map_pb.junctions.add()
    junction.id = 300000000
    # lanes
    for feature in scenario.map_features:
        _id = feature.id
        if feature.lane.ByteSize() > 0:
            lane = feature.lane
            if lane.type == 3:
                continue
            junction.lane_ids.append(_id)
            lane_pb = map_pb.lanes.add()
            lane_pb.id = _id
            lane_pb.center_line_type = lane.type
            lane_pb.max_speed = lane.speed_limit_mph * 0.44704  # mph to m/s
            polyline = map_pb2.Polyline()
            length = 0
            p = lane.polyline[0]
            polyline.nodes.append(
                geo_pb2.XYPosition(
                    x=p.x,
                    y=p.y,
                )
            )
            for p in lane.polyline[1:]:
                length += np.sqrt(
                    (p.x - polyline.nodes[-1].x) ** 2
                    + (p.y - polyline.nodes[-1].y) ** 2
                )
                polyline.nodes.append(
                    geo_pb2.XYPosition(
                        x=p.x,
                        y=p.y,
                    )
                )
            lane_pb.length = length
            lane_pb.center_line.CopyFrom(polyline)
            predecessor = []
            for el in lane.entry_lanes:
                predecessor.append(
                    map_pb2.LaneConnection(
                        id=el, type=map_pb2.LANE_CONNECTION_TYPE_HEAD
                    )
                )
            lane_pb.predecessors.extend(predecessor)
            successor = []
            for el in lane.exit_lanes:
                successor.append(
                    map_pb2.LaneConnection(
                        id=el, type=map_pb2.LANE_CONNECTION_TYPE_TAIL
                    )
                )
            lane_pb.successors.extend(successor)
            lane_pb.parent_id = junction.id
        if feature.road_line.ByteSize() > 0:
            road_line = feature.road_line
            road_line_pb = junction.road_lines.add()
            road_line_pb.type = road_line.type
            for p in road_line.polyline:
                road_line_pb.nodes.append(
                    geo_pb2.XYPosition(
                        x=p.x,
                        y=p.y,
                    )
                )
        if feature.road_edge.ByteSize() > 0:
            road_edge = feature.road_edge
            road_edge_pb = junction.road_edges.add()
            road_edge_pb.type = road_edge.type
            for p in road_edge.polyline:
                road_edge_pb.nodes.append(
                    geo_pb2.XYPosition(
                        x=p.x,
                        y=p.y,
                    )
                )
    # agents
    agents_pb = agent_pb2.Agents()
    agents_pb.ads_index = scenario.sdc_track_index
    for track in scenario.tracks:
        pb = agents_pb.agents.add()
        pb.id = track.id
        attr = default_attr()
        attr.type = track.object_type
        valid_states = []
        valid_index = []
        length, width = 0, 0
        for i, state in enumerate(track.states):
            if state.valid:
                valid_states.append(state)
                valid_index.append(i)
                length += state.length
                width += state.width
        length /= len(valid_states)
        width /= len(valid_states)
        attr.length = length
        attr.width = width
        pb.attribute.CopyFrom(attr)
        # home
        home_xy = valid_states[0].center_x, valid_states[0].center_y
        home = geo_pb2.Position(
            xy_position=geo_pb2.XYPosition(
                x=home_xy[0],
                y=home_xy[1],
            )
        )
        pb.home.CopyFrom(home)
        # schedule
        schedule = pb.schedules.add()
        schedule.loop_count = 1
        schedule.departure_time = valid_index[0] * TIME_INTERVAL
        trip = schedule.trips.add()
        end_xy = valid_states[-1].center_x, valid_states[-1].center_y
        trip.mode = (
            trip_pb2.TRIP_MODE_DRIVE_ONLY
            if attr.type == agent_pb2.AGENT_TYPE_VEHICLE
            else trip_pb2.TRIP_MODE_WALK_ONLY
        )
        trip.end.CopyFrom(
            geo_pb2.Position(
                xy_position=geo_pb2.XYPosition(
                    x=end_xy[0],
                    y=end_xy[1],
                )
            )
        )
        for vs in valid_states:
            state = trip.agent_states.add()
            state.position.CopyFrom(
                geo_pb2.XYPosition(
                    x=vs.center_x,
                    y=vs.center_y,
                )
            )
            state.velocity_x = vs.velocity_x
            state.velocity_y = vs.velocity_y
            state.heading = vs.heading
    return map_pb, agents_pb


def main():
    data_path = Path("/root/ads-test/data/womd_raw/training")
    pb_dir = "/root/ads-test/lcsim/cache/waymo"
    dataset = tf.data.TFRecordDataset(
        [str(p) for p in data_path.glob("*")], compression_type=""
    )
    bar = tqdm(total=len(list(dataset)))
    for data in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())
        map_pb, agents_pb = convert(scenario)
        scenario_id = scenario.scenario_id
        dir_path = os.path.join(pb_dir, str(scenario_id))
        os.makedirs(dir_path, exist_ok=True)
        map_file = os.path.join(dir_path, "map.pb")
        with open(map_file, "wb") as f:
            f.write(map_pb.SerializeToString())
        agents_file = os.path.join(dir_path, "agents.pb")
        with open(agents_file, "wb") as f:
            f.write(agents_pb.SerializeToString())
        bar.update(1)


if __name__ == "__main__":
    main()
