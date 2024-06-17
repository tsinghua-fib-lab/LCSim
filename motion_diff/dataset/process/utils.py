import numpy as np
from waymo_open_dataset.protos import scenario_pb2


polyline_type = {
    # for lane
    "TYPE_UNDEFINED": -1,
    "TYPE_FREEWAY": 1,
    "TYPE_SURFACE_STREET": 2,
    "TYPE_BIKE_LANE": 3,
    # for roadline
    "TYPE_UNKNOWN": -1,
    "TYPE_BROKEN_SINGLE_WHITE": 6,
    "TYPE_SOLID_SINGLE_WHITE": 7,
    "TYPE_SOLID_DOUBLE_WHITE": 8,
    "TYPE_BROKEN_SINGLE_YELLOW": 9,
    "TYPE_BROKEN_DOUBLE_YELLOW": 10,
    "TYPE_SOLID_SINGLE_YELLOW": 11,
    "TYPE_SOLID_DOUBLE_YELLOW": 12,
    "TYPE_PASSING_DOUBLE_YELLOW": 13,
    # for roadedge
    "TYPE_ROAD_EDGE_BOUNDARY": 15,
    "TYPE_ROAD_EDGE_MEDIAN": 16,
    # for stopsign
    "TYPE_STOP_SIGN": 17,
    # for crosswalk
    "TYPE_CROSSWALK": 18,
    # for speed bump
    "TYPE_SPEED_BUMP": 19,
}

lane_type = {
    0: "TYPE_UNDEFINED",
    1: "TYPE_FREEWAY",
    2: "TYPE_SURFACE_STREET",
    3: "TYPE_BIKE_LANE",
}

road_line_type = {
    0: "TYPE_UNKNOWN",
    1: "TYPE_BROKEN_SINGLE_WHITE",
    2: "TYPE_SOLID_SINGLE_WHITE",
    3: "TYPE_SOLID_DOUBLE_WHITE",
    4: "TYPE_BROKEN_SINGLE_YELLOW",
    5: "TYPE_BROKEN_DOUBLE_YELLOW",
    6: "TYPE_SOLID_SINGLE_YELLOW",
    7: "TYPE_SOLID_DOUBLE_YELLOW",
    8: "TYPE_PASSING_DOUBLE_YELLOW",
}

road_edge_type = {
    0: "TYPE_UNKNOWN",
    # // Physical road boundary that doesn't have traffic on the other side (e.g.,
    # // a curb or the k-rail on the right side of a freeway).
    1: "TYPE_ROAD_EDGE_BOUNDARY",
    # // Physical road boundary that separates the car from other traffic
    # // (e.g. a k-rail or an island).
    2: "TYPE_ROAD_EDGE_MEDIAN",
}


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(
        np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1e9
    )
    return polyline_dir


def decode_roadgraph_feature(scenario: scenario_pb2.Scenario) -> dict:
    map_features = scenario.map_features
    # roadgraph
    roadgraph_xyz = []
    roadgraph_dir = []
    roadgraph_type = []
    roadgraph_id = []
    for feature in map_features:
        if feature.lane.ByteSize() > 0:
            pt_type = polyline_type[lane_type[feature.lane.type]]
            xyz = np.stack(
                [np.array([p.x, p.y, p.z]) for p in feature.lane.polyline],
                axis=0,
            )
            dir = get_polyline_dir(xyz)
        elif feature.road_line.ByteSize() > 0:
            pt_type = polyline_type[road_line_type[feature.road_line.type]]
            xyz = np.stack(
                [np.array([p.x, p.y, p.z]) for p in feature.road_line.polyline],
                axis=0,
            )
            dir = get_polyline_dir(xyz)
        elif feature.road_edge.ByteSize() > 0:
            pt_type = polyline_type[road_edge_type[feature.road_edge.type]]
            xyz = np.stack(
                [np.array([p.x, p.y, p.z]) for p in feature.road_edge.polyline],
                axis=0,
            )
            dir = get_polyline_dir(xyz)
        elif feature.crosswalk.ByteSize() > 0:
            pt_type = polyline_type["TYPE_CROSSWALK"]
            xyz = np.stack(
                [np.array([p.x, p.y, p.z]) for p in feature.crosswalk.polygon],
                axis=0,
            )
            dir = get_polyline_dir(xyz)
        elif feature.speed_bump.ByteSize() > 0:
            pt_type = polyline_type["TYPE_SPEED_BUMP"]
            xyz = np.stack(
                [np.array([p.x, p.y, p.z]) for p in feature.speed_bump.polygon],
                axis=0,
            )
            dir = get_polyline_dir(xyz)
        elif feature.stop_sign.ByteSize() > 0:
            pt_type = polyline_type["TYPE_STOP_SIGN"]
            point = feature.stop_sign.position
            xyz = np.stack([np.array([point.x, point.y, point.z])], axis=0)
            dir = np.zeros((1, 3))
        else:
            continue
        roadgraph_xyz.append(xyz)
        roadgraph_dir.append(dir)
        roadgraph_type.append(np.ones_like(xyz[:, 0], dtype=np.int8) * pt_type)
        roadgraph_id.append(np.ones_like(xyz[:, 0], dtype=np.int16) * feature.id)
    roadgraph_xyz = np.concatenate(roadgraph_xyz, axis=0, dtype=np.float32)
    roadgraph_dir = np.concatenate(roadgraph_dir, axis=0, dtype=np.float32)
    roadgraph_type = np.concatenate(roadgraph_type, axis=0, dtype=np.int8)
    roadgraph_id = np.concatenate(roadgraph_id, axis=0, dtype=np.int16)
    assert (
        roadgraph_xyz.shape[0]
        == roadgraph_dir.shape[0]
        == roadgraph_type.shape[0]
        == roadgraph_id.shape[0]
    )
    return {
        "roadgraph_xyz": roadgraph_xyz,
        "roadgraph_dir": roadgraph_dir,
        "roadgraph_type": roadgraph_type,
        "roadgraph_id": roadgraph_id,
    }


def decode_agent_feature(scenario: scenario_pb2.Scenario) -> dict:
    tracks = scenario.tracks
    agent_type = []
    agent_xyz = []
    agent_vel = []
    agent_heading = []
    agent_shape = []
    agent_valid = []
    for track in tracks:
        agent_type.append(track.object_type)
        states = track.states
        xyz = np.stack(
            [
                np.array([s.center_x, s.center_y, s.center_z], dtype=np.float32)
                for s in states
            ],
            axis=0,
        )
        vel = np.stack(
            [np.array([s.velocity_x, s.velocity_y], dtype=np.float32) for s in states],
            axis=0,
        )
        heading = np.stack(
            [np.array([s.heading], dtype=np.float32) for s in states],
            axis=0,
        )
        shape = np.stack(
            [np.array([s.length, s.width, s.height], dtype=np.float32) for s in states],
            axis=0,
        )
        valid = np.stack(
            [np.array([s.valid], dtype=np.bool8) for s in states],
            axis=0,
        )
        agent_xyz.append(xyz)
        agent_vel.append(vel)
        agent_heading.append(heading)
        agent_shape.append(shape)
        agent_valid.append(valid)
    agent_type = np.array(agent_type, dtype=np.int8)
    agent_xyz = np.stack(agent_xyz, axis=0)
    agent_vel = np.stack(agent_vel, axis=0)
    agent_heading = np.stack(agent_heading, axis=0)
    agent_shape = np.stack(agent_shape, axis=0)
    agent_valid = np.stack(agent_valid, axis=0)
    assert (
        agent_type.shape[0]
        == agent_xyz.shape[0]
        == agent_vel.shape[0]
        == agent_heading.shape[0]
        == agent_shape.shape[0]
        == agent_valid.shape[0]
    )
    return {
        "agent_type": agent_type,
        "agent_xyz": agent_xyz,
        "agent_vel": agent_vel,
        "agent_heading": agent_heading,
        "agent_shape": agent_shape,
        "agent_valid": agent_valid,
    }
