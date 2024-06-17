from geo import geo_pb2 as _geo_pb2
from map import light_pb2 as _light_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union
CENTER_LINE_TYPE_BIKE: CenterLineType
CENTER_LINE_TYPE_FREEWAY: CenterLineType
CENTER_LINE_TYPE_SURFACE_STREET: CenterLineType
CENTER_LINE_TYPE_UNSPECIFIED: CenterLineType
DESCRIPTOR: _descriptor.FileDescriptor
LANE_CONNECTION_TYPE_HEAD: LaneConnectionType
LANE_CONNECTION_TYPE_TAIL: LaneConnectionType
LANE_CONNECTION_TYPE_UNSPECIFIED: LaneConnectionType
LANE_TURN_AROUND: LaneTurn
LANE_TURN_LEFT: LaneTurn
LANE_TURN_RIGHT: LaneTurn
LANE_TURN_STRAIGHT: LaneTurn
LANE_TURN_UNSPECIFIED: LaneTurn
LANE_TYPE_DRIVING: LaneType
LANE_TYPE_UNSPECIFIED: LaneType
LANE_TYPE_WALKING: LaneType
ROAD_EDGE_TYPE_BOUNDARY: RoadEdgeType
ROAD_EDGE_TYPE_MEDIAN: RoadEdgeType
ROAD_EDGE_TYPE_UNSPECIFIED: RoadEdgeType
ROAD_LINE_TYPE_BROKEN_DOUBLE_YELLOW: RoadLineType
ROAD_LINE_TYPE_BROKEN_SINGLE_WHITE: RoadLineType
ROAD_LINE_TYPE_BROKEN_SINGLE_YELLOW: RoadLineType
ROAD_LINE_TYPE_PASSING_DOUBLE_YELLOW: RoadLineType
ROAD_LINE_TYPE_SOLID_DOUBLE_WHITE: RoadLineType
ROAD_LINE_TYPE_SOLID_DOUBLE_YELLOW: RoadLineType
ROAD_LINE_TYPE_SOLID_SINGLE_WHITE: RoadLineType
ROAD_LINE_TYPE_SOLID_SINGLE_YELLOW: RoadLineType
ROAD_LINE_TYPE_UNSPECIFIED: RoadLineType

class Header(_message.Message):
    __slots__ = ['date', 'east', 'name', 'north', 'projection', 'south', 'west']
    DATE_FIELD_NUMBER: _ClassVar[int]
    EAST_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NORTH_FIELD_NUMBER: _ClassVar[int]
    PROJECTION_FIELD_NUMBER: _ClassVar[int]
    SOUTH_FIELD_NUMBER: _ClassVar[int]
    WEST_FIELD_NUMBER: _ClassVar[int]
    date: str
    east: float
    name: str
    north: float
    projection: str
    south: float
    west: float

    def __init__(self, name: _Optional[str]=..., date: _Optional[str]=..., north: _Optional[float]=..., south: _Optional[float]=..., east: _Optional[float]=..., west: _Optional[float]=..., projection: _Optional[str]=...) -> None:
        ...

class Junction(_message.Message):
    __slots__ = ['driving_lane_groups', 'fixed_program', 'id', 'lane_ids', 'phases', 'road_edges', 'road_lines']
    DRIVING_LANE_GROUPS_FIELD_NUMBER: _ClassVar[int]
    FIXED_PROGRAM_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    LANE_IDS_FIELD_NUMBER: _ClassVar[int]
    PHASES_FIELD_NUMBER: _ClassVar[int]
    ROAD_EDGES_FIELD_NUMBER: _ClassVar[int]
    ROAD_LINES_FIELD_NUMBER: _ClassVar[int]
    driving_lane_groups: _containers.RepeatedCompositeFieldContainer[JunctionLaneGroup]
    fixed_program: _light_pb2.TrafficLight
    id: int
    lane_ids: _containers.RepeatedScalarFieldContainer[int]
    phases: _containers.RepeatedCompositeFieldContainer[_light_pb2.AvailablePhase]
    road_edges: _containers.RepeatedCompositeFieldContainer[RoadEdge]
    road_lines: _containers.RepeatedCompositeFieldContainer[RoadLine]

    def __init__(self, id: _Optional[int]=..., lane_ids: _Optional[_Iterable[int]]=..., driving_lane_groups: _Optional[_Iterable[_Union[JunctionLaneGroup, _Mapping]]]=..., phases: _Optional[_Iterable[_Union[_light_pb2.AvailablePhase, _Mapping]]]=..., fixed_program: _Optional[_Union[_light_pb2.TrafficLight, _Mapping]]=..., road_lines: _Optional[_Iterable[_Union[RoadLine, _Mapping]]]=..., road_edges: _Optional[_Iterable[_Union[RoadEdge, _Mapping]]]=...) -> None:
        ...

class JunctionLaneGroup(_message.Message):
    __slots__ = ['in_angle', 'in_road_id', 'lane_ids', 'out_angle', 'out_road_id', 'turn']
    IN_ANGLE_FIELD_NUMBER: _ClassVar[int]
    IN_ROAD_ID_FIELD_NUMBER: _ClassVar[int]
    LANE_IDS_FIELD_NUMBER: _ClassVar[int]
    OUT_ANGLE_FIELD_NUMBER: _ClassVar[int]
    OUT_ROAD_ID_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    in_angle: float
    in_road_id: int
    lane_ids: _containers.RepeatedScalarFieldContainer[int]
    out_angle: float
    out_road_id: int
    turn: LaneTurn

    def __init__(self, in_road_id: _Optional[int]=..., in_angle: _Optional[float]=..., out_road_id: _Optional[int]=..., out_angle: _Optional[float]=..., lane_ids: _Optional[_Iterable[int]]=..., turn: _Optional[_Union[LaneTurn, str]]=...) -> None:
        ...

class Lane(_message.Message):
    __slots__ = ['center_line', 'center_line_type', 'id', 'left_border_line', 'left_lane_ids', 'length', 'max_speed', 'overlaps', 'parent_id', 'predecessors', 'right_border_line', 'right_lane_ids', 'successors', 'turn', 'type', 'width']
    CENTER_LINE_FIELD_NUMBER: _ClassVar[int]
    CENTER_LINE_TYPE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    LEFT_BORDER_LINE_FIELD_NUMBER: _ClassVar[int]
    LEFT_LANE_IDS_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    MAX_SPEED_FIELD_NUMBER: _ClassVar[int]
    OVERLAPS_FIELD_NUMBER: _ClassVar[int]
    PARENT_ID_FIELD_NUMBER: _ClassVar[int]
    PREDECESSORS_FIELD_NUMBER: _ClassVar[int]
    RIGHT_BORDER_LINE_FIELD_NUMBER: _ClassVar[int]
    RIGHT_LANE_IDS_FIELD_NUMBER: _ClassVar[int]
    SUCCESSORS_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    center_line: Polyline
    center_line_type: CenterLineType
    id: int
    left_border_line: Polyline
    left_lane_ids: _containers.RepeatedScalarFieldContainer[int]
    length: float
    max_speed: float
    overlaps: _containers.RepeatedCompositeFieldContainer[LaneOverlap]
    parent_id: int
    predecessors: _containers.RepeatedCompositeFieldContainer[LaneConnection]
    right_border_line: Polyline
    right_lane_ids: _containers.RepeatedScalarFieldContainer[int]
    successors: _containers.RepeatedCompositeFieldContainer[LaneConnection]
    turn: LaneTurn
    type: LaneType
    width: float

    def __init__(self, id: _Optional[int]=..., type: _Optional[_Union[LaneType, str]]=..., turn: _Optional[_Union[LaneTurn, str]]=..., max_speed: _Optional[float]=..., length: _Optional[float]=..., width: _Optional[float]=..., center_line: _Optional[_Union[Polyline, _Mapping]]=..., left_border_line: _Optional[_Union[Polyline, _Mapping]]=..., right_border_line: _Optional[_Union[Polyline, _Mapping]]=..., predecessors: _Optional[_Iterable[_Union[LaneConnection, _Mapping]]]=..., successors: _Optional[_Iterable[_Union[LaneConnection, _Mapping]]]=..., left_lane_ids: _Optional[_Iterable[int]]=..., right_lane_ids: _Optional[_Iterable[int]]=..., parent_id: _Optional[int]=..., overlaps: _Optional[_Iterable[_Union[LaneOverlap, _Mapping]]]=..., center_line_type: _Optional[_Union[CenterLineType, str]]=...) -> None:
        ...

class LaneConnection(_message.Message):
    __slots__ = ['id', 'type']
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    id: int
    type: LaneConnectionType

    def __init__(self, id: _Optional[int]=..., type: _Optional[_Union[LaneConnectionType, str]]=...) -> None:
        ...

class LaneOverlap(_message.Message):
    __slots__ = ['other', 'self', 'self_first']
    OTHER_FIELD_NUMBER: _ClassVar[int]
    SELF_FIELD_NUMBER: _ClassVar[int]
    SELF_FIRST_FIELD_NUMBER: _ClassVar[int]
    other: _geo_pb2.LanePosition
    self: _geo_pb2.LanePosition
    self_first: bool

    def __init__(self, self_: _Optional[_Union[_geo_pb2.LanePosition, _Mapping]]=..., other: _Optional[_Union[_geo_pb2.LanePosition, _Mapping]]=..., self_first: bool=...) -> None:
        ...

class Map(_message.Message):
    __slots__ = ['header', 'junctions', 'lanes', 'roads']
    HEADER_FIELD_NUMBER: _ClassVar[int]
    JUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    LANES_FIELD_NUMBER: _ClassVar[int]
    ROADS_FIELD_NUMBER: _ClassVar[int]
    header: Header
    junctions: _containers.RepeatedCompositeFieldContainer[Junction]
    lanes: _containers.RepeatedCompositeFieldContainer[Lane]
    roads: _containers.RepeatedCompositeFieldContainer[Road]

    def __init__(self, header: _Optional[_Union[Header, _Mapping]]=..., lanes: _Optional[_Iterable[_Union[Lane, _Mapping]]]=..., roads: _Optional[_Iterable[_Union[Road, _Mapping]]]=..., junctions: _Optional[_Iterable[_Union[Junction, _Mapping]]]=...) -> None:
        ...

class NextRoadLane(_message.Message):
    __slots__ = ['lane_id_a', 'lane_id_b', 'road_id']
    LANE_ID_A_FIELD_NUMBER: _ClassVar[int]
    LANE_ID_B_FIELD_NUMBER: _ClassVar[int]
    ROAD_ID_FIELD_NUMBER: _ClassVar[int]
    lane_id_a: int
    lane_id_b: int
    road_id: int

    def __init__(self, road_id: _Optional[int]=..., lane_id_a: _Optional[int]=..., lane_id_b: _Optional[int]=...) -> None:
        ...

class NextRoadLanePlan(_message.Message):
    __slots__ = ['next_road_lanes']
    NEXT_ROAD_LANES_FIELD_NUMBER: _ClassVar[int]
    next_road_lanes: _containers.RepeatedCompositeFieldContainer[NextRoadLane]

    def __init__(self, next_road_lanes: _Optional[_Iterable[_Union[NextRoadLane, _Mapping]]]=...) -> None:
        ...

class Polyline(_message.Message):
    __slots__ = ['nodes']
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[_geo_pb2.XYPosition]

    def __init__(self, nodes: _Optional[_Iterable[_Union[_geo_pb2.XYPosition, _Mapping]]]=...) -> None:
        ...

class Road(_message.Message):
    __slots__ = ['id', 'lane_ids', 'name', 'next_road_lane_plans', 'road_connections', 'road_edges', 'road_lines']
    ID_FIELD_NUMBER: _ClassVar[int]
    LANE_IDS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NEXT_ROAD_LANE_PLANS_FIELD_NUMBER: _ClassVar[int]
    ROAD_CONNECTIONS_FIELD_NUMBER: _ClassVar[int]
    ROAD_EDGES_FIELD_NUMBER: _ClassVar[int]
    ROAD_LINES_FIELD_NUMBER: _ClassVar[int]
    id: int
    lane_ids: _containers.RepeatedScalarFieldContainer[int]
    name: str
    next_road_lane_plans: _containers.RepeatedCompositeFieldContainer[NextRoadLanePlan]
    road_connections: _containers.RepeatedCompositeFieldContainer[RoadConnection]
    road_edges: _containers.RepeatedCompositeFieldContainer[RoadEdge]
    road_lines: _containers.RepeatedCompositeFieldContainer[RoadLine]

    def __init__(self, id: _Optional[int]=..., name: _Optional[str]=..., lane_ids: _Optional[_Iterable[int]]=..., next_road_lane_plans: _Optional[_Iterable[_Union[NextRoadLanePlan, _Mapping]]]=..., road_lines: _Optional[_Iterable[_Union[RoadLine, _Mapping]]]=..., road_edges: _Optional[_Iterable[_Union[RoadEdge, _Mapping]]]=..., road_connections: _Optional[_Iterable[_Union[RoadConnection, _Mapping]]]=...) -> None:
        ...

class RoadConnection(_message.Message):
    __slots__ = ['nodes', 'road_id', 'type']
    NODES_FIELD_NUMBER: _ClassVar[int]
    ROAD_ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[_geo_pb2.XYPosition]
    road_id: int
    type: RoadLineType

    def __init__(self, type: _Optional[_Union[RoadLineType, str]]=..., nodes: _Optional[_Iterable[_Union[_geo_pb2.XYPosition, _Mapping]]]=..., road_id: _Optional[int]=...) -> None:
        ...

class RoadEdge(_message.Message):
    __slots__ = ['nodes', 'type']
    NODES_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[_geo_pb2.XYPosition]
    type: RoadEdgeType

    def __init__(self, type: _Optional[_Union[RoadEdgeType, str]]=..., nodes: _Optional[_Iterable[_Union[_geo_pb2.XYPosition, _Mapping]]]=...) -> None:
        ...

class RoadLine(_message.Message):
    __slots__ = ['nodes', 'type']
    NODES_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[_geo_pb2.XYPosition]
    type: RoadLineType

    def __init__(self, type: _Optional[_Union[RoadLineType, str]]=..., nodes: _Optional[_Iterable[_Union[_geo_pb2.XYPosition, _Mapping]]]=...) -> None:
        ...

class LaneType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class LaneTurn(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class LaneConnectionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class CenterLineType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class RoadLineType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class RoadEdgeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []