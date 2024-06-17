from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union
DESCRIPTOR: _descriptor.FileDescriptor
JOURNEY_TYPE_BY_BUS: JourneyType
JOURNEY_TYPE_DRIVING: JourneyType
JOURNEY_TYPE_UNSPECIFIED: JourneyType
JOURNEY_TYPE_WALKING: JourneyType
MOVING_DIRECTION_BACKWARD: MovingDirection
MOVING_DIRECTION_FORWARD: MovingDirection
MOVING_DIRECTION_UNSPECIFIED: MovingDirection
ROUTE_TYPE_BY_BUS: RouteType
ROUTE_TYPE_DRIVING: RouteType
ROUTE_TYPE_UNSPECIFIED: RouteType
ROUTE_TYPE_WALKING: RouteType

class BusJourneyBody(_message.Message):
    __slots__ = ['end_station_id', 'line_id', 'start_station_id']
    END_STATION_ID_FIELD_NUMBER: _ClassVar[int]
    LINE_ID_FIELD_NUMBER: _ClassVar[int]
    START_STATION_ID_FIELD_NUMBER: _ClassVar[int]
    end_station_id: int
    line_id: int
    start_station_id: int

    def __init__(self, line_id: _Optional[int]=..., start_station_id: _Optional[int]=..., end_station_id: _Optional[int]=...) -> None:
        ...

class BusLine(_message.Message):
    __slots__ = ['count', 'distances', 'interval', 'line_id', 'stops']
    COUNT_FIELD_NUMBER: _ClassVar[int]
    DISTANCES_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_FIELD_NUMBER: _ClassVar[int]
    LINE_ID_FIELD_NUMBER: _ClassVar[int]
    STOPS_FIELD_NUMBER: _ClassVar[int]
    count: int
    distances: _containers.RepeatedScalarFieldContainer[float]
    interval: int
    line_id: int
    stops: _containers.RepeatedScalarFieldContainer[int]

    def __init__(self, line_id: _Optional[int]=..., stops: _Optional[_Iterable[int]]=..., distances: _Optional[_Iterable[float]]=..., interval: _Optional[int]=..., count: _Optional[int]=...) -> None:
        ...

class BusLines(_message.Message):
    __slots__ = ['lines']
    LINES_FIELD_NUMBER: _ClassVar[int]
    lines: _containers.RepeatedCompositeFieldContainer[BusLine]

    def __init__(self, lines: _Optional[_Iterable[_Union[BusLine, _Mapping]]]=...) -> None:
        ...

class DrivingJourneyBody(_message.Message):
    __slots__ = ['eta', 'road_ids']
    ETA_FIELD_NUMBER: _ClassVar[int]
    ROAD_IDS_FIELD_NUMBER: _ClassVar[int]
    eta: float
    road_ids: _containers.RepeatedScalarFieldContainer[int]

    def __init__(self, road_ids: _Optional[_Iterable[int]]=..., eta: _Optional[float]=...) -> None:
        ...

class Journey(_message.Message):
    __slots__ = ['by_bus', 'driving', 'type', 'walking']
    BY_BUS_FIELD_NUMBER: _ClassVar[int]
    DRIVING_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    WALKING_FIELD_NUMBER: _ClassVar[int]
    by_bus: BusJourneyBody
    driving: DrivingJourneyBody
    type: JourneyType
    walking: WalkingJourneyBody

    def __init__(self, type: _Optional[_Union[JourneyType, str]]=..., driving: _Optional[_Union[DrivingJourneyBody, _Mapping]]=..., walking: _Optional[_Union[WalkingJourneyBody, _Mapping]]=..., by_bus: _Optional[_Union[BusJourneyBody, _Mapping]]=...) -> None:
        ...

class RoadStatus(_message.Message):
    __slots__ = ['id', 'speed']
    ID_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    id: int
    speed: _containers.RepeatedScalarFieldContainer[float]

    def __init__(self, id: _Optional[int]=..., speed: _Optional[_Iterable[float]]=...) -> None:
        ...

class RoadStatuses(_message.Message):
    __slots__ = ['road_statuses']
    ROAD_STATUSES_FIELD_NUMBER: _ClassVar[int]
    road_statuses: _containers.RepeatedCompositeFieldContainer[RoadStatus]

    def __init__(self, road_statuses: _Optional[_Iterable[_Union[RoadStatus, _Mapping]]]=...) -> None:
        ...

class WalkingJourneyBody(_message.Message):
    __slots__ = ['eta', 'route']
    ETA_FIELD_NUMBER: _ClassVar[int]
    ROUTE_FIELD_NUMBER: _ClassVar[int]
    eta: float
    route: _containers.RepeatedCompositeFieldContainer[WalkingRouteSegment]

    def __init__(self, route: _Optional[_Iterable[_Union[WalkingRouteSegment, _Mapping]]]=..., eta: _Optional[float]=...) -> None:
        ...

class WalkingRouteSegment(_message.Message):
    __slots__ = ['lane_id', 'moving_direction']
    LANE_ID_FIELD_NUMBER: _ClassVar[int]
    MOVING_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    lane_id: int
    moving_direction: MovingDirection

    def __init__(self, lane_id: _Optional[int]=..., moving_direction: _Optional[_Union[MovingDirection, str]]=...) -> None:
        ...

class RouteType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class JourneyType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class MovingDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []