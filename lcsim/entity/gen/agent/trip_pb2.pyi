from geo import geo_pb2 as _geo_pb2
from routing import routing_pb2 as _routing_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union
DESCRIPTOR: _descriptor.FileDescriptor
TRIP_MODE_BIKE_WALK: TripMode
TRIP_MODE_BUS_WALK: TripMode
TRIP_MODE_DRIVE_ONLY: TripMode
TRIP_MODE_UNSPECIFIED: TripMode
TRIP_MODE_WALK_ONLY: TripMode

class AgentState(_message.Message):
    __slots__ = ['heading', 'position', 'velocity_x', 'velocity_y']
    HEADING_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_X_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_Y_FIELD_NUMBER: _ClassVar[int]
    heading: float
    position: _geo_pb2.XYPosition
    velocity_x: float
    velocity_y: float

    def __init__(self, position: _Optional[_Union[_geo_pb2.XYPosition, _Mapping]]=..., velocity_x: _Optional[float]=..., velocity_y: _Optional[float]=..., heading: _Optional[float]=...) -> None:
        ...

class Schedule(_message.Message):
    __slots__ = ['departure_time', 'loop_count', 'trips', 'wait_time']
    DEPARTURE_TIME_FIELD_NUMBER: _ClassVar[int]
    LOOP_COUNT_FIELD_NUMBER: _ClassVar[int]
    TRIPS_FIELD_NUMBER: _ClassVar[int]
    WAIT_TIME_FIELD_NUMBER: _ClassVar[int]
    departure_time: float
    loop_count: int
    trips: _containers.RepeatedCompositeFieldContainer[Trip]
    wait_time: float

    def __init__(self, trips: _Optional[_Iterable[_Union[Trip, _Mapping]]]=..., loop_count: _Optional[int]=..., departure_time: _Optional[float]=..., wait_time: _Optional[float]=...) -> None:
        ...

class Trip(_message.Message):
    __slots__ = ['activity', 'agent_states', 'arrival_time', 'departure_time', 'end', 'mode', 'routes', 'wait_time']
    ACTIVITY_FIELD_NUMBER: _ClassVar[int]
    AGENT_STATES_FIELD_NUMBER: _ClassVar[int]
    ARRIVAL_TIME_FIELD_NUMBER: _ClassVar[int]
    DEPARTURE_TIME_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    ROUTES_FIELD_NUMBER: _ClassVar[int]
    WAIT_TIME_FIELD_NUMBER: _ClassVar[int]
    activity: str
    agent_states: _containers.RepeatedCompositeFieldContainer[AgentState]
    arrival_time: float
    departure_time: float
    end: _geo_pb2.Position
    mode: TripMode
    routes: _containers.RepeatedCompositeFieldContainer[_routing_pb2.Journey]
    wait_time: float

    def __init__(self, mode: _Optional[_Union[TripMode, str]]=..., end: _Optional[_Union[_geo_pb2.Position, _Mapping]]=..., departure_time: _Optional[float]=..., wait_time: _Optional[float]=..., arrival_time: _Optional[float]=..., activity: _Optional[str]=..., routes: _Optional[_Iterable[_Union[_routing_pb2.Journey, _Mapping]]]=..., agent_states: _Optional[_Iterable[_Union[AgentState, _Mapping]]]=...) -> None:
        ...

class TripMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []