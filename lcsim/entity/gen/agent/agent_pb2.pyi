from agent import trip_pb2 as _trip_pb2
from geo import geo_pb2 as _geo_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union
AGENT_TYPE_CYCLIST: AgentType
AGENT_TYPE_OTHER: AgentType
AGENT_TYPE_PEDESTRIAN: AgentType
AGENT_TYPE_UNSPECIFIED: AgentType
AGENT_TYPE_VEHICLE: AgentType
DESCRIPTOR: _descriptor.FileDescriptor

class Agent(_message.Message):
    __slots__ = ['attribute', 'home', 'id', 'schedules', 'vehicle_attribute']
    ATTRIBUTE_FIELD_NUMBER: _ClassVar[int]
    HOME_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    SCHEDULES_FIELD_NUMBER: _ClassVar[int]
    VEHICLE_ATTRIBUTE_FIELD_NUMBER: _ClassVar[int]
    attribute: AgentAttribute
    home: _geo_pb2.Position
    id: int
    schedules: _containers.RepeatedCompositeFieldContainer[_trip_pb2.Schedule]
    vehicle_attribute: VehicleAttribute

    def __init__(self, id: _Optional[int]=..., attribute: _Optional[_Union[AgentAttribute, _Mapping]]=..., home: _Optional[_Union[_geo_pb2.Position, _Mapping]]=..., schedules: _Optional[_Iterable[_Union[_trip_pb2.Schedule, _Mapping]]]=..., vehicle_attribute: _Optional[_Union[VehicleAttribute, _Mapping]]=...) -> None:
        ...

class AgentAttribute(_message.Message):
    __slots__ = ['length', 'max_acceleration', 'max_braking_acceleration', 'max_speed', 'type', 'usual_acceleration', 'usual_braking_acceleration', 'width']
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    MAX_ACCELERATION_FIELD_NUMBER: _ClassVar[int]
    MAX_BRAKING_ACCELERATION_FIELD_NUMBER: _ClassVar[int]
    MAX_SPEED_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    USUAL_ACCELERATION_FIELD_NUMBER: _ClassVar[int]
    USUAL_BRAKING_ACCELERATION_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    length: float
    max_acceleration: float
    max_braking_acceleration: float
    max_speed: float
    type: AgentType
    usual_acceleration: float
    usual_braking_acceleration: float
    width: float

    def __init__(self, type: _Optional[_Union[AgentType, str]]=..., length: _Optional[float]=..., width: _Optional[float]=..., max_speed: _Optional[float]=..., max_acceleration: _Optional[float]=..., max_braking_acceleration: _Optional[float]=..., usual_acceleration: _Optional[float]=..., usual_braking_acceleration: _Optional[float]=...) -> None:
        ...

class Agents(_message.Message):
    __slots__ = ['ads_index', 'agents']
    ADS_INDEX_FIELD_NUMBER: _ClassVar[int]
    AGENTS_FIELD_NUMBER: _ClassVar[int]
    ads_index: int
    agents: _containers.RepeatedCompositeFieldContainer[Agent]

    def __init__(self, agents: _Optional[_Iterable[_Union[Agent, _Mapping]]]=..., ads_index: _Optional[int]=...) -> None:
        ...

class VehicleAttribute(_message.Message):
    __slots__ = ['lane_change_length', 'min_gap']
    LANE_CHANGE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MIN_GAP_FIELD_NUMBER: _ClassVar[int]
    lane_change_length: float
    min_gap: float

    def __init__(self, lane_change_length: _Optional[float]=..., min_gap: _Optional[float]=...) -> None:
        ...

class AgentType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []