from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union
DESCRIPTOR: _descriptor.FileDescriptor

class LanePosition(_message.Message):
    __slots__ = ['lane_id', 's']
    LANE_ID_FIELD_NUMBER: _ClassVar[int]
    S_FIELD_NUMBER: _ClassVar[int]
    lane_id: int
    s: float

    def __init__(self, lane_id: _Optional[int]=..., s: _Optional[float]=...) -> None:
        ...

class LongLatBBox(_message.Message):
    __slots__ = ['max_latitude', 'max_longitude', 'min_latitude', 'min_longitude']
    MAX_LATITUDE_FIELD_NUMBER: _ClassVar[int]
    MAX_LONGITUDE_FIELD_NUMBER: _ClassVar[int]
    MIN_LATITUDE_FIELD_NUMBER: _ClassVar[int]
    MIN_LONGITUDE_FIELD_NUMBER: _ClassVar[int]
    max_latitude: float
    max_longitude: float
    min_latitude: float
    min_longitude: float

    def __init__(self, min_longitude: _Optional[float]=..., min_latitude: _Optional[float]=..., max_longitude: _Optional[float]=..., max_latitude: _Optional[float]=...) -> None:
        ...

class LongLatPosition(_message.Message):
    __slots__ = ['latitude', 'longitude', 'z']
    LATITUDE_FIELD_NUMBER: _ClassVar[int]
    LONGITUDE_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    latitude: float
    longitude: float
    z: float

    def __init__(self, longitude: _Optional[float]=..., latitude: _Optional[float]=..., z: _Optional[float]=...) -> None:
        ...

class Position(_message.Message):
    __slots__ = ['lane_position', 'longlat_position', 'xy_position']
    LANE_POSITION_FIELD_NUMBER: _ClassVar[int]
    LONGLAT_POSITION_FIELD_NUMBER: _ClassVar[int]
    XY_POSITION_FIELD_NUMBER: _ClassVar[int]
    lane_position: LanePosition
    longlat_position: LongLatPosition
    xy_position: XYPosition

    def __init__(self, lane_position: _Optional[_Union[LanePosition, _Mapping]]=..., longlat_position: _Optional[_Union[LongLatPosition, _Mapping]]=..., xy_position: _Optional[_Union[XYPosition, _Mapping]]=...) -> None:
        ...

class XYPosition(_message.Message):
    __slots__ = ['x', 'y', 'z']
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float

    def __init__(self, x: _Optional[float]=..., y: _Optional[float]=..., z: _Optional[float]=...) -> None:
        ...