"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rgeo/geo.proto\x12\x03geo"L\n\x0fLongLatPosition\x12\x11\n\tlongitude\x18\x01 \x01(\x01\x12\x10\n\x08latitude\x18\x02 \x01(\x01\x12\x0e\n\x01z\x18\x03 \x01(\x01H\x00\x88\x01\x01B\x04\n\x02_z"8\n\nXYPosition\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\x0e\n\x01z\x18\x03 \x01(\x01H\x00\x88\x01\x01B\x04\n\x02_z"*\n\x0cLanePosition\x12\x0f\n\x07lane_id\x18\x01 \x01(\x05\x12\t\n\x01s\x18\x02 \x01(\x01"\xd0\x01\n\x08Position\x12-\n\rlane_position\x18\x01 \x01(\x0b2\x11.geo.LanePositionH\x00\x88\x01\x01\x123\n\x10longlat_position\x18\x02 \x01(\x0b2\x14.geo.LongLatPositionH\x01\x88\x01\x01\x12)\n\x0bxy_position\x18\x03 \x01(\x0b2\x0f.geo.XYPositionH\x02\x88\x01\x01B\x10\n\x0e_lane_positionB\x13\n\x11_longlat_positionB\x0e\n\x0c_xy_position"g\n\x0bLongLatBBox\x12\x15\n\rmin_longitude\x18\x01 \x01(\x01\x12\x14\n\x0cmin_latitude\x18\x02 \x01(\x01\x12\x15\n\rmax_longitude\x18\x03 \x01(\x01\x12\x14\n\x0cmax_latitude\x18\x04 \x01(\x01b\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'geo.geo_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _LONGLATPOSITION._serialized_start = 22
    _LONGLATPOSITION._serialized_end = 98
    _XYPOSITION._serialized_start = 100
    _XYPOSITION._serialized_end = 156
    _LANEPOSITION._serialized_start = 158
    _LANEPOSITION._serialized_end = 200
    _POSITION._serialized_start = 203
    _POSITION._serialized_end = 411
    _LONGLATBBOX._serialized_start = 413
    _LONGLATBBOX._serialized_end = 516