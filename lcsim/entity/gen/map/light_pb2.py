"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fmap/light.proto\x12\x03map":\n\x05Phase\x12\x10\n\x08duration\x18\x01 \x01(\x01\x12\x1f\n\x06states\x18\x02 \x03(\x0e2\x0f.map.LightState"1\n\x0eAvailablePhase\x12\x1f\n\x06states\x18\x01 \x03(\x0e2\x0f.map.LightState"?\n\x0cTrafficLight\x12\x13\n\x0bjunction_id\x18\x01 \x01(\x05\x12\x1a\n\x06phases\x18\x02 \x03(\x0b2\n.map.Phase*m\n\nLightState\x12\x1b\n\x17LIGHT_STATE_UNSPECIFIED\x10\x00\x12\x13\n\x0fLIGHT_STATE_RED\x10\x01\x12\x15\n\x11LIGHT_STATE_GREEN\x10\x02\x12\x16\n\x12LIGHT_STATE_YELLOW\x10\x03b\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'map.light_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _LIGHTSTATE._serialized_start = 200
    _LIGHTSTATE._serialized_end = 309
    _PHASE._serialized_start = 24
    _PHASE._serialized_end = 82
    _AVAILABLEPHASE._serialized_start = 84
    _AVAILABLEPHASE._serialized_end = 133
    _TRAFFICLIGHT._serialized_start = 135
    _TRAFFICLIGHT._serialized_end = 198