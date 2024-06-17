"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
from ..geo import geo_pb2 as geo_dot_geo__pb2
from ..routing import routing_pb2 as routing_dot_routing__pb2
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10agent/trip.proto\x12\x05agent\x1a\rgeo/geo.proto\x1a\x15routing/routing.proto"h\n\nAgentState\x12!\n\x08position\x18\x01 \x01(\x0b2\x0f.geo.XYPosition\x12\x12\n\nvelocity_x\x18\x02 \x01(\x01\x12\x12\n\nvelocity_y\x18\x03 \x01(\x01\x12\x0f\n\x07heading\x18\x04 \x01(\x01"\xb2\x02\n\x04Trip\x12\x1d\n\x04mode\x18\x01 \x01(\x0e2\x0f.agent.TripMode\x12\x1a\n\x03end\x18\x02 \x01(\x0b2\r.geo.Position\x12\x1b\n\x0edeparture_time\x18\x03 \x01(\x01H\x00\x88\x01\x01\x12\x16\n\twait_time\x18\x04 \x01(\x01H\x01\x88\x01\x01\x12\x19\n\x0carrival_time\x18\x05 \x01(\x01H\x02\x88\x01\x01\x12\x15\n\x08activity\x18\x06 \x01(\tH\x03\x88\x01\x01\x12 \n\x06routes\x18\x07 \x03(\x0b2\x10.routing.Journey\x12\'\n\x0cagent_states\x18\x08 \x03(\x0b2\x11.agent.AgentStateB\x11\n\x0f_departure_timeB\x0c\n\n_wait_timeB\x0f\n\r_arrival_timeB\x0b\n\t_activity"\x90\x01\n\x08Schedule\x12\x1a\n\x05trips\x18\x01 \x03(\x0b2\x0b.agent.Trip\x12\x12\n\nloop_count\x18\x02 \x01(\x05\x12\x1b\n\x0edeparture_time\x18\x03 \x01(\x01H\x00\x88\x01\x01\x12\x16\n\twait_time\x18\x04 \x01(\x01H\x01\x88\x01\x01B\x11\n\x0f_departure_timeB\x0c\n\n_wait_time*\x89\x01\n\x08TripMode\x12\x19\n\x15TRIP_MODE_UNSPECIFIED\x10\x00\x12\x17\n\x13TRIP_MODE_WALK_ONLY\x10\x01\x12\x18\n\x14TRIP_MODE_DRIVE_ONLY\x10\x02\x12\x16\n\x12TRIP_MODE_BUS_WALK\x10\x04\x12\x17\n\x13TRIP_MODE_BIKE_WALK\x10\x05b\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'agent.trip_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _TRIPMODE._serialized_start = 628
    _TRIPMODE._serialized_end = 765
    _AGENTSTATE._serialized_start = 65
    _AGENTSTATE._serialized_end = 169
    _TRIP._serialized_start = 172
    _TRIP._serialized_end = 478
    _SCHEDULE._serialized_start = 481
    _SCHEDULE._serialized_end = 625