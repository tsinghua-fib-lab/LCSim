"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
from ..agent import trip_pb2 as agent_dot_trip__pb2
from ..geo import geo_pb2 as geo_dot_geo__pb2
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11agent/agent.proto\x12\x05agent\x1a\x10agent/trip.proto\x1a\rgeo/geo.proto"\xde\x01\n\x0eAgentAttribute\x12\x1e\n\x04type\x18\x01 \x01(\x0e2\x10.agent.AgentType\x12\x0e\n\x06length\x18\x02 \x01(\x01\x12\r\n\x05width\x18\x03 \x01(\x01\x12\x11\n\tmax_speed\x18\x04 \x01(\x01\x12\x18\n\x10max_acceleration\x18\x05 \x01(\x01\x12 \n\x18max_braking_acceleration\x18\x06 \x01(\x01\x12\x1a\n\x12usual_acceleration\x18\x07 \x01(\x01\x12"\n\x1ausual_braking_acceleration\x18\x08 \x01(\x01"?\n\x10VehicleAttribute\x12\x1a\n\x12lane_change_length\x18\x01 \x01(\x01\x12\x0f\n\x07min_gap\x18\x02 \x01(\x01"\xcd\x01\n\x05Agent\x12\n\n\x02id\x18\x01 \x01(\x05\x12(\n\tattribute\x18\x02 \x01(\x0b2\x15.agent.AgentAttribute\x12\x1b\n\x04home\x18\x03 \x01(\x0b2\r.geo.Position\x12"\n\tschedules\x18\x04 \x03(\x0b2\x0f.agent.Schedule\x127\n\x11vehicle_attribute\x18\x05 \x01(\x0b2\x17.agent.VehicleAttributeH\x00\x88\x01\x01B\x14\n\x12_vehicle_attribute"L\n\x06Agents\x12\x1c\n\x06agents\x18\x01 \x03(\x0b2\x0c.agent.Agent\x12\x16\n\tads_index\x18\x02 \x01(\x05H\x00\x88\x01\x01B\x0c\n\n_ads_index*\x88\x01\n\tAgentType\x12\x1a\n\x16AGENT_TYPE_UNSPECIFIED\x10\x00\x12\x16\n\x12AGENT_TYPE_VEHICLE\x10\x01\x12\x19\n\x15AGENT_TYPE_PEDESTRIAN\x10\x02\x12\x16\n\x12AGENT_TYPE_CYCLIST\x10\x03\x12\x14\n\x10AGENT_TYPE_OTHER\x10\x04b\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'agent.agent_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _AGENTTYPE._serialized_start = 638
    _AGENTTYPE._serialized_end = 774
    _AGENTATTRIBUTE._serialized_start = 62
    _AGENTATTRIBUTE._serialized_end = 284
    _VEHICLEATTRIBUTE._serialized_start = 286
    _VEHICLEATTRIBUTE._serialized_end = 349
    _AGENT._serialized_start = 352
    _AGENT._serialized_end = 557
    _AGENTS._serialized_start = 559
    _AGENTS._serialized_end = 635