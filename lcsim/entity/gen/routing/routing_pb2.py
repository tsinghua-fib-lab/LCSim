"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x15routing/routing.proto\x12\x07routing"3\n\x12DrivingJourneyBody\x12\x10\n\x08road_ids\x18\x02 \x03(\x05\x12\x0b\n\x03eta\x18\x03 \x01(\x01"Z\n\x13WalkingRouteSegment\x12\x0f\n\x07lane_id\x18\x01 \x01(\x05\x122\n\x10moving_direction\x18\x02 \x01(\x0e2\x18.routing.MovingDirection"N\n\x12WalkingJourneyBody\x12+\n\x05route\x18\x01 \x03(\x0b2\x1c.routing.WalkingRouteSegment\x12\x0b\n\x03eta\x18\x02 \x01(\x01"S\n\x0eBusJourneyBody\x12\x0f\n\x07line_id\x18\x01 \x01(\x05\x12\x18\n\x10start_station_id\x18\x02 \x01(\x05\x12\x16\n\x0eend_station_id\x18\x03 \x01(\x05"\xe4\x01\n\x07Journey\x12"\n\x04type\x18\x01 \x01(\x0e2\x14.routing.JourneyType\x121\n\x07driving\x18\x02 \x01(\x0b2\x1b.routing.DrivingJourneyBodyH\x00\x88\x01\x01\x121\n\x07walking\x18\x03 \x01(\x0b2\x1b.routing.WalkingJourneyBodyH\x01\x88\x01\x01\x12,\n\x06by_bus\x18\x04 \x01(\x0b2\x17.routing.BusJourneyBodyH\x02\x88\x01\x01B\n\n\x08_drivingB\n\n\x08_walkingB\t\n\x07_by_bus"]\n\x07BusLine\x12\x0f\n\x07line_id\x18\x01 \x01(\x05\x12\r\n\x05stops\x18\x02 \x03(\x05\x12\x11\n\tdistances\x18\x03 \x03(\x01\x12\x10\n\x08interval\x18\x04 \x01(\x05\x12\r\n\x05count\x18\x05 \x01(\x05"+\n\x08BusLines\x12\x1f\n\x05lines\x18\x01 \x03(\x0b2\x10.routing.BusLine"\'\n\nRoadStatus\x12\n\n\x02id\x18\x01 \x01(\x05\x12\r\n\x05speed\x18\x02 \x03(\x01":\n\x0cRoadStatuses\x12*\n\rroad_statuses\x18\x01 \x03(\x0b2\x13.routing.RoadStatus*n\n\tRouteType\x12\x1a\n\x16ROUTE_TYPE_UNSPECIFIED\x10\x00\x12\x16\n\x12ROUTE_TYPE_DRIVING\x10\x01\x12\x16\n\x12ROUTE_TYPE_WALKING\x10\x02\x12\x15\n\x11ROUTE_TYPE_BY_BUS\x10\x03*x\n\x0bJourneyType\x12\x1c\n\x18JOURNEY_TYPE_UNSPECIFIED\x10\x00\x12\x18\n\x14JOURNEY_TYPE_DRIVING\x10\x01\x12\x18\n\x14JOURNEY_TYPE_WALKING\x10\x02\x12\x17\n\x13JOURNEY_TYPE_BY_BUS\x10\x03*p\n\x0fMovingDirection\x12 \n\x1cMOVING_DIRECTION_UNSPECIFIED\x10\x00\x12\x1c\n\x18MOVING_DIRECTION_FORWARD\x10\x01\x12\x1d\n\x19MOVING_DIRECTION_BACKWARD\x10\x02b\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'routing.routing_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _ROUTETYPE._serialized_start = 816
    _ROUTETYPE._serialized_end = 926
    _JOURNEYTYPE._serialized_start = 928
    _JOURNEYTYPE._serialized_end = 1048
    _MOVINGDIRECTION._serialized_start = 1050
    _MOVINGDIRECTION._serialized_end = 1162
    _DRIVINGJOURNEYBODY._serialized_start = 34
    _DRIVINGJOURNEYBODY._serialized_end = 85
    _WALKINGROUTESEGMENT._serialized_start = 87
    _WALKINGROUTESEGMENT._serialized_end = 177
    _WALKINGJOURNEYBODY._serialized_start = 179
    _WALKINGJOURNEYBODY._serialized_end = 257
    _BUSJOURNEYBODY._serialized_start = 259
    _BUSJOURNEYBODY._serialized_end = 342
    _JOURNEY._serialized_start = 345
    _JOURNEY._serialized_end = 573
    _BUSLINE._serialized_start = 575
    _BUSLINE._serialized_end = 668
    _BUSLINES._serialized_start = 670
    _BUSLINES._serialized_end = 713
    _ROADSTATUS._serialized_start = 715
    _ROADSTATUS._serialized_end = 754
    _ROADSTATUSES._serialized_start = 756
    _ROADSTATUSES._serialized_end = 814