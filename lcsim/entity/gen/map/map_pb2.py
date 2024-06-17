"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
from ..geo import geo_pb2 as geo_dot_geo__pb2
from ..map import light_pb2 as map_dot_light__pb2
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rmap/map.proto\x12\x03map\x1a\rgeo/geo.proto\x1a\x0fmap/light.proto"*\n\x08Polyline\x12\x1e\n\x05nodes\x18\x01 \x03(\x0b2\x0f.geo.XYPosition"r\n\x06Header\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04date\x18\x02 \x01(\t\x12\r\n\x05north\x18\x03 \x01(\x01\x12\r\n\x05south\x18\x04 \x01(\x01\x12\x0c\n\x04east\x18\x05 \x01(\x01\x12\x0c\n\x04west\x18\x06 \x01(\x01\x12\x12\n\nprojection\x18\x07 \x01(\t"d\n\x0bLaneOverlap\x12\x1f\n\x04self\x18\x01 \x01(\x0b2\x11.geo.LanePosition\x12 \n\x05other\x18\x02 \x01(\x0b2\x11.geo.LanePosition\x12\x12\n\nself_first\x18\x03 \x01(\x08"C\n\x0eLaneConnection\x12\n\n\x02id\x18\x01 \x01(\x05\x12%\n\x04type\x18\x02 \x01(\x0e2\x17.map.LaneConnectionType"\xe6\x03\n\x04Lane\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x1b\n\x04type\x18\x02 \x01(\x0e2\r.map.LaneType\x12\x1b\n\x04turn\x18\x03 \x01(\x0e2\r.map.LaneTurn\x12\x11\n\tmax_speed\x18\x04 \x01(\x01\x12\x0e\n\x06length\x18\x05 \x01(\x01\x12\r\n\x05width\x18\x06 \x01(\x01\x12"\n\x0bcenter_line\x18\x07 \x01(\x0b2\r.map.Polyline\x12+\n\x10left_border_line\x18\x08 \x01(\x0b2\r.map.PolylineB\x02\x18\x01\x12,\n\x11right_border_line\x18\t \x01(\x0b2\r.map.PolylineB\x02\x18\x01\x12)\n\x0cpredecessors\x18\n \x03(\x0b2\x13.map.LaneConnection\x12\'\n\nsuccessors\x18\x0b \x03(\x0b2\x13.map.LaneConnection\x12\x15\n\rleft_lane_ids\x18\x0c \x03(\x05\x12\x16\n\x0eright_lane_ids\x18\r \x03(\x05\x12\x11\n\tparent_id\x18\x0e \x01(\x05\x12"\n\x08overlaps\x18\x0f \x03(\x0b2\x10.map.LaneOverlap\x12-\n\x10center_line_type\x18\x10 \x01(\x0e2\x13.map.CenterLineType"E\n\x0cNextRoadLane\x12\x0f\n\x07road_id\x18\x01 \x01(\x05\x12\x11\n\tlane_id_a\x18\x02 \x01(\x05\x12\x11\n\tlane_id_b\x18\x03 \x01(\x05">\n\x10NextRoadLanePlan\x12*\n\x0fnext_road_lanes\x18\x01 \x03(\x0b2\x11.map.NextRoadLane"K\n\x08RoadLine\x12\x1f\n\x04type\x18\x01 \x01(\x0e2\x11.map.RoadLineType\x12\x1e\n\x05nodes\x18\x02 \x03(\x0b2\x0f.geo.XYPosition"b\n\x0eRoadConnection\x12\x1f\n\x04type\x18\x01 \x01(\x0e2\x11.map.RoadLineType\x12\x1e\n\x05nodes\x18\x02 \x03(\x0b2\x0f.geo.XYPosition\x12\x0f\n\x07road_id\x18\x03 \x01(\x05"K\n\x08RoadEdge\x12\x1f\n\x04type\x18\x01 \x01(\x0e2\x11.map.RoadEdgeType\x12\x1e\n\x05nodes\x18\x02 \x03(\x0b2\x0f.geo.XYPosition"\xdc\x01\n\x04Road\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x0c\n\x04name\x18\x04 \x01(\t\x12\x10\n\x08lane_ids\x18\x02 \x03(\x05\x123\n\x14next_road_lane_plans\x18\x03 \x03(\x0b2\x15.map.NextRoadLanePlan\x12!\n\nroad_lines\x18\x05 \x03(\x0b2\r.map.RoadLine\x12!\n\nroad_edges\x18\x06 \x03(\x0b2\r.map.RoadEdge\x12-\n\x10road_connections\x18\x07 \x03(\x0b2\x13.map.RoadConnection"\x90\x01\n\x11JunctionLaneGroup\x12\x12\n\nin_road_id\x18\x01 \x01(\x05\x12\x10\n\x08in_angle\x18\x02 \x01(\x01\x12\x13\n\x0bout_road_id\x18\x03 \x01(\x05\x12\x11\n\tout_angle\x18\x04 \x01(\x01\x12\x10\n\x08lane_ids\x18\x05 \x03(\x05\x12\x1b\n\x04turn\x18\x06 \x01(\x0e2\r.map.LaneTurn"\x89\x02\n\x08Junction\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x10\n\x08lane_ids\x18\x02 \x03(\x05\x123\n\x13driving_lane_groups\x18\x03 \x03(\x0b2\x16.map.JunctionLaneGroup\x12#\n\x06phases\x18\x04 \x03(\x0b2\x13.map.AvailablePhase\x12-\n\rfixed_program\x18\x05 \x01(\x0b2\x11.map.TrafficLightH\x00\x88\x01\x01\x12!\n\nroad_lines\x18\x06 \x03(\x0b2\r.map.RoadLine\x12!\n\nroad_edges\x18\x07 \x03(\x0b2\r.map.RoadEdgeB\x10\n\x0e_fixed_program"x\n\x03Map\x12\x1b\n\x06header\x18\x01 \x01(\x0b2\x0b.map.Header\x12\x18\n\x05lanes\x18\x02 \x03(\x0b2\t.map.Lane\x12\x18\n\x05roads\x18\x03 \x03(\x0b2\t.map.Road\x12 \n\tjunctions\x18\x04 \x03(\x0b2\r.map.Junction*S\n\x08LaneType\x12\x19\n\x15LANE_TYPE_UNSPECIFIED\x10\x00\x12\x15\n\x11LANE_TYPE_DRIVING\x10\x01\x12\x15\n\x11LANE_TYPE_WALKING\x10\x02*|\n\x08LaneTurn\x12\x19\n\x15LANE_TURN_UNSPECIFIED\x10\x00\x12\x16\n\x12LANE_TURN_STRAIGHT\x10\x01\x12\x12\n\x0eLANE_TURN_LEFT\x10\x02\x12\x13\n\x0fLANE_TURN_RIGHT\x10\x03\x12\x14\n\x10LANE_TURN_AROUND\x10\x04*x\n\x12LaneConnectionType\x12$\n LANE_CONNECTION_TYPE_UNSPECIFIED\x10\x00\x12\x1d\n\x19LANE_CONNECTION_TYPE_HEAD\x10\x01\x12\x1d\n\x19LANE_CONNECTION_TYPE_TAIL\x10\x02*\x90\x01\n\x0eCenterLineType\x12 \n\x1cCENTER_LINE_TYPE_UNSPECIFIED\x10\x00\x12\x1c\n\x18CENTER_LINE_TYPE_FREEWAY\x10\x01\x12#\n\x1fCENTER_LINE_TYPE_SURFACE_STREET\x10\x02\x12\x19\n\x15CENTER_LINE_TYPE_BIKE\x10\x03*\xf0\x02\n\x0cRoadLineType\x12\x1e\n\x1aROAD_LINE_TYPE_UNSPECIFIED\x10\x00\x12&\n"ROAD_LINE_TYPE_BROKEN_SINGLE_WHITE\x10\x01\x12%\n!ROAD_LINE_TYPE_SOLID_SINGLE_WHITE\x10\x02\x12%\n!ROAD_LINE_TYPE_SOLID_DOUBLE_WHITE\x10\x03\x12\'\n#ROAD_LINE_TYPE_BROKEN_SINGLE_YELLOW\x10\x04\x12\'\n#ROAD_LINE_TYPE_BROKEN_DOUBLE_YELLOW\x10\x05\x12&\n"ROAD_LINE_TYPE_SOLID_SINGLE_YELLOW\x10\x06\x12&\n"ROAD_LINE_TYPE_SOLID_DOUBLE_YELLOW\x10\x07\x12(\n$ROAD_LINE_TYPE_PASSING_DOUBLE_YELLOW\x10\x08*f\n\x0cRoadEdgeType\x12\x1e\n\x1aROAD_EDGE_TYPE_UNSPECIFIED\x10\x00\x12\x1b\n\x17ROAD_EDGE_TYPE_BOUNDARY\x10\x01\x12\x19\n\x15ROAD_EDGE_TYPE_MEDIAN\x10\x02b\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'map.map_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _LANE.fields_by_name['left_border_line']._options = None
    _LANE.fields_by_name['left_border_line']._serialized_options = b'\x18\x01'
    _LANE.fields_by_name['right_border_line']._options = None
    _LANE.fields_by_name['right_border_line']._serialized_options = b'\x18\x01'
    _LANETYPE._serialized_start = 2023
    _LANETYPE._serialized_end = 2106
    _LANETURN._serialized_start = 2108
    _LANETURN._serialized_end = 2232
    _LANECONNECTIONTYPE._serialized_start = 2234
    _LANECONNECTIONTYPE._serialized_end = 2354
    _CENTERLINETYPE._serialized_start = 2357
    _CENTERLINETYPE._serialized_end = 2501
    _ROADLINETYPE._serialized_start = 2504
    _ROADLINETYPE._serialized_end = 2872
    _ROADEDGETYPE._serialized_start = 2874
    _ROADEDGETYPE._serialized_end = 2976
    _POLYLINE._serialized_start = 54
    _POLYLINE._serialized_end = 96
    _HEADER._serialized_start = 98
    _HEADER._serialized_end = 212
    _LANEOVERLAP._serialized_start = 214
    _LANEOVERLAP._serialized_end = 314
    _LANECONNECTION._serialized_start = 316
    _LANECONNECTION._serialized_end = 383
    _LANE._serialized_start = 386
    _LANE._serialized_end = 872
    _NEXTROADLANE._serialized_start = 874
    _NEXTROADLANE._serialized_end = 943
    _NEXTROADLANEPLAN._serialized_start = 945
    _NEXTROADLANEPLAN._serialized_end = 1007
    _ROADLINE._serialized_start = 1009
    _ROADLINE._serialized_end = 1084
    _ROADCONNECTION._serialized_start = 1086
    _ROADCONNECTION._serialized_end = 1184
    _ROADEDGE._serialized_start = 1186
    _ROADEDGE._serialized_end = 1261
    _ROAD._serialized_start = 1264
    _ROAD._serialized_end = 1484
    _JUNCTIONLANEGROUP._serialized_start = 1487
    _JUNCTIONLANEGROUP._serialized_end = 1631
    _JUNCTION._serialized_start = 1634
    _JUNCTION._serialized_end = 1899
    _MAP._serialized_start = 1901
    _MAP._serialized_end = 2021