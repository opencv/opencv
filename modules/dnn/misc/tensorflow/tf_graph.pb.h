// Forwarding header for the renamed tf_graph.proto.
// Allows tf_io.hpp to use a consistent include name in both PROTOBUF_UPDATE_FILES
// modes: ON (generated from tf_graph.proto) and OFF (pre-generated graph.pb.h).
#include "graph.pb.h"
