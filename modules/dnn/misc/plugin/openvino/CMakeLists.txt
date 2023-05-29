#include_directories("${OPENCV_MODULE_opencv_dnn_BINARY_DIR}")  # Cannot open include file: 'layers/layers_common.simd_declarations.hpp'
ocv_create_builtin_dnn_plugin(opencv_dnn_openvino ocv.3rdparty.openvino ${dnn_plugin_srcs})
