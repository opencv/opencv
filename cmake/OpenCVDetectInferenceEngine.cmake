# The script detects Intel(R) OpenVINO(TM) runtime installation
#
# Result:
# - target ocv.3rdparty.openvino

if(WITH_OPENVINO)
  find_package(OpenVINO QUIET)
  if(OpenVINO_FOUND)
    message(STATUS "OpenVINO FOUND: ${OpenVINO_VERSION}")
    math(EXPR ver "${OpenVINO_VERSION_MAJOR} * 1000000 + ${OpenVINO_VERSION_MINOR} * 10000 + ${OpenVINO_VERSION_PATCH} * 100")
    ocv_add_external_target(openvino "" "openvino::runtime" "INF_ENGINE_RELEASE=${ver};HAVE_NGRAPH;HAVE_DNN_NGRAPH;HAVE_INF_ENGINE")
    set(HAVE_OPENVINO 1)
    return()
  endif()
endif()
