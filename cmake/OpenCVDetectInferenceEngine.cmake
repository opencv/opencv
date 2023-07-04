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

# ======================

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

# ======================

find_package(InferenceEngine QUIET)
if(InferenceEngine_FOUND)
  set(INF_ENGINE_TARGET ${InferenceEngine_LIBRARIES})
  set(INF_ENGINE_VERSION "${InferenceEngine_VERSION}")
  message(STATUS "Detected InferenceEngine: cmake package (${InferenceEngine_VERSION})")
endif()

if(DEFINED InferenceEngine_VERSION)
  message(STATUS "InferenceEngine: ${InferenceEngine_VERSION}")
  if(NOT INF_ENGINE_RELEASE AND NOT (InferenceEngine_VERSION VERSION_LESS "2021.4"))
    math(EXPR INF_ENGINE_RELEASE_INIT "${InferenceEngine_VERSION_MAJOR} * 1000000 + ${InferenceEngine_VERSION_MINOR} * 10000 + ${InferenceEngine_VERSION_PATCH} * 100")
  endif()
endif()
if(NOT INF_ENGINE_RELEASE AND NOT INF_ENGINE_RELEASE_INIT)
  message(STATUS "WARNING: InferenceEngine version has not been set, 2021.4.2 will be used by default. Set INF_ENGINE_RELEASE variable if you experience build errors.")
  set(INF_ENGINE_RELEASE_INIT "2021040200")
elseif(DEFINED INF_ENGINE_RELEASE)
  set(INF_ENGINE_RELEASE_INIT "${INF_ENGINE_RELEASE}")
endif()
set(INF_ENGINE_RELEASE "${INF_ENGINE_RELEASE_INIT}" CACHE STRING "Force IE version, should be in form YYYYAABBCC (e.g. 2020.1.0.2 -> 2020010002)")

set(tgts)
set(defs)

# Add more features to the target
if(INF_ENGINE_TARGET)
  set_target_properties(${INF_ENGINE_TARGET} PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS "HAVE_INF_ENGINE=1;INF_ENGINE_RELEASE=${INF_ENGINE_RELEASE}"
  )
  list(APPEND tgts ${INF_ENGINE_TARGET})
  list(APPEND defs "INF_ENGINE_RELEASE=${INF_ENGINE_RELEASE}" "HAVE_INF_ENGINE")
endif()

if(WITH_NGRAPH OR NOT DEFINED WITH_NGRAPH)
  find_package(ngraph QUIET)
  if(ngraph_FOUND)
    ocv_assert(TARGET ngraph::ngraph)
    if(INF_ENGINE_RELEASE VERSION_LESS "2019039999")
      message(WARNING "nGraph is not tested with current InferenceEngine version: INF_ENGINE_RELEASE=${INF_ENGINE_RELEASE}")
    endif()
    message(STATUS "Detected ngraph: cmake package (${ngraph_VERSION})")
    set(HAVE_NGRAPH ON)
    list(APPEND tgts ngraph::ngraph)
    list(APPEND defs "HAVE_NGRAPH" "HAVE_DNN_NGRAPH")
  endif()
endif()

ocv_add_external_target(openvino "" "${tgts}" "${defs}")
