# The script detects Intel(R) Inference Engine installation
#
# Cache variables:
# INF_ENGINE_RELEASE - a number reflecting IE source interface (linked with OpenVINO release)
#
# Detect parameters:
# 1. Native cmake IE package:
#    - environment variable InferenceEngine_DIR is set to location of cmake module
# 2. Custom location:
#    - INF_ENGINE_INCLUDE_DIRS - headers search location
#    - INF_ENGINE_LIB_DIRS     - library search location
# 3. OpenVINO location:
#    - environment variable INTEL_OPENVINO_DIR is set to location of OpenVINO installation dir
#    - INF_ENGINE_PLATFORM - part of name of library directory representing its platform
#
# Result:
# INF_ENGINE_TARGET - set to name of imported library target representing InferenceEngine
#

function(add_custom_ie_build _inc _lib _lib_rel _lib_dbg _msg)
  if(NOT _inc OR NOT (_lib OR _lib_rel OR _lib_dbg))
    return()
  endif()
  add_library(inference_engine UNKNOWN IMPORTED)
  set_target_properties(inference_engine PROPERTIES
    IMPORTED_LOCATION "${_lib}"
    IMPORTED_IMPLIB_RELEASE "${_lib_rel}"
    IMPORTED_IMPLIB_DEBUG "${_lib_dbg}"
    INTERFACE_INCLUDE_DIRECTORIES "${_inc}"
  )

  find_library(ie_builder_custom_lib "inference_engine_nn_builder" PATHS "${INF_ENGINE_LIB_DIRS}" NO_DEFAULT_PATH)
  if(EXISTS "${ie_builder_custom_lib}")
    add_library(inference_engine_nn_builder UNKNOWN IMPORTED)
    set_target_properties(inference_engine_nn_builder PROPERTIES
      IMPORTED_LOCATION "${ie_builder_custom_lib}"
    )
  endif()

  if(NOT INF_ENGINE_RELEASE VERSION_GREATER "2018050000")
    find_library(INF_ENGINE_OMP_LIBRARY iomp5 PATHS "${INF_ENGINE_OMP_DIR}" NO_DEFAULT_PATH)
    if(NOT INF_ENGINE_OMP_LIBRARY)
      message(WARNING "OpenMP for IE have not been found. Set INF_ENGINE_OMP_DIR variable if you experience build errors.")
    else()
      set_target_properties(inference_engine PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES "${INF_ENGINE_OMP_LIBRARY}")
    endif()
  endif()
  set(INF_ENGINE_VERSION "Unknown" CACHE STRING "")
  set(INF_ENGINE_TARGET inference_engine)
  if(TARGET inference_engine_nn_builder)
    list(APPEND INF_ENGINE_TARGET inference_engine_nn_builder)
    set(_msg "${_msg}, with IE NN Builder API")
  endif()
  set(INF_ENGINE_TARGET "${INF_ENGINE_TARGET}" PARENT_SCOPE)
  message(STATUS "Detected InferenceEngine: ${_msg}")
endfunction()

# ======================

find_package(InferenceEngine QUIET)
if(InferenceEngine_FOUND)
  set(INF_ENGINE_TARGET ${InferenceEngine_LIBRARIES})
  set(INF_ENGINE_VERSION "${InferenceEngine_VERSION}" CACHE STRING "")
  message(STATUS "Detected InferenceEngine: cmake package (${InferenceEngine_VERSION})")
endif()

if(NOT INF_ENGINE_TARGET AND INF_ENGINE_LIB_DIRS AND INF_ENGINE_INCLUDE_DIRS)
  find_path(ie_custom_inc "inference_engine.hpp" PATHS "${INF_ENGINE_INCLUDE_DIRS}" NO_DEFAULT_PATH)
  find_library(ie_custom_lib "inference_engine" PATHS "${INF_ENGINE_LIB_DIRS}" NO_DEFAULT_PATH)
  find_library(ie_custom_lib_rel "inference_engine" PATHS "${INF_ENGINE_LIB_DIRS}/Release" NO_DEFAULT_PATH)
  find_library(ie_custom_lib_dbg "inference_engine" PATHS "${INF_ENGINE_LIB_DIRS}/Debug" NO_DEFAULT_PATH)
  add_custom_ie_build("${ie_custom_inc}" "${ie_custom_lib}" "${ie_custom_lib_rel}" "${ie_custom_lib_dbg}" "INF_ENGINE_{INCLUDE,LIB}_DIRS")
endif()

set(_loc "$ENV{INTEL_OPENVINO_DIR}")
if(NOT _loc AND DEFINED ENV{INTEL_CVSDK_DIR})
  set(_loc "$ENV{INTEL_CVSDK_DIR}")  # OpenVINO 2018.x
endif()
if(NOT INF_ENGINE_TARGET AND _loc)
  if(NOT INF_ENGINE_RELEASE VERSION_GREATER "2018050000")
    set(INF_ENGINE_PLATFORM_DEFAULT "ubuntu_16.04")
  else()
    set(INF_ENGINE_PLATFORM_DEFAULT "")
  endif()
  set(INF_ENGINE_PLATFORM "${INF_ENGINE_PLATFORM_DEFAULT}" CACHE STRING "InferenceEngine platform (library dir)")
  find_path(ie_custom_env_inc "inference_engine.hpp" PATHS "${_loc}/deployment_tools/inference_engine/include" NO_DEFAULT_PATH)
  find_library(ie_custom_env_lib "inference_engine" PATHS "${_loc}/deployment_tools/inference_engine/lib/${INF_ENGINE_PLATFORM}/intel64" NO_DEFAULT_PATH)
  find_library(ie_custom_env_lib_rel "inference_engine" PATHS "${_loc}/deployment_tools/inference_engine/lib/intel64/Release" NO_DEFAULT_PATH)
  find_library(ie_custom_env_lib_dbg "inference_engine" PATHS "${_loc}/deployment_tools/inference_engine/lib/intel64/Debug" NO_DEFAULT_PATH)
  add_custom_ie_build("${ie_custom_env_inc}" "${ie_custom_env_lib}" "${ie_custom_env_lib_rel}" "${ie_custom_env_lib_dbg}" "OpenVINO (${_loc})")
endif()

# Add more features to the target

if(INF_ENGINE_TARGET)
  if(NOT INF_ENGINE_RELEASE)
    message(WARNING "InferenceEngine version have not been set, 2019R3 will be used by default. Set INF_ENGINE_RELEASE variable if you experience build errors.")
  endif()
  set(INF_ENGINE_RELEASE "2019030000" CACHE STRING "Force IE version, should be in form YYYYAABBCC (e.g. 2018R2.0.2 -> 2018020002)")
  set_target_properties(${INF_ENGINE_TARGET} PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "HAVE_INF_ENGINE=1;INF_ENGINE_RELEASE=${INF_ENGINE_RELEASE}"
  )
endif()

if(WITH_NGRAPH)
  find_package(ngraph QUIET)
  if(ngraph_FOUND)
    ocv_assert(TARGET ngraph::ngraph)
    if(INF_ENGINE_RELEASE VERSION_LESS "2019039999")
      message(WARNING "nGraph is not tested with current InferenceEngine version: INF_ENGINE_RELEASE=${INF_ENGINE_RELEASE}")
    endif()
    message(STATUS "Detected ngraph: cmake package (${ngraph_VERSION})")
    set(HAVE_NGRAPH ON)
  endif()
endif()
