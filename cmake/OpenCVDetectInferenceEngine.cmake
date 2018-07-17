# The script detects Intel(R) Inference Engine installation
#
# Parameters:
# INTEL_CVSDK_DIR - Path to Inference Engine root folder
# IE_PLUGINS_PATH - Path to folder with Inference Engine plugins
#
# On return this will define:
#
# HAVE_INF_ENGINE          - True if Intel Inference Engine was found
# INF_ENGINE_INCLUDE_DIRS  - Inference Engine include folder
# INF_ENGINE_LIBRARIES     - Inference Engine libraries and it's dependencies
#
macro(ie_fail)
    set(HAVE_INF_ENGINE FALSE)
    return()
endmacro()


find_package(InferenceEngine QUIET)
if(InferenceEngine_FOUND)
  set(INF_ENGINE_LIBRARIES "${InferenceEngine_LIBRARIES}")
  set(INF_ENGINE_INCLUDE_DIRS "${InferenceEngine_INCLUDE_DIRS}")
  set(INF_ENGINE_VERSION "${InferenceEngine_VERSION}")
  set(HAVE_INF_ENGINE TRUE)
  return()
endif()

ocv_check_environment_variables(INTEL_CVSDK_DIR INF_ENGINE_ROOT_DIR IE_PLUGINS_PATH)

if(NOT INF_ENGINE_ROOT_DIR OR NOT EXISTS "${INF_ENGINE_ROOT_DIR}/include/inference_engine.hpp")
    set(ie_root_paths "${INF_ENGINE_ROOT_DIR}")
    if(DEFINED INTEL_CVSDK_DIR)
        list(APPEND ie_root_paths "${INTEL_CVSDK_DIR}/")
        list(APPEND ie_root_paths "${INTEL_CVSDK_DIR}/deployment_tools/inference_engine")
    endif()

    if(NOT ie_root_paths)
        list(APPEND ie_root_paths "/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/")
    endif()

    find_path(INF_ENGINE_ROOT_DIR include/inference_engine.hpp PATHS ${ie_root_paths})
    if(INF_ENGINE_ROOT_DIR MATCHES "-NOTFOUND$")
      unset(INF_ENGINE_ROOT_DIR CACHE)
    endif()
endif()

set(INF_ENGINE_INCLUDE_DIRS "${INF_ENGINE_ROOT_DIR}/include" CACHE PATH "Path to Inference Engine include directory")

if(NOT INF_ENGINE_ROOT_DIR
    OR NOT EXISTS "${INF_ENGINE_ROOT_DIR}"
    OR NOT EXISTS "${INF_ENGINE_ROOT_DIR}/include/inference_engine.hpp"
)
    message(WARNING "DL IE: Can't detect INF_ENGINE_ROOT_DIR location.")
    ie_fail()
endif()

set(INF_ENGINE_LIBRARIES "")

set(ie_lib_list inference_engine)

if(NOT IS_ABSOLUTE "${IE_PLUGINS_PATH}")
  set(IE_PLUGINS_PATH "${INF_ENGINE_ROOT_DIR}/${IE_PLUGINS_PATH}")
endif()

link_directories(
  ${INF_ENGINE_ROOT_DIR}/external/mkltiny_lnx/lib
  ${INF_ENGINE_ROOT_DIR}/external/cldnn/lib
)

foreach(lib ${ie_lib_list})
    find_library(${lib} NAMES ${lib} HINTS ${IE_PLUGINS_PATH})
    if(NOT ${lib})
        message(WARNING "DL IE: Can't find library: '${lib}'")
        ie_fail()
    endif()
    list(APPEND INF_ENGINE_LIBRARIES ${${lib}})
endforeach()

set(HAVE_INF_ENGINE TRUE)
