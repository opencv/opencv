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

if(NOT INF_ENGINE_ROOT_DIR OR NOT EXISTS "${INF_ENGINE_ROOT_DIR}/include/inference_engine.hpp")
    set(ie_root_paths "${INF_ENGINE_ROOT_DIR}")
    if(DEFINED ENV{INTEL_CVSDK_DIR})
        list(APPEND ie_root_paths "$ENV{INTEL_CVSDK_DIR}")
        list(APPEND ie_root_paths "$ENV{INTEL_CVSDK_DIR}/inference_engine")
    endif()
    if(DEFINED INTEL_CVSDK_DIR)
        list(APPEND ie_root_paths "${INTEL_CVSDK_DIR}")
        list(APPEND ie_root_paths "${INTEL_CVSDK_DIR}/inference_engine")
    endif()

    if(WITH_INF_ENGINE AND NOT ie_root_paths)
        list(APPEND ie_root_paths "/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/inference_engine")
    endif()

    find_path(INF_ENGINE_ROOT_DIR include/inference_engine.hpp PATHS ${ie_root_paths})
endif()

set(INF_ENGINE_INCLUDE_DIRS "${INF_ENGINE_ROOT_DIR}/include" CACHE PATH "Path to Inference Engine include directory")

if(NOT INF_ENGINE_ROOT_DIR
    OR NOT EXISTS "${INF_ENGINE_ROOT_DIR}"
    OR NOT EXISTS "${INF_ENGINE_INCLUDE_DIRS}"
    OR NOT EXISTS "${INF_ENGINE_INCLUDE_DIRS}/inference_engine.hpp"
)
    ie_fail()
endif()

set(INF_ENGINE_LIBRARIES "")

set(ie_lib_list inference_engine)
if(UNIX)
    list(APPEND ie_lib_list mklml_intel iomp5)
endif()

foreach(lib ${ie_lib_list})
    find_library(${lib}
        NAMES ${lib}
        # For inference_engine
        HINTS ${IE_PLUGINS_PATH}
        HINTS "$ENV{IE_PLUGINS_PATH}"
        # For mklml_intel, iomp5
        HINTS ${INTEL_CVSDK_DIR}/external/mklml_lnx/lib
        HINTS ${INTEL_CVSDK_DIR}/inference_engine/external/mklml_lnx/lib
    )
    if(NOT ${lib})
        ie_fail()
    endif()
    list(APPEND INF_ENGINE_LIBRARIES ${${lib}})
endforeach()

set(HAVE_INF_ENGINE TRUE)

include_directories(${INF_ENGINE_INCLUDE_DIRS})
list(APPEND OPENCV_LINKER_LIBS ${INF_ENGINE_LIBRARIES})
add_definitions(-DHAVE_INF_ENGINE)
