# Locate AMD MIGraphX (graph-compiler inference engine) for the DNN MIGraphX backend.
# Supported system: UNIX (ROCm/MIGraphX is Linux-only).
if(NOT UNIX)
  set(HAVE_MIGRAPHX OFF)
  message(WARNING "MIGraphX: ROCm/MIGraphX supports unix but not ${CMAKE_SYSTEM_NAME}. Turning off HAVE_MIGRAPHX")
  return()
endif()
ocv_check_environment_variables(MIGRAPHX_ROOT)
if(NOT MIGRAPHX_ROOT AND DEFINED ENV{ROCM_PATH})
  set(MIGRAPHX_ROOT "$ENV{ROCM_PATH}")
endif()
if(NOT MIGRAPHX_ROOT)
  set(MIGRAPHX_ROOT "/opt/rocm")
endif()

find_path(MIGRAPHX_INCLUDE_DIR
  NAMES migraphx/migraphx.hpp
  HINTS "${MIGRAPHX_ROOT}/include" /opt/rocm/include)
find_library(MIGRAPHX_C_LIBRARY
  NAMES migraphx_c
  HINTS "${MIGRAPHX_ROOT}/lib" /opt/rocm/lib)

if(MIGRAPHX_INCLUDE_DIR AND MIGRAPHX_C_LIBRARY)
  set(HAVE_MIGRAPHX ON)
  set(MIGRAPHX_INCLUDE_DIRS "${MIGRAPHX_INCLUDE_DIR}")
  set(MIGRAPHX_LIBRARIES "${MIGRAPHX_C_LIBRARY}")
  message(STATUS "MIGraphX: found (include=${MIGRAPHX_INCLUDE_DIR}, lib=${MIGRAPHX_C_LIBRARY})")
else()
  set(HAVE_MIGRAPHX OFF)
  message(STATUS "MIGraphX: NOT found (need migraphx/migraphx.hpp + libmigraphx_c). Set MIGRAPHX_ROOT or ROCM_PATH.")
endif()

mark_as_advanced(MIGRAPHX_INCLUDE_DIR MIGRAPHX_C_LIBRARY)
