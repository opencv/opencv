message(STATUS "##########")
message(STATUS "# NDSRVP #")
message(STATUS "##########")

cmake_minimum_required(VERSION ${MIN_VER_CMAKE} FATAL_ERROR)

# toolchain setup

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mext-dsp")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mext-dsp")

# project setup

add_library(ndsrvp_hal STATIC ndsrvp_hal.cpp ndsrvp_hal.hpp)
set_target_properties(ndsrvp_hal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})
if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(ndsrvp_hal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
target_include_directories(ndsrvp_hal PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/modules/core/include
  ${CMAKE_SOURCE_DIR}/modules/imgproc/include
  ${CMAKE_SOURCE_DIR}/modules/features2d/include)

# project info

set(NDSRVP_HAL_FOUND TRUE CACHE INTERNAL "")
set(NDSRVP_HAL_VERSION "0.0.1" CACHE INTERNAL "")
set(NDSRVP_HAL_LIBRARIES "ndsrvp_hal" CACHE INTERNAL "")
set(NDSRVP_HAL_HEADERS "ndsrvp_hal.hpp" CACHE INTERNAL "")
set(NDSRVP_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "")
