message(STATUS "##########")
message(STATUS "# NDSRVP #")
message(STATUS "##########")

cmake_minimum_required(VERSION ${MIN_VER_CMAKE} FATAL_ERROR)

# project setup

set(NDSRVP_INCLUDE_DIR include)
set(NDSRVP_SOURCE_DIR src)

file(GLOB ndsrvp_headers RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "${NDSRVP_INCLUDE_DIR}/*.hpp")
file(GLOB ndsrvp_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "${NDSRVP_SOURCE_DIR}/*.cpp")

add_library(ndsrvp_hal STATIC)
target_sources(ndsrvp_hal PRIVATE ${ndsrvp_headers} ${ndsrvp_sources})

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
