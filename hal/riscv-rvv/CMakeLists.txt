cmake_minimum_required(VERSION ${MIN_VER_CMAKE} FATAL_ERROR)

set(RVV_HAL_INCLUDE_DIR include)
set(RVV_HAL_SOURCE_DIR src)

file(GLOB rvv_hal_headers RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "${RVV_HAL_INCLUDE_DIR}/*.hpp")
file(GLOB rvv_hal_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "${RVV_HAL_SOURCE_DIR}/**/*.cpp")

set(HAL_LIB_NAME "rvv_hal")
add_library(${HAL_LIB_NAME} STATIC)
target_sources(${HAL_LIB_NAME} PRIVATE ${rvv_hal_headers} ${rvv_hal_sources})

set_target_properties(${HAL_LIB_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})
if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${HAL_LIB_NAME} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
target_include_directories(${HAL_LIB_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/modules/core/include
  ${CMAKE_SOURCE_DIR}/modules/imgproc/include) #   ${CMAKE_SOURCE_DIR}/modules/features2d/include

set(RVV_HAL_FOUND TRUE CACHE INTERNAL "")
set(RVV_HAL_VERSION "0.0.1" CACHE INTERNAL "")
set(RVV_HAL_LIBRARIES ${HAL_LIB_NAME} CACHE INTERNAL "")
set(RVV_HAL_HEADERS "rvv_hal.hpp" CACHE INTERNAL "")
set(RVV_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "")
