cmake_minimum_required(VERSION ${MIN_VER_CMAKE} FATAL_ERROR)

set(HALRVV071_INCLUDE_DIR include)
set(HALRVV071_SOURCE_DIR src)

file(GLOB halrvv071_headers RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "${HALRVV071_INCLUDE_DIR}/*.hpp")
file(GLOB halrvv071_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "${HALRVV071_SOURCE_DIR}/*.cpp")

add_library(halrvv071_hal STATIC)
target_sources(halrvv071_hal PRIVATE ${halrvv071_headers} ${halrvv071_sources})

set_target_properties(halrvv071_hal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})
if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(halrvv071_hal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
target_include_directories(halrvv071_hal PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_SOURCE_DIR}/modules/core/include
     ${CMAKE_SOURCE_DIR}/modules/imgproc/include)


set(HALRVV071_HAL_FOUND TRUE CACHE INTERNAL "")
set(HALRVV071_HAL_VERSION "0.0.1" CACHE INTERNAL "")
set(HALRVV071_HAL_LIBRARIES "halrvv071_hal" CACHE INTERNAL "")
set(HALRVV071_HAL_HEADERS "rvv_hal.hpp" CACHE INTERNAL "")
set(HALRVV071_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "")