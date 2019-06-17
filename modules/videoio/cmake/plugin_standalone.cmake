#=============================================
# standalone build

include(FindPkgConfig)

#=============================================
# build with OpenCV
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVUtils.cmake")

function(ocv_create_videoio_plugin default_name target target_desc videoio_src_file)

  set(OPENCV_PLUGIN_NAME ${default_name} CACHE STRING "")
  set(OPENCV_PLUGIN_DESTINATION "" CACHE PATH "")
  project(${OPENCV_PLUGIN_NAME} LANGUAGES CXX)

  set(BUILD_SHARED_LIBS ON CACHE BOOL "")
  if(NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "Static plugin build does not make sense")
  endif()

  if(NOT OpenCV_SOURCE_DIR)
    message(FATAL_ERROR "OpenCV_SOURCE_DIR must be set to build the plugin!")
  endif()

  include("${OpenCV_SOURCE_DIR}/modules/videoio/cmake/init.cmake")

  if(NOT TARGET ${target})
    message(FATAL_ERROR "${target_desc} was not found!")
  endif()

  get_filename_component(modules_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
  set(videoio_ROOT "${modules_ROOT}/videoio")
  set(core_ROOT "${modules_ROOT}/core")
  set(imgproc_ROOT "${modules_ROOT}/imgproc")
  set(imgcodecs_ROOT "${modules_ROOT}/imgcodecs")

  add_library(${OPENCV_PLUGIN_NAME} MODULE "${videoio_ROOT}/src/${videoio_src_file}")
  target_include_directories(${OPENCV_PLUGIN_NAME} PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${videoio_ROOT}/src"
    "${videoio_ROOT}/include"
    "${core_ROOT}/include"
    "${imgproc_ROOT}/include"
    "${imgcodecs_ROOT}/include"
  )
  target_compile_definitions(${OPENCV_PLUGIN_NAME} PRIVATE BUILD_PLUGIN)

  target_link_libraries(${OPENCV_PLUGIN_NAME} PRIVATE ${target})
  set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_VISIBILITY_PRESET hidden
  )

  if(DEFINED OPENCV_PLUGIN_MODULE_PREFIX)
    set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES PREFIX "${OPENCV_PLUGIN_MODULE_PREFIX}")
  endif()

  # Hack for Windows
  if(WIN32)
    find_package(OpenCV REQUIRED core imgproc videoio)
    target_link_libraries(${OPENCV_PLUGIN_NAME} PRIVATE ${OpenCV_LIBS})
  endif()

  if(NOT OpenCV_FOUND)  # build against sources (Linux)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/opencv2/opencv_modules.hpp" "#pragma once")
  endif()

  if(OPENCV_PLUGIN_DESTINATION)
    set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OPENCV_PLUGIN_DESTINATION}")
    message(STATUS "Output destination: ${OPENCV_PLUGIN_DESTINATION}")
  endif()

  install(TARGETS ${OPENCV_PLUGIN_NAME} LIBRARY DESTINATION . COMPONENT plugins)

  message(STATUS "Library name: ${OPENCV_PLUGIN_NAME}")

endfunction()
