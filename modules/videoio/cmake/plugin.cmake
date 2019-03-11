#=============================================
# build with OpenCV

function(ocv_create_builtin_videoio_plugin name target videoio_src_file)

  ocv_debug_message("ocv_create_builtin_videoio_plugin(${ARGV})")

  if(NOT TARGET ${target})
    message(FATAL_ERROR "${target} does not exist!")
  endif()
  if(NOT OpenCV_SOURCE_DIR)
    message(FATAL_ERROR "OpenCV_SOURCE_DIR must be set to build the plugin!")
  endif()

  message(STATUS "Video I/O: add builtin plugin '${name}'")

  add_library(${name} MODULE
    "${CMAKE_CURRENT_LIST_DIR}/src/${videoio_src_file}"
  )
  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_compile_definitions(${name} PRIVATE BUILD_PLUGIN)
  target_link_libraries(${name} PRIVATE ${target})

  foreach(mod opencv_videoio opencv_core opencv_imgproc opencv_imgcodecs)
    target_link_libraries(${name} PRIVATE ${mod})
    target_include_directories(${name} PRIVATE "${OPENCV_MODULE_${mod}_LOCATION}/include")
  endforeach()

  set_target_properties(${name} PROPERTIES
    CXX_STANDARD 11
    CXX_VISIBILITY_PRESET hidden
  )
  install(TARGETS ${name} LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT plugins OPTIONAL)

  add_dependencies(opencv_videoio_plugins ${name})

endfunction()

#=============================================
# standalone build

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

  set(modules_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../..")
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

  # Fixes for build
  target_compile_definitions(${OPENCV_PLUGIN_NAME} PRIVATE __OPENCV_BUILD)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cvconfig.h" "#pragma once")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cv_cpu_config.h" "#pragma once")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/opencv2/opencv_modules.hpp" "#pragma once")

  target_link_libraries(${OPENCV_PLUGIN_NAME} PRIVATE ${target})
  set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_VISIBILITY_PRESET hidden
  )

  # Hack for Windows
  if(WIN32)
    find_package(OpenCV REQUIRED core imgproc videoio)
    target_link_libraries(${OPENCV_PLUGIN_NAME} ${OpenCV_LIBS})
  endif()

  if(OPENCV_PLUGIN_DESTINATION)
    set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OPENCV_PLUGIN_DESTINATION}")
    message(STATUS "Output destination: ${OPENCV_PLUGIN_DESTINATION}")
  endif()

  message(STATUS "Library name: ${OPENCV_PLUGIN_NAME}")

endfunction()
