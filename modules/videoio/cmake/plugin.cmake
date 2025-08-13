function(ocv_create_builtin_videoio_plugin name target)

  ocv_debug_message("ocv_create_builtin_videoio_plugin(${ARGV})")

  if(NOT TARGET ${target})
    message(FATAL_ERROR "${target} does not exist!")
  endif()
  if(NOT OpenCV_SOURCE_DIR)
    message(FATAL_ERROR "OpenCV_SOURCE_DIR must be set to build the plugin!")
  endif()

  message(STATUS "Video I/O: add builtin plugin '${name}'")

  foreach(src ${ARGN})
    list(APPEND sources "${CMAKE_CURRENT_LIST_DIR}/src/${src}")
  endforeach()

  add_library(${name} MODULE ${sources})
  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_compile_definitions(${name} PRIVATE BUILD_PLUGIN)
  target_link_libraries(${name} PRIVATE ${target})

  foreach(mod opencv_videoio opencv_core opencv_imgproc opencv_imgcodecs)
    ocv_target_link_libraries(${name} LINK_PRIVATE ${mod})
    ocv_target_include_directories(${name} "${OPENCV_MODULE_${mod}_LOCATION}/include")
  endforeach()

  if(WIN32)
    set(OPENCV_PLUGIN_VERSION "${OPENCV_DLLVERSION}" CACHE STRING "")
    if(CMAKE_CXX_SIZEOF_DATA_PTR EQUAL 8)
      set(OPENCV_PLUGIN_ARCH "_64" CACHE STRING "")
    else()
      set(OPENCV_PLUGIN_ARCH "" CACHE STRING "")
    endif()
  else()
    set(OPENCV_PLUGIN_VERSION "" CACHE STRING "")
    set(OPENCV_PLUGIN_ARCH "" CACHE STRING "")
  endif()

  set_target_properties(${name} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    OUTPUT_NAME "${name}${OPENCV_PLUGIN_VERSION}${OPENCV_PLUGIN_ARCH}"
  )

  if(WIN32)
    set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    install(TARGETS ${name} OPTIONAL LIBRARY DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT plugins)
  else()
    install(TARGETS ${name} OPTIONAL LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT plugins)
  endif()

  add_dependencies(opencv_videoio_plugins ${name})

endfunction()
