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
