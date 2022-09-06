function(ocv_create_builtin_dnn_plugin name target)

  ocv_debug_message("ocv_create_builtin_dnn_plugin(${ARGV})")

  if(NOT TARGET ${target})
    message(FATAL_ERROR "${target} does not exist!")
  endif()
  if(NOT OpenCV_SOURCE_DIR)
    message(FATAL_ERROR "OpenCV_SOURCE_DIR must be set to build the plugin!")
  endif()

  message(STATUS "DNN: add builtin plugin '${name}'")

  set(ENABLE_PRECOMPILED_HEADERS OFF)  # no support for PCH in plugins, conflicts with module's source files

  # TODO: update CPU optimizations scripts to support plugins
  add_definitions(-D__OPENCV_BUILD=1)
  add_definitions(-DBUILD_PLUGIN=1)
  include_directories("${OPENCV_MODULE_opencv_dnn_BINARY_DIR}")  # Cannot open include file: 'layers/layers_common.simd_declarations.hpp'

  foreach(src ${ARGN})
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/${src}")
      list(APPEND sources "${CMAKE_CURRENT_LIST_DIR}/src/${src}")
    elseif(IS_ABSOLUTE "${src}")
      list(APPEND sources "${src}")
    else()
      message(FATAL_ERROR "Unknown source: ${src}")
    endif()
  endforeach()

  if(OPENCV_MODULE_${the_module}_SOURCES_DISPATCHED)
    list(APPEND sources ${OPENCV_MODULE_${the_module}_SOURCES_DISPATCHED})
  endif()

  set(__${name}_DEPS_EXT "")
  ocv_compiler_optimization_process_sources(sources __${name}_DEPS_EXT ${name})

  add_library(${name} MODULE ${sources})
  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(${name} PRIVATE ${target} ${__${name}_DEPS_EXT})
  target_link_libraries(${name} PRIVATE ${__plugin_libs})

  foreach(mod opencv_dnn
      opencv_core
      opencv_imgproc
      opencv_dnn
  )
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
    CXX_STANDARD 11
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

  add_dependencies(opencv_dnn_plugins ${name})

endfunction()
