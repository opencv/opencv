if(OPENCV_INITIAL_PASS)
  # generator for Objective-C source code and documentation signatures
  add_subdirectory(generator)
endif()

if(NOT APPLE_FRAMEWORK OR NOT BUILD_OBJC OR NOT PYTHON_DEFAULT_AVAILABLE)
  ocv_module_disable(objc)
endif()

set(the_description "The Objective-C bindings")
ocv_add_module(objc BINDINGS opencv_core opencv_imgproc PRIVATE_REQUIRED opencv_objc_bindings_generator)

include(${CMAKE_CURRENT_SOURCE_DIR}/common.cmake)


# UTILITY: copy common Objective-C test files and add them to _deps
# copy_common_tests(<source-folder> <destination-folder> <variable-to-store-deps>)
macro(copy_common_tests _src_location _dst_location _deps)
  set(_src ${_src_location})
  set(_dst ${_dst_location})
  file(GLOB_RECURSE _files RELATIVE "${_src}" "${_src}/src/*")
  foreach(f ${_files})
    add_custom_command(
        OUTPUT "${_dst}/${f}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}/${f}" "${_dst}/${f}"
        MAIN_DEPENDENCY "${_src}/${f}"
        COMMENT "Copying ${f}")
    list(APPEND ${_deps} "${_src}/${f}" "${_dst}/${f}")
  endforeach()
  unset(_files)
  unset(_src)
  unset(_dst)
endmacro()


add_subdirectory(objc_build)  # generates ${the_module} target 

if(BUILD_TESTS)
  add_subdirectory(test)
endif()
