if(OPENCV_INITIAL_PASS)
  # generator for Objective-C source code and documentation signatures
  add_subdirectory(generator)
endif()

if(NOT APPLE_FRAMEWORK)
  return()
endif()

set(the_description "The Objective-C bindings")
ocv_add_module(objc BINDINGS opencv_core opencv_imgproc PRIVATE_REQUIRED opencv_objc_bindings_generator)

add_custom_target(${the_module}
    ALL
    COMMENT "Objective-C framework"
)
add_dependencies(${the_module} gen_opencv_objc_source)

#include(${CMAKE_CURRENT_SOURCE_DIR}/common.cmake)
