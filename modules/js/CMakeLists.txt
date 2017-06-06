# ----------------------------------------------------------------------------
#  CMake file for js support
# ----------------------------------------------------------------------------

message(STATUS "---------------- Start of JavaScript module ----------------------")

set(the_description "The js bindings")
set(MODULE_NAME js)

ocv_add_module(${MODULE_NAME} BINDINGS)

# TODO: add emscripten path
ocv_module_include_directories()

# get list of modules to wrap
message(STATUS "Wrapped in ${MODULE_NAME}:")
set(OPENCV_JS_MODULES)
foreach(m ${OPENCV_MODULES_BUILD})
  if (";${OPENCV_MODULE_${m}_WRAPPERS};" MATCHES ";${MODULE_NAME};" AND HAVE_${m})
    list(APPEND OPENCV_JS_MODULES ${m})
    message(STATUS "\t${m}")
  endif()
endforeach()

add_definitions("-std=c++11")

link_libraries(${OPENCV_MODULE_${the_module}_DEPS})

ocv_add_executable(${the_module} ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp)

set_target_properties(${the_module} PROPERTIES LINK_FLAGS "-s MODULARIZE=1 -s EXPORT_NAME=\"'cv'\" -s DEMANGLE_SUPPORT=1 --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/data --bind")

unset(MODULE_NAME)

message(STATUS "---------------- End of JavaScript module ----------------------")