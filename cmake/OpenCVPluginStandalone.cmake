# Standalone OpenCV plugins build scripts
#
# Useful OpenCV common build variables:
# - CMAKE_BUILD_TYPE=Release/Debug
# - BUILD_WITH_DEBUG_INFO=ON
# - ENABLE_BUILD_HARDENING=ON
#
# Plugin configuration variables:
# - OPENCV_PLUGIN_DEPS - set of extra dependencies (modules), used for include dirs, target_link_libraries
# - OPENCV_PLUGIN_SUFFIX
# - OPENCV_PLUGIN_NAME
# - OPENCV_PLUGIN_OUTPUT_NAME_FULL (overrides both OPENCV_PLUGIN_NAME / OPENCV_PLUGIN_SUFFIX)
#
#=============================================

if(NOT OpenCV_SOURCE_DIR)
  message(FATAL_ERROR "OpenCV_SOURCE_DIR must be set to build the plugin!")
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

set(BUILD_SHARED_LIBS ON CACHE BOOL "")
if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "Static plugin build does not make sense")
endif()

# re-use OpenCV build scripts
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVUtils.cmake")
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectCXXCompiler.cmake")
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVCompilerOptions.cmake")

function(ocv_create_plugin module default_name dependency_target dependency_target_desc)

  set(OPENCV_PLUGIN_NAME ${default_name} CACHE STRING "")
  set(OPENCV_PLUGIN_DESTINATION "" CACHE PATH "")
  project(${OPENCV_PLUGIN_NAME} LANGUAGES CXX)

  if(NOT TARGET ${dependency_target})
    message(FATAL_ERROR "${dependency_target_desc} was not found! (missing target ${dependency_target})")
  endif()

  set(modules_ROOT "${OpenCV_SOURCE_DIR}/modules")
  set(module_ROOT "${modules_ROOT}/${module}")

  foreach(src ${ARGN})
    list(APPEND sources "${module_ROOT}/${src}")
  endforeach()

  add_library(${OPENCV_PLUGIN_NAME} MODULE
      "${sources}"
      ${OPENCV_PLUGIN_EXTRA_SRC_FILES}
  )

  if(OPENCV_PLUGIN_DEPS)
    foreach(d ${OPENCV_PLUGIN_DEPS})
      list(APPEND OPENCV_PLUGIN_EXTRA_INCLUDES "${modules_ROOT}/${d}/include")
    endforeach()
  endif()

  target_include_directories(${OPENCV_PLUGIN_NAME} PRIVATE
      "${CMAKE_CURRENT_BINARY_DIR}"
      "${module_ROOT}/src"
      "${module_ROOT}/include"
      ${OPENCV_PLUGIN_EXTRA_INCLUDES}
  )
  target_compile_definitions(${OPENCV_PLUGIN_NAME} PRIVATE "BUILD_PLUGIN=1")

  target_link_libraries(${OPENCV_PLUGIN_NAME} PRIVATE ${dependency_target})
  set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_VISIBILITY_PRESET hidden
  )

  if(DEFINED OPENCV_PLUGIN_MODULE_PREFIX)
    set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES PREFIX "${OPENCV_PLUGIN_MODULE_PREFIX}")
  endif()

  if(APPLE)
    set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  elseif(WIN32)
    # Hack for Windows only, Linux/MacOS uses global symbol table (without exact .so binding)
    find_package(OpenCV REQUIRED ${module} ${OPENCV_PLUGIN_DEPS})
    target_link_libraries(${OPENCV_PLUGIN_NAME} PRIVATE ${OpenCV_LIBRARIES})
  endif()

  if(NOT OpenCV_FOUND)  # build against sources (Linux)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/opencv2/opencv_modules.hpp" "#pragma once")
  endif()

  if(WIN32)
    ocv_update(OPENCV_DEBUG_POSTFIX d)
  endif()
  set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}")

  if(DEFINED OPENCV_PLUGIN_SUFFIX)
    # custom value
  else()
    if(WIN32)
      ocv_update(OPENCV_PLUGIN_VERSION "${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}")
      if(CMAKE_CXX_SIZEOF_DATA_PTR EQUAL 8)
        ocv_update(OPENCV_PLUGIN_ARCH "_64")
      else()
        ocv_update(OPENCV_PLUGIN_ARCH "")
      endif()
    else()
      # empty
    endif()
    ocv_update(OPENCV_PLUGIN_SUFFIX "${OPENCV_PLUGIN_VERSION}${OPENCV_PLUGIN_ARCH}")
  endif()

  if(OPENCV_PLUGIN_DESTINATION)
    set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OPENCV_PLUGIN_DESTINATION}")
    message(STATUS "Output destination: ${OPENCV_PLUGIN_DESTINATION}")
  endif()

  if(OPENCV_PLUGIN_OUTPUT_NAME_FULL)
    set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES OUTPUT_NAME "${OPENCV_PLUGIN_OUTPUT_NAME_FULL}")
  elseif(OPENCV_PLUGIN_OUTPUT_NAME)
    set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES OUTPUT_NAME "${OPENCV_PLUGIN_OUTPUT_NAME}${OPENCV_PLUGIN_SUFFIX}")
  else()
    set_target_properties(${OPENCV_PLUGIN_NAME} PROPERTIES OUTPUT_NAME "${OPENCV_PLUGIN_NAME}${OPENCV_PLUGIN_SUFFIX}")
  endif()

  install(TARGETS ${OPENCV_PLUGIN_NAME} LIBRARY DESTINATION . COMPONENT plugins)

  message(STATUS "Library name: ${OPENCV_PLUGIN_NAME}")

endfunction()
