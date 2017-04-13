#
# This is a helper file to build OpenCV addons
# It is used in the core part of OpenCV, but distributed for external packaging
#

cmake_minimum_required(VERSION "2.8.12" FATAL_ERROR)

set(OPENCV_ADDON_CMAKE_DIR ${CMAKE_CURRENT_LIST_DIR})

include(${CMAKE_CURRENT_LIST_DIR}/OpenCVUtils.cmake)

if("x${CMAKE_BUILD_TYPE}" STREQUAL "x")
  set(CMAKE_BUILD_TYPE Release)
endif()

if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()

if(NOT DEFINED OpenCV_PACKAGE_NAME)
  message(FATAL_ERROR "OpenCV_PACKAGE_NAME is not defined")
endif()

if(OpenCV_FOUND)
  if(";${OpenCV_ADDONS};" MATCHES ";${OpenCV_PACKAGE_NAME};")
    message(FATAL_ERROR "find_package(OpenCV) is used somewhere without OpenCV_SKIP_ADDONS with ${OpenCV_PACKAGE_NAME} or 'ALL'.")
  endif()
else()
  set(OpenCV_SKIP_ADDONS ${OpenCV_SKIP_ADDONS} "${OpenCV_PACKAGE_NAME}")
  find_package(OpenCV ${OpenCV_VERSION} REQUIRED)
endif()

if(NOT ADDON_SEPARATE_INSTALL)
  set(CMAKE_INSTALL_PREFIX "${OpenCV_INSTALL_PATH}")
endif()

if(NOT DEFINED OPENCV_MODULE_TYPE)
  set(OPENCV_MODULE_TYPE SHARED)
endif()

add_custom_target(opencv_extra_modules)
if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(opencv_extra_modules PROPERTIES FOLDER "addons")
endif()

if(NOT DEFINED LIBRARY_OUTPUT_PATH)
  set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
endif()
if(NOT DEFINED EXECUTABLE_OUTPUT_PATH)
  set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
endif()
if(NOT DEFINED OPENCV_BIN_INSTALL_PATH)
  set(OPENCV_BIN_INSTALL_PATH bin)
endif()
if(NOT DEFINED OPENCV_LIB_INSTALL_PATH)
  list(GET OpenCV_LIBS 0 __library_target)
  if(TARGET ${__library_target})
    string(TOUPPER "${CMAKE_BUILD_TYPE}" __build_type_up)
    get_target_property(__fname ${__library_target} LOCATION_${__build_type_up})
    if(fname)
      get_filename_component(__fpath ${__fname} PATH)
      file(RELATIVE_PATH OPENCV_LIB_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}" "${__fpath}")
    endif()
  endif()
  if(NOT OPENCV_LIB_INSTALL_PATH)
    set(OPENCV_LIB_INSTALL_PATH lib)
  endif()
endif()
set(OPENCV_LIBVERSION "${OpenCV_VERSION}")
set(OPENCV_SOVERSION "${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}")

ocv_clear_vars(OpenCVExtraModules_TARGETS)

add_definitions(-D__OPENCV_ADDON_BUILD=1)



#
# based on cmake/OpenCVModule.cmake
#

# clean flags for modules enabled on previous cmake run
# this is necessary to correctly handle modules removal
foreach(mod ${OPENCV_MODULES_BUILD} ${OPENCV_MODULES_DISABLED_USER})
  if(HAVE_${mod})
    unset(HAVE_${mod} CACHE)
  endif()
  unset(OPENCV_MODULE_${mod}_HEADERS CACHE)
  unset(OPENCV_MODULE_${mod}_SOURCES CACHE)
  unset(OPENCV_MODULE_${mod}_DEPS CACHE)
  unset(OPENCV_MODULE_${mod}_DEPS_EXT CACHE)
  unset(OPENCV_MODULE_${mod}_REQ_DEPS CACHE)
  unset(OPENCV_MODULE_${mod}_OPT_DEPS CACHE)
  unset(OPENCV_MODULE_${mod}_PRIVATE_REQ_DEPS CACHE)
  unset(OPENCV_MODULE_${mod}_PRIVATE_OPT_DEPS CACHE)
  unset(OPENCV_MODULE_${mod}_LINK_DEPS CACHE)
  unset(OPENCV_MODULE_${mod}_WRAPPERS CACHE)
endforeach()

foreach(mod ${OpenCV_LIB_COMPONENTS})
  set(HAVE_${mod} ON CACHE INTERNAL "")
endforeach()

set(OPENCV_MODULES_PUBLIC         "" CACHE INTERNAL "List of OpenCV modules marked for export")
set(OPENCV_MODULES_BUILD          "" CACHE INTERNAL "List of OpenCV modules included into the build")
set(OPENCV_MODULES_DISABLED_USER  "" CACHE INTERNAL "List of OpenCV modules explicitly disabled by user")

macro(ocv_add_dependencies full_modname)
  ocv_debug_message("ocv_add_dependencies(" ${full_modname} ${ARGN} ")")
  #we don't clean the dependencies here to allow this macro several times for every module
  foreach(d "REQUIRED" ${ARGN})
    if(d STREQUAL "REQUIRED")
      set(__depsvar OPENCV_MODULE_${full_modname}_REQ_DEPS)
    elseif(d STREQUAL "OPTIONAL")
      set(__depsvar OPENCV_MODULE_${full_modname}_OPT_DEPS)
    elseif(d STREQUAL "PRIVATE_REQUIRED")
      set(__depsvar OPENCV_MODULE_${full_modname}_PRIVATE_REQ_DEPS)
    elseif(d STREQUAL "PRIVATE_OPTIONAL")
      set(__depsvar OPENCV_MODULE_${full_modname}_PRIVATE_OPT_DEPS)
    elseif(d STREQUAL "WRAP")
      set(__depsvar OPENCV_MODULE_${full_modname}_WRAPPERS)
    else()
      list(APPEND ${__depsvar} "${d}")
    endif()
  endforeach()
  unset(__depsvar)

  ocv_list_unique(OPENCV_MODULE_${full_modname}_REQ_DEPS)
  ocv_list_unique(OPENCV_MODULE_${full_modname}_OPT_DEPS)
  ocv_list_unique(OPENCV_MODULE_${full_modname}_PRIVATE_REQ_DEPS)
  ocv_list_unique(OPENCV_MODULE_${full_modname}_PRIVATE_OPT_DEPS)
  ocv_list_unique(OPENCV_MODULE_${full_modname}_WRAPPERS)

  set(OPENCV_MODULE_${full_modname}_REQ_DEPS ${OPENCV_MODULE_${full_modname}_REQ_DEPS}
    CACHE INTERNAL "Required dependencies of ${full_modname} module")
  set(OPENCV_MODULE_${full_modname}_OPT_DEPS ${OPENCV_MODULE_${full_modname}_OPT_DEPS}
    CACHE INTERNAL "Optional dependencies of ${full_modname} module")
  set(OPENCV_MODULE_${full_modname}_PRIVATE_REQ_DEPS ${OPENCV_MODULE_${full_modname}_PRIVATE_REQ_DEPS}
    CACHE INTERNAL "Required private dependencies of ${full_modname} module")
  set(OPENCV_MODULE_${full_modname}_PRIVATE_OPT_DEPS ${OPENCV_MODULE_${full_modname}_PRIVATE_OPT_DEPS}
    CACHE INTERNAL "Optional private dependencies of ${full_modname} module")
  set(OPENCV_MODULE_${full_modname}_WRAPPERS ${OPENCV_MODULE_${full_modname}_WRAPPERS}
    CACHE INTERNAL "List of wrappers supporting module ${full_modname}")
endmacro()

# declare new OpenCV module in current folder
# Usage:
#   ocv_add_module(<name> [INTERNAL|BINDINGS] [REQUIRED] [<list of dependencies>] [OPTIONAL <list of optional dependencies>] [WRAP <list of wrappers>])
# Example:
#   ocv_add_module(yaom INTERNAL opencv_core opencv_highgui opencv_flann OPTIONAL opencv_cudev)
macro(ocv_add_module _name)
  ocv_debug_message("ocv_add_module(" ${_name} ${ARGN} ")")
  string(TOLOWER "${_name}" name)
  set(the_module opencv_${name})

  #guard agains redefinition
  if(";${OPENCV_MODULES_BUILD};${OPENCV_MODULES_DISABLED_USER};" MATCHES ";${the_module};")
    message(FATAL_ERROR "Redefinition of the ${the_module} module.
at:                    ${CMAKE_CURRENT_SOURCE_DIR}
previously defined at: ${OPENCV_MODULE_${the_module}_LOCATION}
")
  endif()

  if(NOT DEFINED the_description)
    set(the_description "The ${name} OpenCV module")
  endif()

  if(NOT DEFINED BUILD_${the_module}_INIT)
    set(BUILD_${the_module}_INIT ON)
  endif()

  # create option to enable/disable this module
  option(BUILD_${the_module} "Include ${the_module} module into the OpenCV build" ${BUILD_${the_module}_INIT})

  # remember the module details
  set(OPENCV_MODULE_${the_module}_DESCRIPTION "${the_description}" CACHE INTERNAL "Brief description of ${the_module} module")
  set(OPENCV_MODULE_${the_module}_LOCATION    "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Location of ${the_module} module sources")

  # parse list of dependencies
  if("${ARGV1}" STREQUAL "INTERNAL" OR "${ARGV1}" STREQUAL "BINDINGS")
    ocv_module_disable(${the_module})
  else()
    set(OPENCV_MODULE_${the_module}_CLASS "PUBLIC" CACHE INTERNAL "The category of the module")
    ocv_add_dependencies(${the_module} ${ARGN})
    if(BUILD_${the_module})
      set(OPENCV_MODULES_PUBLIC ${OPENCV_MODULES_PUBLIC} "${the_module}" CACHE INTERNAL "List of OpenCV modules marked for export")
    else()
      return()
    endif()
  endif()

  if(BUILD_${the_module})
    set(OPENCV_MODULES_BUILD ${OPENCV_MODULES_BUILD} "${the_module}" CACHE INTERNAL "List of OpenCV modules included into the build")
    set(HAVE_${the_module} ON CACHE INTERNAL "") # There is support for dependency resolver
  else()
    set(OPENCV_MODULES_DISABLED_USER ${OPENCV_MODULES_DISABLED_USER} "${the_module}" CACHE INTERNAL "List of OpenCV modules explicitly disabled by user")
  endif()
  return()
endmacro()

# excludes module from current configuration
macro(ocv_module_disable the_module)
  set(HAVE_${the_module} OFF CACHE INTERNAL "Module ${__modname} can not be built in current configuration")
  if(BUILD_${__modname})
    # touch variable controlling build of the module to suppress "unused variable" CMake warning
  endif()
  return() # leave the current folder
endmacro()


# setup include paths for the list of passed modules
macro(ocv_include_modules)
  foreach(d ${ARGN})
    if(d MATCHES "^opencv_" AND HAVE_${d})
      if (EXISTS "${OPENCV_MODULE_${d}_LOCATION}/include")
        ocv_include_directories("${OPENCV_MODULE_${d}_LOCATION}/include")
      endif()
    elseif(EXISTS "${d}" AND IS_DIRECTORY "${d}")
      ocv_include_directories("${d}")
    endif()
  endforeach()
endmacro()

# same as previous but with dependencies
macro(ocv_include_modules_recurse)
  ocv_include_modules(${ARGN})
  foreach(d ${ARGN})
    if(d MATCHES "^opencv_" AND HAVE_${d} AND DEFINED OPENCV_MODULE_${d}_DEPS)
      foreach (sub ${OPENCV_MODULE_${d}_DEPS})
        ocv_include_modules(${sub})
      endforeach()
    endif()
  endforeach()
endmacro()

# setup include paths for the list of passed modules
macro(ocv_target_include_modules target)
  foreach(d ${ARGN})
    if(d MATCHES "^opencv_" AND HAVE_${d})
      if (EXISTS "${OPENCV_MODULE_${d}_LOCATION}/include")
        ocv_target_include_directories(${target} "${OPENCV_MODULE_${d}_LOCATION}/include")
      endif()
    elseif(EXISTS "${d}" AND IS_DIRECTORY "${d}")
      ocv_target_include_directories(${target} "${d}")
    endif()
  endforeach()
endmacro()

# setup include paths for the list of passed modules and recursively add dependent modules
macro(ocv_target_include_modules_recurse target)
  foreach(d ${ARGN})
    if(d MATCHES "^opencv_" AND HAVE_${d})
      if (EXISTS "${OPENCV_MODULE_${d}_LOCATION}/include")
        ocv_target_include_directories(${target} "${OPENCV_MODULE_${d}_LOCATION}/include")
      endif()
      if(OPENCV_MODULE_${d}_DEPS)
        ocv_target_include_modules(${target} ${OPENCV_MODULE_${d}_DEPS})
      endif()
    elseif(EXISTS "${d}" AND IS_DIRECTORY "${d}")
      ocv_target_include_directories(${target} "${d}")
    endif()
  endforeach()
endmacro()

# setup include path for OpenCV headers for specified module
# ocv_module_include_directories(<extra include directories/extra include modules>)
macro(ocv_module_include_directories)
  ocv_target_include_directories(${the_module}
      "${OPENCV_MODULE_${the_module}_LOCATION}/include"
      "${OPENCV_MODULE_${the_module}_LOCATION}/src"
      "${CMAKE_CURRENT_BINARY_DIR}" # for OpenCL kernels / precompiled headers
      )
  ocv_target_include_modules(${the_module} ${OPENCV_MODULE_${the_module}_DEPS} ${ARGN})
endmacro()


# sets header and source files for the current module
# NB: all files specified as headers will be installed
# Usage:
# ocv_set_module_sources([HEADERS] <list of files> [SOURCES] <list of files>)
macro(ocv_set_module_sources)
  ocv_debug_message("ocv_set_module_sources(" ${ARGN} ")")

  set(OPENCV_MODULE_${the_module}_HEADERS "")
  set(OPENCV_MODULE_${the_module}_SOURCES "")

  foreach(f "HEADERS" ${ARGN})
    if(f STREQUAL "HEADERS" OR f STREQUAL "SOURCES")
      set(__filesvar "OPENCV_MODULE_${the_module}_${f}")
    else()
      list(APPEND ${__filesvar} "${f}")
    endif()
  endforeach()

  # use full paths for module to be independent from the module location
  ocv_convert_to_full_paths(OPENCV_MODULE_${the_module}_HEADERS)

  set(OPENCV_MODULE_${the_module}_HEADERS ${OPENCV_MODULE_${the_module}_HEADERS} CACHE INTERNAL "List of header files for ${the_module}")
  set(OPENCV_MODULE_${the_module}_SOURCES ${OPENCV_MODULE_${the_module}_SOURCES} CACHE INTERNAL "List of source files for ${the_module}")
endmacro()

# finds and sets headers and sources for the standard OpenCV module
# Usage:
# ocv_glob_module_sources(<extra sources&headers in the same format as used in ocv_set_module_sources>)
macro(ocv_glob_module_sources)
  ocv_debug_message("ocv_glob_module_sources(" ${ARGN} ")")

  file(GLOB_RECURSE lib_srcs
       "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"
  )
  file(GLOB_RECURSE lib_int_hdrs
       "${CMAKE_CURRENT_LIST_DIR}/src/*.hpp"
       "${CMAKE_CURRENT_LIST_DIR}/src/*.h"
  )
  file(GLOB_RECURSE lib_hdrs
       "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp"
       "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.h"
  )
  if (APPLE)
    file(GLOB_RECURSE lib_srcs_apple
         "${CMAKE_CURRENT_LIST_DIR}/src/*.mm"
    )
    list(APPEND lib_srcs ${lib_srcs_apple})
  endif()

  ocv_source_group("Src" DIRBASE "${CMAKE_CURRENT_LIST_DIR}/src" FILES ${lib_srcs} ${lib_int_hdrs})
  ocv_source_group("Include" DIRBASE "${CMAKE_CURRENT_LIST_DIR}/include" FILES ${lib_hdrs} ${lib_hdrs_detail})

  file(GLOB cl_kernels
       "${CMAKE_CURRENT_LIST_DIR}/src/opencl/*.cl"
  )
  if(cl_kernels)
    set(OCL_NAME opencl_kernels_${name})
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.hpp"
      COMMAND ${CMAKE_COMMAND} "-DMODULE_NAME=${name}" "-DCL_DIR=${CMAKE_CURRENT_LIST_DIR}/src/opencl" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" -P "${OPENCV_ADDON_CMAKE_DIR}/cl2cpp.cmake"
      DEPENDS ${cl_kernels} "${OPENCV_ADDON_CMAKE_DIR}/cl2cpp.cmake")
    ocv_source_group("Src\\opencl\\kernels" FILES ${cl_kernels})
    ocv_source_group("Src\\opencl\\kernels\\autogenerated" FILES "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.hpp")
    list(APPEND lib_srcs ${cl_kernels} "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.hpp")
  endif()

  ocv_set_module_sources(${ARGN} HEADERS ${lib_hdrs}
                         SOURCES ${lib_srcs} ${lib_int_hdrs})
endmacro()

# creates OpenCV module in current folder
# creates new target, configures standard dependencies, compilers flags, install rules
# Usage:
#   ocv_create_module(<extra link dependencies>)
#   ocv_create_module()
macro(ocv_create_module)
  ocv_debug_message("ocv_create_module(" ${ARGN} ")")
  set(the_module_target ${the_module})

  ocv_add_library(${the_module} ${OPENCV_MODULE_TYPE} ${OPENCV_MODULE_${the_module}_HEADERS} ${OPENCV_MODULE_${the_module}_SOURCES})

  ocv_target_link_libraries(${the_module} LINK_PRIVATE ${OPENCV_MODULE_${the_module}_DEPS})
  ocv_target_link_libraries(${the_module} LINK_PRIVATE ${OPENCV_MODULE_${the_module}_DEPS_EXT} ${OpenCV_LIBS} ${ARGN})

  add_dependencies(opencv_extra_modules ${the_module})

  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_module} PROPERTIES FOLDER "modules")
  endif()

  set_target_properties(${the_module} PROPERTIES
    OUTPUT_NAME "${the_module}${OPENCV_DLLVERSION}"
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    COMPILE_PDB_NAME "${the_module}${OPENCV_DLLVERSION}"
    COMPILE_PDB_NAME_DEBUG "${the_module}${OPENCV_DLLVERSION}${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}"
    COMPILE_PDB_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}"
    LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}"
    RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
  )

  if(MSVC)
    if(CMAKE_CROSSCOMPILING)
      set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:secchk")
    endif()
    set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libc /DEBUG")
  endif()

  get_target_property(_target_type ${the_module} TYPE)
  if("${_target_type}" STREQUAL "SHARED_LIBRARY")
    if(NOT ANDROID)
      # Android SDK build scripts can include only .so files into final .apk
      # As result we should not set version properties for Android
      set_target_properties(${the_module} PROPERTIES
        VERSION "${OPENCV_LIBVERSION}"
        SOVERSION "${OPENCV_SOVERSION}"
      )
    endif()
    set_target_properties(${the_module} PROPERTIES COMPILE_DEFINITIONS CVAPI_EXPORTS)
    set_target_properties(${the_module} PROPERTIES DEFINE_SYMBOL CVAPI_EXPORTS)
    ocv_install_target(${the_module} EXPORT OpenCVExtraModules OPTIONAL
      RUNTIME DESTINATION "${OPENCV_BIN_INSTALL_PATH}" COMPONENT libs
      LIBRARY DESTINATION "${OPENCV_LIB_INSTALL_PATH}" COMPONENT libs
      ARCHIVE DESTINATION "${OPENCV_LIB_INSTALL_PATH}" COMPONENT dev
      )
    install(TARGETS ${the_module}
      LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT dev NAMELINK_ONLY)
  endif()

  list(GET OpenCV_INCLUDE_DIRS 0 __include_dir)
  file(RELATIVE_PATH __include_dir "${OpenCV_INSTALL_PATH}" "${__include_dir}")
  foreach(hdr ${OPENCV_MODULE_${the_module}_HEADERS})
    string(REGEX REPLACE "^.*opencv2/" "opencv2/" hdr2 "${hdr}")
    if(NOT hdr2 MATCHES "/private/" AND hdr2 MATCHES "^(opencv2/?.*)/[^/]+.h(..)?$")
      install(FILES ${hdr} OPTIONAL DESTINATION "${__include_dir}/${CMAKE_MATCH_1}" COMPONENT dev)
    endif()
  endforeach()

endmacro()

# short command for adding simple OpenCV module
# see ocv_add_module for argument details
# Usage:
# ocv_define_module(module_name [INTERNAL] [REQUIRED] [<list of dependencies>] [OPTIONAL <list of optional dependencies>])
macro(ocv_define_module module_name)
  ocv_debug_message("ocv_define_module(" ${module_name} ${ARGN} ")")

  ocv_add_module(${module_name} ${ARGN})
  ocv_glob_module_sources()
  ocv_module_include_directories()
  ocv_create_module()

  ocv_add_accuracy_tests()
  ocv_add_perf_tests()
  ocv_add_samples()
endmacro()

# ensures that all passed modules are available
# sets OCV_DEPENDENCIES_FOUND variable to TRUE/FALSE
macro(ocv_check_dependencies)
  set(OCV_DEPENDENCIES_FOUND TRUE)
  foreach(d ${ARGN})
    if(d MATCHES "^opencv_[^ ]+$" AND NOT HAVE_${d})
      set(OCV_DEPENDENCIES_FOUND FALSE)
      break()
    endif()
  endforeach()
endmacro()

# this is a command for adding OpenCV performance tests to the module
# ocv_add_perf_tests(<extra_dependencies>)
function(ocv_add_perf_tests)
  # not supported
endfunction()

# this is a command for adding OpenCV accuracy/regression tests to the module
# ocv_add_accuracy_tests([FILES <source group name> <list of sources>] [DEPENDS_ON] <list of extra dependencies>)
function(ocv_add_accuracy_tests)
  # not supported
endfunction()

function(ocv_add_samples)
  # not supported
endfunction()


#
# based on cmake/OpenCVGenConfig.cmake
#
function(ocv_addon_finalize)
  macro(ocv_add_module _name)
    string(TOLOWER "${_name}" name)
    set(the_module opencv_${name})
    set(OPENCV_MODULE_${the_module}_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "")
    project(${the_module})
  endmacro()
  foreach(m ${OPENCV_MODULES_BUILD})
    foreach(d
        ${OPENCV_MODULE_${m}_REQ_DEPS} ${OPENCV_MODULE_${m}_PRIVATE_REQ_DEPS}
        ${OPENCV_MODULE_${m}_OPT_DEPS} ${OPENCV_MODULE_${m}_PRIVATE_OPT_DEPS}
        )
      string(TOLOWER "${d}" __d)
      if(HAVE_${d} OR HAVE_${__d} OR TARGET ${d} OR EXISTS ${d})
        set(OPENCV_MODULE_${m}_DEPS ${OPENCV_MODULE_${m}_DEPS} ${d} CACHE INTERNAL "Flattened dependencies of ${m} module")
      endif()
    endforeach()
    add_subdirectory("${OPENCV_MODULE_${m}_LOCATION}" "${CMAKE_CURRENT_BINARY_DIR}/modules_build/${m}")
  endforeach()

  status("")
  status("Package modules:" "${OPENCV_MODULES_BUILD}")
  status("")
  status("Install path:" "${CMAKE_INSTALL_PREFIX}")
  status("")

  set(OPENCV_MODULES_CONFIGCMAKE ${OPENCV_MODULES_BUILD})

  #
  # CMAKE_BINARY_DIR
  #
  set(OpenCV_INCLUDE_DIRS_CONFIGCMAKE "")
  foreach(m ${OPENCV_MODULES_BUILD})
    if(EXISTS "${OPENCV_MODULE_${m}_LOCATION}/include")
      set(OpenCV_INCLUDE_DIRS_CONFIGCMAKE "${OpenCV_INCLUDE_DIRS_CONFIGCMAKE} \"${OPENCV_MODULE_${m}_LOCATION}/include\"")
    endif()
  endforeach()

  export(TARGETS ${OpenCVExtraModules_TARGETS} FILE "${CMAKE_BINARY_DIR}/${OpenCV_PACKAGE_NAME}Modules.cmake")

  configure_file("${OPENCV_ADDON_CMAKE_DIR}/templates/OpenCVAddonConfig.cmake.in" "${CMAKE_BINARY_DIR}/${OpenCV_PACKAGE_NAME}Config.cmake" @ONLY)
  configure_file("${OpenCV_CONFIG_PATH}/OpenCVConfig-version.cmake" "${CMAKE_BINARY_DIR}/${OpenCV_PACKAGE_NAME}Config-version.cmake" COPYONLY)


  #
  # make install
  #
  set(OpenCV_INCLUDE_DIRS_CONFIGCMAKE "")

  function(ocv_gen_config TMP_DIR)
    file(RELATIVE_PATH OpenCV_CONFIG_PATH_RELATIVE "${OpenCV_INSTALL_PATH}" "${OpenCV_CONFIG_PATH}")

    if(OpenCV_CONFIG_PATH_RELATIVE MATCHES "/OpenCV/?$") # /usr/share/OpenCV /lib/cmake/OpenCV
      set(PACKAGE_CONFIG_INSTALL_PATH "${OpenCV_CONFIG_PATH_RELATIVE}/../${OpenCV_PACKAGE_NAME}")
    elseif(OpenCV_CONFIG_PATH_RELATIVE MATCHES "/OpenCV/[cC]make/?$") # /usr/share/OpenCV/cmake
      set(PACKAGE_CONFIG_INSTALL_PATH "${OpenCV_CONFIG_PATH_RELATIVE}/../../${OpenCV_PACKAGE_NAME}/cmake")
    else() # <install_path>/cmake on Windows
      set(PACKAGE_CONFIG_INSTALL_PATH "${OpenCV_CONFIG_PATH_RELATIVE}")
    endif()

    # file(RELATIVE_PATH) doesn't work fine with '..' in the path with some CMake 2.8
    get_filename_component(PACKAGE_CONFIG_INSTALL_PATH_ABSOLUTE "${OpenCV_INSTALL_PATH}/${PACKAGE_CONFIG_INSTALL_PATH}" ABSOLUTE)
    file(RELATIVE_PATH OpenCV_BASECONFIG_PATH_RELATIVE "${PACKAGE_CONFIG_INSTALL_PATH_ABSOLUTE}" "${OpenCV_CONFIG_PATH}")

    configure_file("${OPENCV_ADDON_CMAKE_DIR}/templates/OpenCVConfig-addon.cmake.in" "${TMP_DIR}/${OpenCV_PACKAGE_NAME}.cmake" @ONLY)
    configure_file("${OPENCV_ADDON_CMAKE_DIR}/templates/OpenCVAddonConfig.cmake.in" "${TMP_DIR}/${OpenCV_PACKAGE_NAME}Config.cmake" @ONLY)
    configure_file("${OpenCV_CONFIG_PATH}/OpenCVConfig-version.cmake" "${TMP_DIR}/${OpenCV_PACKAGE_NAME}Config-version.cmake" COPYONLY)
    install(EXPORT OpenCVExtraModules DESTINATION "${OpenCV_CONFIG_PATH_RELATIVE}" FILE ${OpenCV_PACKAGE_NAME}Modules.cmake COMPONENT dev)
    install(FILES
        "${TMP_DIR}/${OpenCV_PACKAGE_NAME}.cmake"
        DESTINATION "${OpenCV_CONFIG_PATH_RELATIVE}/addons" COMPONENT dev)
    install(FILES
        "${TMP_DIR}/${OpenCV_PACKAGE_NAME}Config-version.cmake"
        "${TMP_DIR}/${OpenCV_PACKAGE_NAME}Config.cmake"
        DESTINATION "${PACKAGE_CONFIG_INSTALL_PATH}" COMPONENT dev)
  endfunction()

  if(UNIX) # ANDROID is UNIX too
    ocv_gen_config("${CMAKE_BINARY_DIR}/unix-install")
  elseif(WIN32)
    ocv_gen_config("${CMAKE_BINARY_DIR}/win-install")
  endif()
endfunction()




function(ocv_output_status msg)
  message(STATUS "${msg}")
endfunction()

status("")
status("Bootstrap OpenCV addon build system...")
status("")
status("Package name:" "${OpenCV_PACKAGE_NAME}")
status("")

# C/C++ options
status("Configuration:" "${CMAKE_BUILD_TYPE}")
status("C/C++:")
status("  C++ Compiler:"           ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1})
status("  C++ flags (Release):"    ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
status("  C++ flags (Debug):"      ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
status("  C Compiler:"             ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1})
status("  C flags (Release):"      ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE})
status("  C flags (Debug):"        ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG})
if(WIN32)
  status("  Linker flags (Release):" ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE})
  status("  Linker flags (Debug):"   ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG})
else()
  status("  Linker flags (Release):" ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE})
  status("  Linker flags (Debug):"   ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG})
endif()
status("")

# OpenCV
status("OpenCV:" "${OpenCV_VERSION}")
status("  Modules:" ${OpenCV_LIB_COMPONENTS})
if(OpenCV_ADDONS)
  status("  Addons:" ${OpenCV_ADDONS})
endif()
status("  Config path:" "${OpenCV_CONFIG_PATH}")
status("  Install path:" "${OpenCV_INSTALL_PATH}")
status("    Libraries:" "${OPENCV_LIB_INSTALL_PATH}")
status("    Binaries:" "${OPENCV_BIN_INSTALL_PATH}")
status("")
