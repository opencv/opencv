# Local variables (set for each module):
#
# name       - short name in lower case i.e. core
# the_module - full name in lower case i.e. opencv_core

# Global variables:
#
# OPENCV_MODULE_${the_module}_LOCATION
# OPENCV_MODULE_${the_module}_DESCRIPTION
# OPENCV_MODULE_${the_module}_HEADERS
# OPENCV_MODULE_${the_module}_SOURCES
# OPENCV_MODULE_${the_module}_DEPS - final flattened set of module dependencies
# OPENCV_MODULE_${the_module}_DEPS_EXT
# OPENCV_MODULE_${the_module}_REQ_DEPS
# OPENCV_MODULE_${the_module}_OPT_DEPS
# HAVE_${the_module} - for fast check of module availability

# To control the setup of the module you could also set:
# the_description - text to be used as current module description
# OPENCV_MODULE_TYPE - STATIC|SHARED - set to force override global settings for current module

# The verbose template for OpenCV module:
#
#   ocv_add_module(modname <dependencies>)
#   ocv_glob_module_sources() or glob them manually and ocv_set_module_sources(...)
#   ocv_module_include_directories(<extra include directories>)
#   ocv_create_module()
#   <add extra link dependencies, compiler options, etc>
#   ocv_add_precompiled_headers(${the_module})
#   <add extra installation rules>
#   ocv_add_accuracy_tests(<extra dependencies>)
#   ocv_add_perf_tests(<extra dependencies>)
#
#
# If module have no "extra" then you can define it in one line:
#
#   ocv_define_module(modname <dependencies>)

# clean flags for modules enabled on previous cmake run
# this is necessary to correctly handle modules removal
foreach(mod ${OPENCV_MODULES_BUILD})
  if(HAVE_${mod})
    unset(HAVE_${mod} CACHE)
  endif()
endforeach()

# clean modules info which needs to be recalculated
set(OPENCV_MODULES_PUBLIC         "" CACHE INTERNAL "List of OpenCV modules marked for export")
set(OPENCV_MODULES_BUILD          "" CACHE INTERNAL "List of OpenCV modules included into the build")
set(OPENCV_MODULES_DISABLED_USER  "" CACHE INTERNAL "List of OpenCV modules explicitly disabled by user")
set(OPENCV_MODULES_DISABLED_AUTO  "" CACHE INTERNAL "List of OpenCV modules implicitly disabled due to dependencies")
set(OPENCV_MODULES_DISABLED_FORCE "" CACHE INTERNAL "List of OpenCV modules which can not be build in current configuration")

# adds dependencies to OpenCV module
# Usage:
#   add_dependencies(opencv_<name> [REQUIRED] [<list of dependencies>] [OPTIONAL <list of modules>])
# Notes:
# * <list of dependencies> - can include full names of modules or full pathes to shared/static libraries or cmake targets
macro(ocv_add_dependencies full_modname)
  #we don't clean the dependencies here to allow this macro several times for every module
  foreach(d "REQIRED" ${ARGN})
    if(d STREQUAL "REQIRED")
      set(__depsvar OPENCV_MODULE_${full_modname}_REQ_DEPS)
    elseif(d STREQUAL "OPTIONAL")
      set(__depsvar OPENCV_MODULE_${full_modname}_OPT_DEPS)
    else()
      list(APPEND ${__depsvar} "${d}")
    endif()
  endforeach()

  if(OPENCV_MODULE_${full_modname}_REQ_DEPS)
    list(REMOVE_DUPLICATES OPENCV_MODULE_${full_modname}_REQ_DEPS)
  endif()
  if(OPENCV_MODULE_${full_modname}_OPT_DEPS)
    list(REMOVE_DUPLICATES OPENCV_MODULE_${full_modname}_OPT_DEPS)
  endif()
  set(OPENCV_MODULE_${full_modname}_REQ_DEPS ${OPENCV_MODULE_${full_modname}_REQ_DEPS} CACHE INTERNAL "Required dependencies of ${full_modname} module")
  set(OPENCV_MODULE_${full_modname}_OPT_DEPS ${OPENCV_MODULE_${full_modname}_OPT_DEPS} CACHE INTERNAL "Optional dependencies of ${full_modname} module")
  
  unset(__depsvar)
endmacro()

# declare new OpenCV module in current folder
# Usage:
#   ocv_add_module(<name> [INTERNAL|BINDINGS] [REQUIRED] [<list of dependencies>] [OPTIONAL <list of optional dependencies>]) 
# Example:
#   ocv_add_module(yaom INTERNAL opencv_core opencv_highgui NOLINK opencv_flann OPTIONAL opencv_gpu)
macro(ocv_add_module _name)
  string(TOLOWER "${_name}" name)
  string(REGEX REPLACE "^opencv_" "" ${name} "${name}")
  set(the_module opencv_${name})
  
  # the first pass - collect modules info, the second pass - create targets
  if(OPENCV_INITIAL_PASS)
    #guard agains redefinition
    if(";${OPENCV_MODULES_BUILD};${OPENCV_MODULES_DISABLED_USER};" MATCHES ";${the_module};")
      message(FATAL_ERROR "Redefinition of the ${the_module} module.
  at:                    ${CMAKE_CURRENT_SOURCE_DIR}
  previously defined at: ${OPENCV_MODULE_${the_module}_LOCATION}
")
    endif()

    #remember module details
    if(NOT DEFINED the_description)
      set(the_description "The ${name} OpenCV module")
    endif()
    set(OPENCV_MODULE_${the_module}_DESCRIPTION "${the_description}" CACHE INTERNAL "Brief description of ${the_module} module")
    set(OPENCV_MODULE_${the_module}_LOCATION    "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Location of ${the_module} module sources")
    unset(OPENCV_MODULE_${the_module}_REQ_DEPS CACHE)
    unset(OPENCV_MODULE_${the_module}_OPT_DEPS CACHE)
    
    #create option to enable/disable this module
    option(BUILD_${the_module} "Include ${the_module} module into the OpenCV build" ON)

    if("${ARGV1}" STREQUAL "INTERNAL" OR "${ARGV1}" STREQUAL "BINDINGS")
      set(__ocv_argn__ ${ARGN})
      list(REMOVE_AT __ocv_argn__ 0)
      ocv_add_dependencies(${the_module} ${__ocv_argn__})
      unset(__ocv_argn__)
    else()
      ocv_add_dependencies(${the_module} ${ARGN})
      if(BUILD_${the_module})
        set(OPENCV_MODULES_PUBLIC ${OPENCV_MODULES_PUBLIC} "${the_module}" CACHE INTERNAL "List of OpenCV modules marked for export")
      endif()
    endif()

    if(BUILD_${the_module})
      set(OPENCV_MODULES_BUILD ${OPENCV_MODULES_BUILD} "${the_module}" CACHE INTERNAL "List of OpenCV modules included into the build")
    else()
      set(OPENCV_MODULES_DISABLED_USER ${OPENCV_MODULES_DISABLED_USER} "${the_module}" CACHE INTERNAL "List of OpenCV modules explicitly disabled by user")
    endif()
    
    #TODO: add submodules if any

    #stop processing of current file
    return()
  else(OPENCV_INITIAL_PASS)
    if(NOT BUILD_${the_module})
      #extra protection from redefinition
      return()
    endif()
    project(${the_module})
  endif(OPENCV_INITIAL_PASS)
endmacro()

# Internal macro; disables OpenCV module
# ocv_module_turn_off(<module name>)
macro(__ocv_module_turn_off the_module)
  list(APPEND OPENCV_MODULES_DISABLED_AUTO "${the_module}")
  list(REMOVE_ITEM OPENCV_MODULES_BUILD "${the_module}")
  list(REMOVE_ITEM OPENCV_MODULES_PUBLIC "${the_module}")
  set(HAVE_${the_module} OFF CACHE INTERNAL "Module ${the_module} can not be built in current configuration")
endmacro()

macro(ocv_module_disable module)
  set(__modname ${module})
  if(NOT __modname MATCHES "^opencv_")
    set(__modname opencv_${module})
  endif()
  list(APPEND OPENCV_MODULES_DISABLED_FORCE "${__modname}")
  set(HAVE_${__modname} OFF CACHE INTERNAL "Module ${__modname} can not be built in current configuration")
  set(OPENCV_MODULE_${__modname}_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Location of ${__modname} module sources")
  set(OPENCV_MODULES_DISABLED_FORCE "${OPENCV_MODULES_DISABLED_FORCE}" CACHE INTERNAL "List of OpenCV modules which can not be build in current configuration")
  unset(__modname)
  return()#leave the current folder
endmacro()


macro(__ocv_flatten_module_required_dependencies the_module)
  set(__flattened_deps "")
  set(__resolved_deps "")
  set(__req_depends ${OPENCV_MODULE_${the_module}_REQ_DEPS})
  
  while(__req_depends)
    list(GET __req_depends 0 __dep)
    list(REMOVE_AT __req_depends 0)
    if(__dep STREQUAL the_module)
      #TODO: think how to deal with cyclic dependency
      __ocv_module_turn_off(${the_module})
      break()
    elseif("${OPENCV_MODULES_DISABLED_USER};${OPENCV_MODULES_DISABLED_AUTO}" MATCHES "(^|;)${__dep}(;|$)")
      #depends on disabled module
      __ocv_module_turn_off(${the_module})
      break()
    elseif("${OPENCV_MODULES_BUILD}" MATCHES "(^|;)${__dep}(;|$)")
      if(__resolved_deps MATCHES "(^|;)${__dep}(;|$)")
        #all dependencies of this module are already resolved
        list(APPEND __flattened_deps "${__dep}")
      else()
        #put all required subdependencies before this dependency and mark it as resolved
        list(APPEND __resolved_deps "${__dep}")
        list(INSERT __req_depends 0 ${OPENCV_MODULE_${__dep}_REQ_DEPS} ${__dep})
      endif()
    elseif(__dep MATCHES "^opencv_")
      #depends on missing module
      __ocv_module_turn_off(${the_module})
      break()
    else()
      #skip non-modules
    endif()
  endwhile()

  if(__flattened_deps)
    list(REMOVE_DUPLICATES __flattened_deps)
    set(OPENCV_MODULE_${the_module}_DEPS ${__flattened_deps})
  else()
    set(OPENCV_MODULE_${the_module}_DEPS "")
  endif()
  
  unset(__resolved_deps)
  unset(__flattened_deps)
  unset(__req_depends)
  unset(__dep)
endmacro()

macro(__ocv_flatten_module_optional_dependencies the_module)
  set(__flattened_deps ${OPENCV_MODULE_${the_module}_DEPS})
  set(__resolved_deps ${OPENCV_MODULE_${the_module}_DEPS})
  set(__opt_depends ${OPENCV_MODULE_${the_module}_OPT_DEPS})
  
  while(__opt_depends)
    list(GET __opt_depends 0 __dep)
    list(REMOVE_AT __opt_depends 0)
    if(__dep STREQUAL the_module)
      #TODO: think how to deal with cyclic dependency
      __ocv_module_turn_off(${the_module})
      break()
    elseif("${OPENCV_MODULES_BUILD}" MATCHES "(^|;)${__dep}(;|$)")
      if(__resolved_deps MATCHES "(^|;)${__dep}(;|$)")
        #all dependencies of this module are already resolved
        list(APPEND __flattened_deps "${__dep}")
      else()
        #put all subdependencies before this dependency and mark it as resolved
        list(APPEND __resolved_deps "${__dep}")
        list(INSERT __opt_depends 0 ${OPENCV_MODULE_${__dep}_REQ_DEPS} ${OPENCV_MODULE_${__dep}_OPT_DEPS} ${__dep})
      endif()
    else()
      #skip non-modules or missing modules
    endif()
  endwhile()
  if(__flattened_deps)
    list(REMOVE_DUPLICATES __flattened_deps)
    set(OPENCV_MODULE_${the_module}_DEPS ${__flattened_deps})
  else()
    set(OPENCV_MODULE_${the_module}_DEPS "")
  endif()
  
  unset(__resolved_deps)
  unset(__flattened_deps)
  unset(__opt_depends)
  unset(__dep)
endmacro()

macro(__ocv_flatten_module_dependencies)
  foreach(m ${OPENCV_MODULES_DISABLED_USER})
    set(HAVE_${m} OFF CACHE INTERNAL "Module ${m} will not be built in current configuration")
  endforeach()
  foreach(m ${OPENCV_MODULES_BUILD})
    set(HAVE_${m} ON CACHE INTERNAL "Module ${m} will not be built in current configuration")
    __ocv_flatten_module_required_dependencies(${m})
  endforeach()
  
  foreach(m ${OPENCV_MODULES_BUILD})
    __ocv_flatten_module_optional_dependencies(${m})
    
    #dependencies from other modules
    set(OPENCV_MODULE_${m}_DEPS ${OPENCV_MODULE_${m}_DEPS} CACHE INTERNAL "Flattened dependencies of ${m} module")
    #extra dependencies
    set(OPENCV_MODULE_${m}_DEPS_EXT ${OPENCV_MODULE_${m}_REQ_DEPS} ${OPENCV_MODULE_${m}_OPT_DEPS})
    if(OPENCV_MODULE_${m}_DEPS_EXT AND OPENCV_MODULE_${m}_DEPS)
      list(REMOVE_ITEM OPENCV_MODULE_${m}_DEPS_EXT ${OPENCV_MODULE_${m}_DEPS})
    endif()
    ocv_list_filterout(OPENCV_MODULE_${m}_DEPS_EXT "^opencv_[^ ]+$")
    set(OPENCV_MODULE_${m}_DEPS_EXT ${OPENCV_MODULE_${m}_DEPS_EXT} CACHE INTERNAL "Extra dependencies of ${m} module")
  endforeach()
  
  set(OPENCV_MODULES_PUBLIC        ${OPENCV_MODULES_PUBLIC}        CACHE INTERNAL "List of OpenCV modules marked for export")
  set(OPENCV_MODULES_BUILD         ${OPENCV_MODULES_BUILD}         CACHE INTERNAL "List of OpenCV modules included into the build")
  set(OPENCV_MODULES_DISABLED_AUTO ${OPENCV_MODULES_DISABLED_AUTO} CACHE INTERNAL "List of OpenCV modules implicitly disabled due to dependencies")
endmacro()

# collect modules from specified directories
# NB: must be called only once!
macro(ocv_glob_modules)
  if(DEFINED OPENCV_INITIAL_PASS)
    message(FATAL_ERROR "OpenCV has already loaded its modules. Calling ocv_glob_modules second time is not allowed.")
  endif()
  set(__directories_observed "")

  #collect modules
  set(OPENCV_INITIAL_PASS ON)
  foreach(__path ${ARGN})
    ocv_get_real_path(__path "${__path}")
    list(FIND __directories_observed "${__path}" __pathIdx)
    if(__pathIdx GREATER -1)
      message(FATAL_ERROR "The directory ${__path} is observed for OpenCV modules second time.")
    endif()
    list(APPEND __directories_observed "${__path}")

    file(GLOB __ocvmodules RELATIVE "${__path}" "${__path}/*")
    if(__ocvmodules)
      list(SORT __ocvmodules)
      foreach(mod ${__ocvmodules})
        ocv_get_real_path(__modpath "${__path}/${mod}")
        if(EXISTS "${__modpath}/CMakeLists.txt")
          list(FIND __directories_observed "${__modpath}" __pathIdx)
          if(__pathIdx GREATER -1)
            message(FATAL_ERROR "The module from ${__modpath} is already loaded.")
          endif()
          list(APPEND __directories_observed "${__modpath}")

          add_subdirectory("${__modpath}" "${CMAKE_CURRENT_BINARY_DIR}/${mod}/.${mod}")
        endif()
      endforeach()
    endif()
  endforeach()
  unset(__ocvmodules)
  unset(__directories_observed)
  unset(__path)
  unset(__modpath)
  unset(__pathIdx)

  #resolve dependencies
  __ocv_flatten_module_dependencies()
  
  #order modules by dependencies
  set(OPENCV_MODULES_BUILD_ "")
  foreach(m ${OPENCV_MODULES_BUILD})
    list(APPEND OPENCV_MODULES_BUILD_ ${OPENCV_MODULE_${m}_DEPS} ${m})
  endforeach()
  ocv_list_unique(OPENCV_MODULES_BUILD_)

  #create modules
  set(OPENCV_INITIAL_PASS OFF PARENT_SCOPE)
  set(OPENCV_INITIAL_PASS OFF)
  foreach(m ${OPENCV_MODULES_BUILD_})
    if(m MATCHES "^opencv_")
      string(REGEX REPLACE "^opencv_" "" __shortname "${m}")
      add_subdirectory("${OPENCV_MODULE_${m}_LOCATION}" "${CMAKE_CURRENT_BINARY_DIR}/${__shortname}")
    endif()
  endforeach()
  unset(__shortname)
endmacro()

# setup include paths for the list of passed modules
macro(ocv_include_modules)
  foreach(d ${ARGN})
    if(d MATCHES "^opencv_" AND HAVE_${d})
      if (EXISTS "${OPENCV_MODULE_${d}_LOCATION}/include")
        ocv_include_directories("${OPENCV_MODULE_${d}_LOCATION}/include")
      endif()
    elseif(EXISTS "${d}")
      ocv_include_directories("${d}")
    endif()
  endforeach()
endmacro()

# setup include path for OpenCV headers for specified module
# ocv_module_include_directories(<extra include directories/extra include modules>)
macro(ocv_module_include_directories)
  ocv_include_directories("${OPENCV_MODULE_${the_module}_LOCATION}/include"
                          "${OPENCV_MODULE_${the_module}_LOCATION}/src"
                          "${CMAKE_CURRENT_BINARY_DIR}"#for precompiled headers
                          )
  ocv_include_modules(${OPENCV_MODULE_${the_module}_DEPS} ${ARGN})
endmacro()


# sets header and source files for the current module
# NB: all files specified as headers will be installed
# Usage:
# ocv_set_module_sources([HEADERS] <list of files> [SOURCES] <list of files>)
macro(ocv_set_module_sources)
  set(OPENCV_MODULE_${the_module}_HEADERS "")
  set(OPENCV_MODULE_${the_module}_SOURCES "")
  
  foreach(f "HEADERS" ${ARGN})
    if(f STREQUAL "HEADERS" OR f STREQUAL "SOURCES")
      set(__filesvar "OPENCV_MODULE_${the_module}_${f}")
    else()
      list(APPEND ${__filesvar} "${f}")
    endif()
  endforeach()
  
  # the hacky way to embeed any files into the OpenCV without modification of its build system
  if(COMMAND ocv_get_module_external_sources)
    ocv_get_module_external_sources()
  endif()

  set(OPENCV_MODULE_${the_module}_HEADERS ${OPENCV_MODULE_${the_module}_HEADERS} CACHE INTERNAL "List of header files for ${the_module}")
  set(OPENCV_MODULE_${the_module}_SOURCES ${OPENCV_MODULE_${the_module}_SOURCES} CACHE INTERNAL "List of source files for ${the_module}")
endmacro()

# finds and sets headers and sources for the standard OpenCV module
# Usage:
# ocv_glob_module_sources(<extra sources&headers in the same format as used in ocv_set_module_sources>)
macro(ocv_glob_module_sources)
  file(GLOB lib_srcs "src/*.cpp")
  file(GLOB lib_int_hdrs "src/*.hpp" "src/*.h")
  file(GLOB lib_hdrs "include/opencv2/${name}/*.hpp" "include/opencv2/${name}/*.h")
  file(GLOB lib_hdrs_detail "include/opencv2/${name}/detail/*.hpp" "include/opencv2/${name}/detail/*.h")

  source_group("Src" FILES ${lib_srcs} ${lib_int_hdrs})
  source_group("Include" FILES ${lib_hdrs})
  source_group("Include\\detail" FILES ${lib_hdrs_detail})

  ocv_set_module_sources(${ARGN} HEADERS ${lib_hdrs} ${lib_hdrs_detail} SOURCES ${lib_srcs} ${lib_int_hdrs})
endmacro()

# creates OpenCV module in current folder
# creates new target, configures standard dependencies, compilers flags, install rules
# Usage:
# ocv_create_module(<extra link dependencies>)
macro(ocv_create_module)
  add_library(${the_module} ${OPENCV_MODULE_TYPE} ${OPENCV_MODULE_${the_module}_HEADERS} ${OPENCV_MODULE_${the_module}_SOURCES})
  target_link_libraries(${the_module} ${OPENCV_MODULE_${the_module}_DEPS} ${OPENCV_MODULE_${the_module}_DEPS_EXT} ${OPENCV_LINKER_LIBS} ${IPP_LIBS} ${ARGN})
  add_dependencies(opencv_modules ${the_module})
  
  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_module} PROPERTIES FOLDER "modules")
  endif()
  
  set_target_properties(${the_module} PROPERTIES
    OUTPUT_NAME "${the_module}${OPENCV_DLLVERSION}"
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    INSTALL_NAME_DIR lib
  )
  
  # For dynamic link numbering convenions
  if(NOT ANDROID)
    # Android SDK build scripts can include only .so files into final .apk
    # As result we should not set version properties for Android
    set_target_properties(${the_module} PROPERTIES
      VERSION ${OPENCV_VERSION}
      SOVERSION ${OPENCV_SOVERSION}
    )
  endif()

  if(BUILD_SHARED_LIBS)
    if(MSVC)
      set_target_properties(${the_module} PROPERTIES DEFINE_SYMBOL CVAPI_EXPORTS)
    else()
      add_definitions(-DCVAPI_EXPORTS)
    endif()
  endif()

  if(MSVC)
    if(CMAKE_CROSSCOMPILING)
      set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:secchk")
    endif()
      set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libc /DEBUG")
  endif()

  install(TARGETS ${the_module}
    RUNTIME DESTINATION bin COMPONENT main
    LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main
    ARCHIVE DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT main

    )

  # only "public" headers need to be installed
  if(OPENCV_MODULE_${the_module}_HEADERS AND OPENCV_MODULES_PUBLIC MATCHES "(^|;)${the_module}(;|$)")
    foreach(hdr ${OPENCV_MODULE_${the_module}_HEADERS})
      if(hdr MATCHES "(opencv2/.*)/[^/]+.h(..)?$")
        install(FILES ${hdr} DESTINATION "${OPENCV_INCLUDE_PREFIX}/${CMAKE_MATCH_1}" COMPONENT main)
      endif()
    endforeach()
  endif()
endmacro()

# opencv precompiled headers macro (can add pch to modules and tests)
# this macro must be called after any "add_definitions" commands, otherwise precompiled headers will not work
# Usage:
# ocv_add_precompiled_headers(${the_module})
macro(ocv_add_precompiled_headers the_target)
    if("${the_target}" MATCHES "^opencv_test_.*$")
        SET(pch_path "test/test_")
    elseif("${the_target}" MATCHES "^opencv_perf_.*$")
        SET(pch_path "perf/perf_")
    else()
        SET(pch_path "src/")
    endif()
    set(pch_header "${CMAKE_CURRENT_SOURCE_DIR}/${pch_path}precomp.hpp")
    
    if(PCHSupport_FOUND AND ENABLE_PRECOMPILED_HEADERS AND EXISTS "${pch_header}")
        if(CMAKE_GENERATOR MATCHES Visual)
            set(${the_target}_pch "${CMAKE_CURRENT_SOURCE_DIR}/${pch_path}precomp.cpp")
            add_native_precompiled_header(${the_target} ${pch_header})
        elseif(CMAKE_GENERATOR MATCHES Xcode)
            add_native_precompiled_header(${the_target} ${pch_header})
        elseif(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_GENERATOR MATCHES Makefiles)
            add_precompiled_header(${the_target} ${pch_header})
        endif()
    endif()
    unset(pch_header)
    unset(pch_path)
    unset(${the_target}_pch)
endmacro()

# short command for adding simple OpenCV module
# see ocv_add_module for argument details
# Usage:
# ocv_define_module(module_name  [INTERNAL] [REQUIRED] [<list of dependencies>] [OPTIONAL <list of optional dependencies>])
macro(ocv_define_module module_name)
  ocv_add_module(${module_name} ${ARGN})
  ocv_glob_module_sources()
  ocv_module_include_directories()
  ocv_create_module()
  ocv_add_precompiled_headers(${the_module})

  ocv_add_accuracy_tests()
  ocv_add_perf_tests()
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

#auxiliary macro to parse arguments of ocv_add_accuracy_tests and ocv_add_perf_tests commands
macro(__ocv_parse_test_sources tests_type)
  set(OPENCV_${tests_type}_${the_module}_SOURCES "")
  set(OPENCV_${tests_type}_${the_module}_DEPS "")
  set(__file_group_name "")
  set(__file_group_sources "")
  foreach(arg "DEPENDS_ON" ${ARGN} "FILES")
    if(arg STREQUAL "FILES")
      set(__currentvar "__file_group_sources")
      if(__file_group_name AND __file_group_sources)
        source_group("${__file_group_name}" FILES ${__file_group_sources})
        list(APPEND OPENCV_${tests_type}_${the_module}_SOURCES ${__file_group_sources})
      endif()
      set(__file_group_name "")
      set(__file_group_sources "")
    elseif(arg STREQUAL "DEPENDS_ON")
      set(__currentvar "OPENCV_TEST_${the_module}_DEPS")
    elseif("${__currentvar}" STREQUAL "__file_group_sources" AND NOT __file_group_name)
      set(__file_group_name "${arg}")
    else()
      list(APPEND ${__currentvar} "${arg}")
    endif()
  endforeach()
  unset(__file_group_name)
  unset(__file_group_sources)
  unset(__currentvar)
endmacro()

# this is a command for adding OpenCV performance tests to the module
# ocv_add_perf_tests(<extra_dependencies>)
macro(ocv_add_perf_tests)
  set(perf_path "${CMAKE_CURRENT_SOURCE_DIR}/perf")
  if(BUILD_PERF_TESTS AND EXISTS "${perf_path}")
    __ocv_parse_test_sources(PERF ${ARGN})

    # opencv_highgui is required for imread/imwrite
    set(perf_deps ${the_module} opencv_ts opencv_highgui ${OPENCV_PERF_${the_module}_DEPS})
    ocv_check_dependencies(${perf_deps})

    if(OCV_DEPENDENCIES_FOUND)
      set(the_target "opencv_perf_${name}")
      #project(${the_target})
    
      ocv_module_include_directories(${perf_deps} "${perf_path}")

      if(NOT OPENCV_PERF_${the_module}_SOURCES)
        file(GLOB perf_srcs "${perf_path}/*.cpp")
        file(GLOB perf_hdrs "${perf_path}/*.hpp" "${perf_path}/*.h")
        source_group("Src" FILES ${perf_srcs})
        source_group("Include" FILES ${perf_hdrs})
        set(OPENCV_PERF_${the_module}_SOURCES ${perf_srcs} ${perf_hdrs})
      endif()

      add_executable(${the_target} ${OPENCV_PERF_${the_module}_SOURCES})
      target_link_libraries(${the_target} ${OPENCV_MODULE_${the_module}_DEPS} ${perf_deps} ${OPENCV_LINKER_LIBS})
      add_dependencies(opencv_perf_tests ${the_target})

      # Additional target properties
      set_target_properties(${the_target} PROPERTIES
        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
      )

      if(ENABLE_SOLUTION_FOLDERS)
        set_target_properties(${the_target} PROPERTIES FOLDER "tests performance")
      endif()

      ocv_add_precompiled_headers(${the_target})

      if (PYTHON_EXECUTABLE)
        add_dependencies(perf ${the_target})
      endif()
    else(OCV_DEPENDENCIES_FOUND)
      #TODO: warn about unsatisfied dependencies
    endif(OCV_DEPENDENCIES_FOUND)
  endif()
endmacro()

# this is a command for adding OpenCV accuracy/regression tests to the module
# ocv_add_accuracy_tests([FILES <source group name> <list of sources>] [DEPENDS_ON] <list of extra dependencies>)
macro(ocv_add_accuracy_tests)
  set(test_path "${CMAKE_CURRENT_SOURCE_DIR}/test")
  ocv_check_dependencies(${test_deps})
  if(BUILD_TESTS AND EXISTS "${test_path}")
    __ocv_parse_test_sources(TEST ${ARGN})

    # opencv_highgui is required for imread/imwrite
    set(test_deps ${the_module} opencv_ts opencv_highgui ${OPENCV_TEST_${the_module}_DEPS})
    ocv_check_dependencies(${test_deps})

    if(OCV_DEPENDENCIES_FOUND)
      set(the_target "opencv_test_${name}")
      #project(${the_target})
    
      ocv_module_include_directories(${test_deps} "${test_path}")

      if(NOT OPENCV_TEST_${the_module}_SOURCES)
        file(GLOB test_srcs "${test_path}/*.cpp")
        file(GLOB test_hdrs "${test_path}/*.hpp" "${test_path}/*.h")
        source_group("Src" FILES ${test_srcs})
        source_group("Include" FILES ${test_hdrs})
        set(OPENCV_TEST_${the_module}_SOURCES ${test_srcs} ${test_hdrs})
      endif()
    
      add_executable(${the_target} ${OPENCV_TEST_${the_module}_SOURCES})
      target_link_libraries(${the_target} ${OPENCV_MODULE_${the_module}_DEPS} ${test_deps} ${OPENCV_LINKER_LIBS})
      add_dependencies(opencv_tests ${the_target})

      # Additional target properties
      set_target_properties(${the_target} PROPERTIES
        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
      )

      if(ENABLE_SOLUTION_FOLDERS)
        set_target_properties(${the_target} PROPERTIES FOLDER "tests accuracy")
      endif()
        
      enable_testing()
      get_target_property(LOC ${the_target} LOCATION)
      add_test(${the_target} "${LOC}")

      ocv_add_precompiled_headers(${the_target})
    else(OCV_DEPENDENCIES_FOUND)
      #TODO: warn about unsatisfied dependencies
    endif(OCV_DEPENDENCIES_FOUND)
  endif()
endmacro()

# internal macro; finds all link dependencies of module
# should be used at the end of CMake processing
macro(__ocv_track_module_link_dependencies the_module optkind)
  set(${the_module}_MODULE_DEPS_${optkind}   "")
  set(${the_module}_EXTRA_DEPS_${optkind}    "")

  get_target_property(__module_type ${the_module} TYPE)
  if(__module_type STREQUAL "STATIC_LIBRARY")
    #in case of static library we have to inherit its dependencies (in right order!!!)
    if(NOT DEFINED ${the_module}_LIB_DEPENDS_${optkind})
      ocv_split_libs_list(${the_module}_LIB_DEPENDS ${the_module}_LIB_DEPENDS_DBG ${the_module}_LIB_DEPENDS_OPT)
    endif()

    set(__resolved_deps "")
    set(__mod_depends ${${the_module}_LIB_DEPENDS_${optkind}})
    set(__has_cycle FALSE)

    while(__mod_depends)
      list(GET __mod_depends 0 __dep)
      list(REMOVE_AT __mod_depends 0)
      if(__dep STREQUAL the_module)
        set(__has_cycle TRUE)
      else()#if("${OPENCV_MODULES_BUILD}" MATCHES "(^|;)${__dep}(;|$)")
        ocv_regex_escape(__rdep "${__dep}")
        if(__resolved_deps MATCHES "(^|;)${__rdep}(;|$)")
          #all dependencies of this module are already resolved
          list(APPEND ${the_module}_MODULE_DEPS_${optkind} "${__dep}")
        else()
          get_target_property(__module_type ${__dep} TYPE)
          if(__module_type STREQUAL "STATIC_LIBRARY")
            if(NOT DEFINED ${__dep}_LIB_DEPENDS_${optkind})
              ocv_split_libs_list(${__dep}_LIB_DEPENDS ${__dep}_LIB_DEPENDS_DBG ${__dep}_LIB_DEPENDS_OPT)
            endif()
            list(INSERT __mod_depends 0 ${${__dep}_LIB_DEPENDS_${optkind}} ${__dep})
            list(APPEND __resolved_deps "${__dep}")
          elseif(NOT __module_type)
            list(APPEND  ${the_module}_EXTRA_DEPS_${optkind} "${__dep}")
          endif()
        endif()
      #else()
       # get_target_property(__dep_location "${__dep}" LOCATION)
      endif()
    endwhile()
   
    ocv_list_unique(${the_module}_MODULE_DEPS_${optkind})
    #ocv_list_reverse(${the_module}_MODULE_DEPS_${optkind})
    ocv_list_unique(${the_module}_EXTRA_DEPS_${optkind})
    #ocv_list_reverse(${the_module}_EXTRA_DEPS_${optkind})

    if(__has_cycle)
      #not sure if it can work
      list(APPEND ${the_module}_MODULE_DEPS_${optkind} "${the_module}")
    endif()

    unset(__dep_location)
    unset(__mod_depends)
    unset(__resolved_deps)
    unset(__has_cycle)
    unset(__rdep)
  endif()#STATIC_LIBRARY
  unset(__module_type)

#message("${the_module}_MODULE_DEPS_${optkind}")
#message("       ${${the_module}_MODULE_DEPS_${optkind}}")
#message("       ${OPENCV_MODULE_${the_module}_DEPS}")
#message("")
#message("${the_module}_EXTRA_DEPS_${optkind}")
#message("       ${${the_module}_EXTRA_DEPS_${optkind}}")
#message("")
endmacro()

# creates lists of build dependencies needed for external projects
macro(ocv_track_build_dependencies)
  foreach(m ${OPENCV_MODULES_BUILD})
    __ocv_track_module_link_dependencies("${m}" OPT)
    __ocv_track_module_link_dependencies("${m}" DBG)
  endforeach()
endmacro()
