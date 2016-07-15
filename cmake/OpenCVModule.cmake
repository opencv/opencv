# Local variables (set for each module):
#
# name       - short name in lower case i.e. core
# the_module - full name in lower case i.e. opencv_core

# Global variables:
#
# OPENCV_MODULE_${the_module}_LOCATION
# OPENCV_MODULE_${the_module}_BINARY_DIR
# OPENCV_MODULE_${the_module}_DESCRIPTION
# OPENCV_MODULE_${the_module}_CLASS - PUBLIC|INTERNAL|BINDINGS
# OPENCV_MODULE_${the_module}_HEADERS
# OPENCV_MODULE_${the_module}_SOURCES
# OPENCV_MODULE_${the_module}_DEPS - final flattened set of module dependencies
# OPENCV_MODULE_${the_module}_DEPS_TO_LINK - differs from above for world build only
# OPENCV_MODULE_${the_module}_DEPS_EXT - non-module dependencies
# OPENCV_MODULE_${the_module}_REQ_DEPS
# OPENCV_MODULE_${the_module}_OPT_DEPS
# OPENCV_MODULE_${the_module}_PRIVATE_REQ_DEPS
# OPENCV_MODULE_${the_module}_PRIVATE_OPT_DEPS
# OPENCV_MODULE_${the_module}_IS_PART_OF_WORLD
# OPENCV_MODULE_${the_module}_CUDA_OBJECTS - compiled CUDA objects list
# OPENCV_MODULE_${the_module}_CHILDREN - list of submodules for compound modules (cmake >= 2.8.8)
# OPENCV_MODULE_${the_module}_WRAPPERS - list of wrappers supporting this module
# HAVE_${the_module} - for fast check of module availability

# To control the setup of the module you could also set:
# the_description - text to be used as current module description
# the_label - label for current module
# OPENCV_MODULE_TYPE - STATIC|SHARED - set to force override global settings for current module
# OPENCV_MODULE_IS_PART_OF_WORLD - ON|OFF (default ON) - should the module be added to the opencv_world?
# BUILD_${the_module}_INIT - ON|OFF (default ON) - initial value for BUILD_${the_module}
# OPENCV_MODULE_CHILDREN - list of submodules

# The verbose template for OpenCV module:
#
#   ocv_add_module(modname <dependencies>)
#   ocv_glob_module_sources(([EXCLUDE_CUDA] <extra sources&headers>)
#                          or glob them manually and ocv_set_module_sources(...)
#   ocv_module_include_directories(<extra include directories>)
#   ocv_create_module()
#   <add extra link dependencies, compiler options, etc>
#   ocv_add_precompiled_headers(${the_module})
#   <add extra installation rules>
#   ocv_add_accuracy_tests(<extra dependencies>)
#   ocv_add_perf_tests(<extra dependencies>)
#   ocv_add_samples(<extra dependencies>)
#
#
# If module have no "extra" then you can define it in one line:
#
#   ocv_define_module(modname <dependencies>)

# clean flags for modules enabled on previous cmake run
# this is necessary to correctly handle modules removal
foreach(mod ${OPENCV_MODULES_BUILD} ${OPENCV_MODULES_DISABLED_USER} ${OPENCV_MODULES_DISABLED_AUTO} ${OPENCV_MODULES_DISABLED_FORCE})
  if(HAVE_${mod})
    unset(HAVE_${mod} CACHE)
  endif()
  unset(OPENCV_MODULE_${mod}_DEPS CACHE)
  unset(OPENCV_MODULE_${mod}_DEPS_EXT CACHE)
  unset(OPENCV_MODULE_${mod}_REQ_DEPS CACHE)
  unset(OPENCV_MODULE_${mod}_OPT_DEPS CACHE)
  unset(OPENCV_MODULE_${mod}_PRIVATE_REQ_DEPS CACHE)
  unset(OPENCV_MODULE_${mod}_PRIVATE_OPT_DEPS CACHE)
  unset(OPENCV_MODULE_${mod}_LINK_DEPS CACHE)
  unset(OPENCV_MODULE_${mod}_WRAPPERS CACHE)
endforeach()

# clean modules info which needs to be recalculated
set(OPENCV_MODULES_PUBLIC         "" CACHE INTERNAL "List of OpenCV modules marked for export")
set(OPENCV_MODULES_BUILD          "" CACHE INTERNAL "List of OpenCV modules included into the build")
set(OPENCV_MODULES_DISABLED_USER  "" CACHE INTERNAL "List of OpenCV modules explicitly disabled by user")
set(OPENCV_MODULES_DISABLED_AUTO  "" CACHE INTERNAL "List of OpenCV modules implicitly disabled due to dependencies")
set(OPENCV_MODULES_DISABLED_FORCE "" CACHE INTERNAL "List of OpenCV modules which can not be build in current configuration")
unset(OPENCV_WORLD_MODULES CACHE)

# adds dependencies to OpenCV module
# Usage:
#   add_dependencies(opencv_<name> [REQUIRED] [<list of dependencies>] [OPTIONAL <list of modules>] [WRAP <list of wrappers>])
# Notes:
# * <list of dependencies> - can include full names of modules or full pathes to shared/static libraries or cmake targets
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

  # hack for python
  set(__python_idx)
  list(FIND OPENCV_MODULE_${full_modname}_WRAPPERS "python" __python_idx)
  if (NOT __python_idx EQUAL -1)
    list(REMOVE_ITEM OPENCV_MODULE_${full_modname}_WRAPPERS "python")
    list(APPEND OPENCV_MODULE_${full_modname}_WRAPPERS "python2" "python3")
  endif()
  unset(__python_idx)

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

  # the first pass - collect modules info, the second pass - create targets
  if(OPENCV_INITIAL_PASS)
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

    set(OPENCV_MODULE_${the_module}_LINK_DEPS "" CACHE INTERNAL "")

    # parse list of dependencies
    if("${ARGV1}" STREQUAL "INTERNAL" OR "${ARGV1}" STREQUAL "BINDINGS")
      set(OPENCV_MODULE_${the_module}_CLASS "${ARGV1}" CACHE INTERNAL "The category of the module")
      set(__ocv_argn__ ${ARGN})
      list(REMOVE_AT __ocv_argn__ 0)
      ocv_add_dependencies(${the_module} ${__ocv_argn__})
      unset(__ocv_argn__)
    else()
      set(OPENCV_MODULE_${the_module}_CLASS "PUBLIC" CACHE INTERNAL "The category of the module")
      ocv_add_dependencies(${the_module} ${ARGN})
      if(BUILD_${the_module})
        set(OPENCV_MODULES_PUBLIC ${OPENCV_MODULES_PUBLIC} "${the_module}" CACHE INTERNAL "List of OpenCV modules marked for export")
      endif()
    endif()

    # add self to the world dependencies
    if((NOT DEFINED OPENCV_MODULE_IS_PART_OF_WORLD
        AND NOT OPENCV_MODULE_${the_module}_CLASS STREQUAL "BINDINGS"
        AND NOT OPENCV_PROCESSING_EXTRA_MODULES
        AND (NOT BUILD_SHARED_LIBS OR NOT "x${OPENCV_MODULE_TYPE}" STREQUAL "xSTATIC"))
        OR OPENCV_MODULE_IS_PART_OF_WORLD
        )
      set(OPENCV_MODULE_${the_module}_IS_PART_OF_WORLD ON CACHE INTERNAL "")
      ocv_add_dependencies(opencv_world OPTIONAL ${the_module})
    else()
      set(OPENCV_MODULE_${the_module}_IS_PART_OF_WORLD OFF CACHE INTERNAL "")
    endif()

    if(NOT DEFINED the_label)
      if(OPENCV_PROCESSING_EXTRA_MODULES)
        set(the_label "Extra")
      else()
        set(the_label "Main")
      endif()
    endif()
    set(OPENCV_MODULE_${the_module}_LABEL "${the_label};${the_module}" CACHE INTERNAL "")

    if(BUILD_${the_module})
      set(OPENCV_MODULES_BUILD ${OPENCV_MODULES_BUILD} "${the_module}" CACHE INTERNAL "List of OpenCV modules included into the build")
    else()
      set(OPENCV_MODULES_DISABLED_USER ${OPENCV_MODULES_DISABLED_USER} "${the_module}" CACHE INTERNAL "List of OpenCV modules explicitly disabled by user")
    endif()

    # add submodules if any
    set(OPENCV_MODULE_${the_module}_CHILDREN "${OPENCV_MODULE_CHILDREN}" CACHE INTERNAL "List of ${the_module} submodules")

    # add reverse wrapper dependencies
    foreach (wrapper ${OPENCV_MODULE_${the_module}_WRAPPERS})
      ocv_add_dependencies(opencv_${wrapper} OPTIONAL ${the_module})
    endforeach()

    # stop processing of current file
    return()
  else()
    set(OPENCV_MODULE_${the_module}_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "")
    if(NOT BUILD_${the_module})
      return() # extra protection from redefinition
    endif()
    if((NOT OPENCV_MODULE_${the_module}_IS_PART_OF_WORLD AND NOT ${the_module} STREQUAL opencv_world) OR NOT ${BUILD_opencv_world})
      project(${the_module})
    endif()
  endif()
endmacro()

# excludes module from current configuration
macro(ocv_module_disable module)
  set(__modname ${module})
  if(NOT __modname MATCHES "^opencv_")
    set(__modname opencv_${module})
  endif()
  list(APPEND OPENCV_MODULES_DISABLED_FORCE "${__modname}")
  set(HAVE_${__modname} OFF CACHE INTERNAL "Module ${__modname} can not be built in current configuration")
  set(OPENCV_MODULE_${__modname}_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Location of ${__modname} module sources")
  set(OPENCV_MODULES_DISABLED_FORCE "${OPENCV_MODULES_DISABLED_FORCE}" CACHE INTERNAL "List of OpenCV modules which can not be build in current configuration")
  if(BUILD_${__modname})
    # touch variable controlling build of the module to suppress "unused variable" CMake warning
  endif()
  unset(__modname)
  return() # leave the current folder
endmacro()


# collect modules from specified directories
# NB: must be called only once!
macro(ocv_glob_modules)
  if(DEFINED OPENCV_INITIAL_PASS)
    message(FATAL_ERROR "OpenCV has already loaded its modules. Calling ocv_glob_modules second time is not allowed.")
  endif()
  set(__directories_observed "")

  # collect modules
  set(OPENCV_INITIAL_PASS ON)
  set(OPENCV_PROCESSING_EXTRA_MODULES 0)
  foreach(__path ${ARGN})
    if("${__path}" STREQUAL "EXTRA")
      set(OPENCV_PROCESSING_EXTRA_MODULES 1)
    else()
      get_filename_component(__path "${__path}" ABSOLUTE)

      list(FIND __directories_observed "${__path}" __pathIdx)
      if(__pathIdx GREATER -1)
        message(FATAL_ERROR "The directory ${__path} is observed for OpenCV modules second time.")
      endif()
      list(APPEND __directories_observed "${__path}")

      set(__count 0)
      file(GLOB __ocvmodules RELATIVE "${__path}" "${__path}/*")
      if(__ocvmodules)
        list(SORT __ocvmodules)
        foreach(mod ${__ocvmodules})
          get_filename_component(__modpath "${__path}/${mod}" ABSOLUTE)
          if(EXISTS "${__modpath}/CMakeLists.txt")

            list(FIND __directories_observed "${__modpath}" __pathIdx)
            if(__pathIdx GREATER -1)
              message(FATAL_ERROR "The module from ${__modpath} is already loaded.")
            endif()
            list(APPEND __directories_observed "${__modpath}")

            add_subdirectory("${__modpath}" "${CMAKE_CURRENT_BINARY_DIR}/${mod}/.${mod}")

            if (DEFINED OPENCV_MODULE_opencv_${mod}_LOCATION)
              math(EXPR __count "${__count} + 1")
            endif()
          endif()
        endforeach()
      endif()
      if (OPENCV_PROCESSING_EXTRA_MODULES AND ${__count} LESS 1)
        message(SEND_ERROR "No extra modules found in folder: ${__path}\nPlease provide path to 'opencv_contrib/modules' folder.")
      endif()
    endif()
  endforeach()
  ocv_clear_vars(__ocvmodules __directories_observed __path __modpath __pathIdx)

  # resolve dependencies
  __ocv_resolve_dependencies()

  # create modules
  set(OPENCV_INITIAL_PASS OFF PARENT_SCOPE)
  set(OPENCV_INITIAL_PASS OFF)
  if(${BUILD_opencv_world})
    foreach(m ${OPENCV_MODULES_BUILD})
      if("${m}" STREQUAL opencv_world)
        add_subdirectory("${OPENCV_MODULE_opencv_world_LOCATION}" "${CMAKE_CURRENT_BINARY_DIR}/world")
      elseif(NOT OPENCV_MODULE_${m}_IS_PART_OF_WORLD AND NOT ${m} STREQUAL opencv_world)
        message(STATUS "Processing module ${m}...")
        if(m MATCHES "^opencv_")
          string(REGEX REPLACE "^opencv_" "" __shortname "${m}")
          add_subdirectory("${OPENCV_MODULE_${m}_LOCATION}" "${CMAKE_CURRENT_BINARY_DIR}/${__shortname}")
        else()
          message(WARNING "Check module name: ${m}")
          add_subdirectory("${OPENCV_MODULE_${m}_LOCATION}" "${CMAKE_CURRENT_BINARY_DIR}/${m}")
        endif()
      endif()
    endforeach()
  else()
    foreach(m ${OPENCV_MODULES_BUILD})
      if(m MATCHES "^opencv_")
        string(REGEX REPLACE "^opencv_" "" __shortname "${m}")
        add_subdirectory("${OPENCV_MODULE_${m}_LOCATION}" "${CMAKE_CURRENT_BINARY_DIR}/${__shortname}")
      else()
        message(WARNING "Check module name: ${m}")
        add_subdirectory("${OPENCV_MODULE_${m}_LOCATION}" "${CMAKE_CURRENT_BINARY_DIR}/${m}")
      endif()
    endforeach()
  endif()
  unset(__shortname)
endmacro()


# disables OpenCV module with missing dependencies
function(__ocv_module_turn_off the_module)
  list(REMOVE_ITEM OPENCV_MODULES_DISABLED_AUTO "${the_module}")
  list(APPEND OPENCV_MODULES_DISABLED_AUTO "${the_module}")
  list(REMOVE_ITEM OPENCV_MODULES_BUILD "${the_module}")
  list(REMOVE_ITEM OPENCV_MODULES_PUBLIC "${the_module}")
  set(HAVE_${the_module} OFF CACHE INTERNAL "Module ${the_module} can not be built in current configuration")

  set(OPENCV_MODULES_DISABLED_AUTO "${OPENCV_MODULES_DISABLED_AUTO}" CACHE INTERNAL "")
  set(OPENCV_MODULES_BUILD "${OPENCV_MODULES_BUILD}" CACHE INTERNAL "")
  set(OPENCV_MODULES_PUBLIC "${OPENCV_MODULES_PUBLIC}" CACHE INTERNAL "")
endfunction()

# sort modules by dependencies
function(__ocv_sort_modules_by_deps __lst)
  ocv_list_sort(${__lst})
  set(input ${${__lst}})
  set(result "")
  set(result_extra "")
  while(input)
    list(LENGTH input length_before)
    foreach (m ${input})
      # check if module is in the result already
      if (NOT ";${result};" MATCHES ";${m};")
        # scan through module dependencies...
        set(unresolved_deps_found FALSE)
        foreach (d ${OPENCV_MODULE_${m}_CHILDREN} ${OPENCV_MODULE_${m}_DEPS})
          # ... which are not already in the result and are enabled
          if ((NOT ";${result};" MATCHES ";${d};") AND HAVE_${d})
            set(unresolved_deps_found TRUE)
            break()
          endif()
        endforeach()
        # chek if all dependencies for this module has been resolved
        if (NOT unresolved_deps_found)
          list(APPEND result ${m})
          list(REMOVE_ITEM input ${m})
        endif()
      endif()
    endforeach()
    list(LENGTH input length_after)
    # check for infinite loop or unresolved dependencies
    if (NOT length_after LESS length_before)
      if(NOT BUILD_SHARED_LIBS)
        if (";${input};" MATCHES ";opencv_world;")
          list(REMOVE_ITEM input "opencv_world")
          list(APPEND result_extra "opencv_world")
        else()
          # We can't do here something
          list(APPEND result ${input})
          break()
        endif()
      else()
        message(FATAL_ERROR WARNING "Unresolved dependencies or loop in dependency graph (${length_after})\n"
          "Processed ${__lst}: ${${__lst}}\n"
          "Good modules: ${result}\n"
          "Bad modules: ${input}"
        )
        list(APPEND result ${input})
        break()
      endif()
    endif()
  endwhile()
  set(${__lst} "${result};${result_extra}" PARENT_SCOPE)
endfunction()

# resolve dependensies
function(__ocv_resolve_dependencies)
  foreach(m ${OPENCV_MODULES_DISABLED_USER})
    set(HAVE_${m} OFF CACHE INTERNAL "Module ${m} will not be built in current configuration")
  endforeach()
  foreach(m ${OPENCV_MODULES_BUILD})
    set(HAVE_${m} ON CACHE INTERNAL "Module ${m} will be built in current configuration")
  endforeach()

  # disable MODULES with unresolved dependencies
  set(has_changes ON)
  while(has_changes)
    set(has_changes OFF)
    foreach(m ${OPENCV_MODULES_BUILD})
      set(__deps ${OPENCV_MODULE_${m}_REQ_DEPS} ${OPENCV_MODULE_${m}_PRIVATE_REQ_DEPS})
      while(__deps)
        ocv_list_pop_front(__deps d)
        string(TOLOWER "${d}" upper_d)
        if(NOT (HAVE_${d} OR HAVE_${upper_d} OR TARGET ${d} OR EXISTS ${d}))
          if(d MATCHES "^opencv_") # TODO Remove this condition in the future and use HAVE_ variables only
            message(STATUS "Module ${m} disabled because ${d} dependency can't be resolved!")
            __ocv_module_turn_off(${m})
            set(has_changes ON)
            break()
          else()
            message(STATUS "Assume that non-module dependency is available: ${d} (for module ${m})")
          endif()
        endif()
      endwhile()
    endforeach()
  endwhile()

#  message(STATUS "List of active modules: ${OPENCV_MODULES_BUILD}")

  foreach(m ${OPENCV_MODULES_BUILD})
    set(deps_${m} ${OPENCV_MODULE_${m}_REQ_DEPS})
    foreach(d ${OPENCV_MODULE_${m}_OPT_DEPS})
      if(NOT (";${deps_${m}};" MATCHES ";${d};"))
        if(HAVE_${d} OR TARGET ${d})
          list(APPEND deps_${m} ${d})
        endif()
      endif()
    endforeach()
#    message(STATUS "Initial deps of ${m} (w/o private deps): ${deps_${m}}")
  endforeach()

  # propagate dependencies
  set(has_changes ON)
  while(has_changes)
    set(has_changes OFF)
    foreach(m2 ${OPENCV_MODULES_BUILD}) # transfer deps of m2 to m
      foreach(m ${OPENCV_MODULES_BUILD})
        if((NOT m STREQUAL m2) AND ";${deps_${m}};" MATCHES ";${m2};")
          foreach(d ${deps_${m2}})
            if(NOT (";${deps_${m}};" MATCHES ";${d};"))
#              message(STATUS "  Transfer dependency ${d} from ${m2} to ${m}")
              list(APPEND deps_${m} ${d})
              set(has_changes ON)
            endif()
            if(BUILD_opencv_world
                AND NOT "${m}" STREQUAL "opencv_world"
                AND NOT "${m2}" STREQUAL "opencv_world"
                AND OPENCV_MODULE_${m2}_IS_PART_OF_WORLD
                AND NOT OPENCV_MODULE_${m}_IS_PART_OF_WORLD)
              if(NOT (";${deps_${m}};" MATCHES ";opencv_world;"))
#                message(STATUS "  Transfer dependency opencv_world alias ${m2} to ${m}")
                list(APPEND deps_${m} opencv_world)
                set(has_changes ON)
              endif()
            endif()
          endforeach()
        endif()
      endforeach()
    endforeach()
  endwhile()

  # process private deps
  foreach(m ${OPENCV_MODULES_BUILD})
    foreach(d ${OPENCV_MODULE_${m}_PRIVATE_REQ_DEPS})
      if(NOT (";${deps_${m}};" MATCHES ";${d};"))
        list(APPEND deps_${m} ${d})
      endif()
    endforeach()
    foreach(d ${OPENCV_MODULE_${m}_PRIVATE_OPT_DEPS})
      if(NOT (";${deps_${m}};" MATCHES ";${d};"))
        if(HAVE_${d} OR TARGET ${d})
          list(APPEND deps_${m} ${d})
        endif()
      endif()
    endforeach()
  endforeach()

  ocv_list_sort(OPENCV_MODULES_BUILD)

  foreach(m ${OPENCV_MODULES_BUILD})
#    message(STATUS "FULL deps of ${m}: ${deps_${m}}")
    set(OPENCV_MODULE_${m}_DEPS ${deps_${m}})
    set(OPENCV_MODULE_${m}_DEPS_EXT ${deps_${m}})
    ocv_list_filterout(OPENCV_MODULE_${m}_DEPS_EXT "^opencv_[^ ]+$")
    if(OPENCV_MODULE_${m}_DEPS_EXT AND OPENCV_MODULE_${m}_DEPS)
      list(REMOVE_ITEM OPENCV_MODULE_${m}_DEPS ${OPENCV_MODULE_${m}_DEPS_EXT})
    endif()
  endforeach()

  # reorder dependencies
  foreach(m ${OPENCV_MODULES_BUILD})
    __ocv_sort_modules_by_deps(OPENCV_MODULE_${m}_DEPS)

    set(LINK_DEPS ${OPENCV_MODULE_${m}_DEPS})

    # process world
    if(BUILD_opencv_world)
      if(OPENCV_MODULE_${m}_IS_PART_OF_WORLD)
        list(APPEND OPENCV_WORLD_MODULES ${m})
      endif()
      foreach(m2 ${OPENCV_MODULES_BUILD})
        if(OPENCV_MODULE_${m2}_IS_PART_OF_WORLD)
          if(";${LINK_DEPS};" MATCHES ";${m2};")
            list(REMOVE_ITEM LINK_DEPS ${m2})
            if(NOT (";${LINK_DEPS};" MATCHES ";opencv_world;") AND NOT (${m} STREQUAL opencv_world))
              list(APPEND LINK_DEPS opencv_world)
            endif()
          endif()
          if(${m} STREQUAL opencv_world)
            list(APPEND OPENCV_MODULE_opencv_world_DEPS_EXT ${OPENCV_MODULE_${m2}_DEPS_EXT})
          endif()
        endif()
      endforeach()
    endif()

    set(OPENCV_MODULE_${m}_DEPS ${OPENCV_MODULE_${m}_DEPS} CACHE INTERNAL "Flattened dependencies of ${m} module")
    set(OPENCV_MODULE_${m}_DEPS_EXT ${OPENCV_MODULE_${m}_DEPS_EXT} CACHE INTERNAL "Extra dependencies of ${m} module")
    set(OPENCV_MODULE_${m}_DEPS_TO_LINK ${LINK_DEPS} CACHE INTERNAL "Flattened dependencies of ${m} module (for linker)")

#    message(STATUS "  module deps of ${m}: ${OPENCV_MODULE_${m}_DEPS}")
#    message(STATUS "  module link deps of ${m}: ${OPENCV_MODULE_${m}_DEPS_TO_LINK}")
#    message(STATUS "  extra deps of ${m}: ${OPENCV_MODULE_${m}_DEPS_EXT}")
#    message(STATUS "")
  endforeach()

  __ocv_sort_modules_by_deps(OPENCV_MODULES_BUILD)

  set(OPENCV_MODULES_PUBLIC        ${OPENCV_MODULES_PUBLIC}        CACHE INTERNAL "List of OpenCV modules marked for export")
  set(OPENCV_MODULES_BUILD         ${OPENCV_MODULES_BUILD}         CACHE INTERNAL "List of OpenCV modules included into the build")
  set(OPENCV_MODULES_DISABLED_AUTO ${OPENCV_MODULES_DISABLED_AUTO} CACHE INTERNAL "List of OpenCV modules implicitly disabled due to dependencies")
  set(OPENCV_WORLD_MODULES         ${OPENCV_WORLD_MODULES}         CACHE INTERNAL "List of OpenCV modules included into the world")
endfunction()


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
    elseif(EXISTS "${d}")
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
    elseif(EXISTS "${d}")
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
      "${CMAKE_CURRENT_BINARY_DIR}" # for precompiled headers
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

  # the hacky way to embeed any files into the OpenCV without modification of its build system
  if(COMMAND ocv_get_module_external_sources)
    ocv_get_module_external_sources()
  endif()

  # use full paths for module to be independent from the module location
  ocv_convert_to_full_paths(OPENCV_MODULE_${the_module}_HEADERS)

  set(OPENCV_MODULE_${the_module}_HEADERS ${OPENCV_MODULE_${the_module}_HEADERS} CACHE INTERNAL "List of header files for ${the_module}")
  set(OPENCV_MODULE_${the_module}_SOURCES ${OPENCV_MODULE_${the_module}_SOURCES} CACHE INTERNAL "List of source files for ${the_module}")
endmacro()

# finds and sets headers and sources for the standard OpenCV module
# Usage:
# ocv_glob_module_sources([EXCLUDE_CUDA] <extra sources&headers in the same format as used in ocv_set_module_sources>)
macro(ocv_glob_module_sources)
  ocv_debug_message("ocv_glob_module_sources(" ${ARGN} ")")
  set(_argn ${ARGN})
  list(FIND _argn "EXCLUDE_CUDA" exclude_cuda)
  if(NOT exclude_cuda EQUAL -1)
    list(REMOVE_AT _argn ${exclude_cuda})
  endif()

  file(GLOB_RECURSE lib_srcs
       "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"
  )
  file(GLOB_RECURSE lib_int_hdrs
       "${CMAKE_CURRENT_LIST_DIR}/src/*.hpp"
       "${CMAKE_CURRENT_LIST_DIR}/src/*.h"
  )
  file(GLOB lib_hdrs
       "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/*.hpp"
       "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.hpp"
       "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/*.h"
       "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/hal/*.hpp"
       "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/hal/*.h"
  )
  file(GLOB lib_hdrs_detail
       "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/detail/*.hpp"
       "${CMAKE_CURRENT_LIST_DIR}/include/opencv2/${name}/detail/*.h"
  )
  if (APPLE)
    file(GLOB_RECURSE lib_srcs_apple
         "${CMAKE_CURRENT_LIST_DIR}/src/*.mm"
    )
    list(APPEND lib_srcs ${lib_srcs_apple})
  endif()

  ocv_source_group("Src" DIRBASE "${CMAKE_CURRENT_LIST_DIR}/src" FILES ${lib_srcs} ${lib_int_hdrs})
  ocv_source_group("Include" DIRBASE "${CMAKE_CURRENT_LIST_DIR}/include" FILES ${lib_hdrs} ${lib_hdrs_detail})

  set(lib_cuda_srcs "")
  set(lib_cuda_hdrs "")
  if(HAVE_CUDA AND exclude_cuda EQUAL -1)
    file(GLOB lib_cuda_srcs
         "${CMAKE_CURRENT_LIST_DIR}/src/cuda/*.cu"
    )
    file(GLOB lib_cuda_hdrs
         "${CMAKE_CURRENT_LIST_DIR}/src/cuda/*.hpp"
    )
    source_group("Src\\Cuda"      FILES ${lib_cuda_srcs} ${lib_cuda_hdrs})
  endif()

  file(GLOB cl_kernels
       "${CMAKE_CURRENT_LIST_DIR}/src/opencl/*.cl"
  )
  if(cl_kernels)
    set(OCL_NAME opencl_kernels_${name})
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.hpp"
      COMMAND ${CMAKE_COMMAND} "-DMODULE_NAME=${name}" "-DCL_DIR=${CMAKE_CURRENT_LIST_DIR}/src/opencl" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" -P "${OpenCV_SOURCE_DIR}/cmake/cl2cpp.cmake"
      DEPENDS ${cl_kernels} "${OpenCV_SOURCE_DIR}/cmake/cl2cpp.cmake")
    ocv_source_group("Src\\opencl\\kernels" FILES ${cl_kernels})
    ocv_source_group("Src\\opencl\\kernels\\autogenerated" FILES "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.hpp")
    list(APPEND lib_srcs ${cl_kernels} "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.hpp")
  endif()

  ocv_set_module_sources(${_argn} HEADERS ${lib_hdrs} ${lib_hdrs_detail}
                         SOURCES ${lib_srcs} ${lib_int_hdrs} ${lib_cuda_srcs} ${lib_cuda_hdrs})
endmacro()

# creates OpenCV module in current folder
# creates new target, configures standard dependencies, compilers flags, install rules
# Usage:
#   ocv_create_module(<extra link dependencies>)
#   ocv_create_module()
macro(ocv_create_module)
  ocv_debug_message("ocv_create_module(" ${ARGN} ")")
  set(OPENCV_MODULE_${the_module}_LINK_DEPS "${OPENCV_MODULE_${the_module}_LINK_DEPS};${ARGN}" CACHE INTERNAL "")
  if(${BUILD_opencv_world} AND OPENCV_MODULE_${the_module}_IS_PART_OF_WORLD)
    # nothing
    set(the_module_target opencv_world)
  else()
    _ocv_create_module(${ARGN})
    set(the_module_target ${the_module})
  endif()

  if(WINRT)
    # removing APPCONTAINER from modules to run from console
    # in case of usual starting of WinRT test apps output is missing
    # so starting of console version w/o APPCONTAINER is required to get test results
    # also this allows to use opencv_extra test data for these tests
    if(NOT "${the_module}" STREQUAL "opencv_ts" AND NOT "${the_module}" STREQUAL "opencv_hal")
      add_custom_command(TARGET ${the_module}
                         POST_BUILD
                         COMMAND link.exe /edit /APPCONTAINER:NO $(TargetPath))
    endif()

    if("${the_module}" STREQUAL "opencv_ts")
      # copy required dll files; WinRT apps need these dlls that are usually substituted by Visual Studio
      # however they are not on path and need to be placed with executables to run from console w/o APPCONTAINER
      add_custom_command(TARGET ${the_module}
        POST_BUILD
        COMMAND copy /y "\"$(VCInstallDir)redist\\$(PlatformTarget)\\Microsoft.VC$(PlatformToolsetVersion).CRT\\msvcp$(PlatformToolsetVersion).dll\"" "\"${CMAKE_BINARY_DIR}\\bin\\$(Configuration)\\msvcp$(PlatformToolsetVersion)_app.dll\""
        COMMAND copy /y "\"$(VCInstallDir)redist\\$(PlatformTarget)\\Microsoft.VC$(PlatformToolsetVersion).CRT\\msvcr$(PlatformToolsetVersion).dll\"" "\"${CMAKE_BINARY_DIR}\\bin\\$(Configuration)\\msvcr$(PlatformToolsetVersion)_app.dll\""
        COMMAND copy /y "\"$(VCInstallDir)redist\\$(PlatformTarget)\\Microsoft.VC$(PlatformToolsetVersion).CRT\\vccorlib$(PlatformToolsetVersion).dll\"" "\"${CMAKE_BINARY_DIR}\\bin\\$(Configuration)\\vccorlib$(PlatformToolsetVersion)_app.dll\"")
    endif()
  endif()
endmacro()

macro(_ocv_create_module)
  # The condition we ought to be testing here is whether ocv_add_precompiled_headers will
  # be called at some point in the future. We can't look into the future, though,
  # so this will have to do.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/precomp.hpp" AND NOT ${the_module} STREQUAL opencv_world)
    get_native_precompiled_header(${the_module} precomp.hpp)
  endif()

  set(sub_objs "")
  set(sub_links "")
  set(cuda_objs "")
  if (OPENCV_MODULE_${the_module}_CHILDREN)
    message(STATUS "Complex module ${the_module}")
    foreach (m ${OPENCV_MODULE_${the_module}_CHILDREN})
      if (BUILD_${m} AND TARGET ${m}_object)
        get_target_property(_sub_links ${m} LINK_LIBRARIES)
        list(APPEND sub_objs $<TARGET_OBJECTS:${m}_object>)
        list(APPEND sub_links ${_sub_links})
        message(STATUS "    + ${m}")
      else()
        message(STATUS "    - ${m}")
      endif()
      list(APPEND cuda_objs ${OPENCV_MODULE_${m}_CUDA_OBJECTS})
    endforeach()
  endif()

  ocv_add_library(${the_module} ${OPENCV_MODULE_TYPE} ${OPENCV_MODULE_${the_module}_HEADERS} ${OPENCV_MODULE_${the_module}_SOURCES}
    "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/cvconfig.h" "${OPENCV_CONFIG_FILE_INCLUDE_DIR}/opencv2/opencv_modules.hpp"
    ${${the_module}_pch} ${sub_objs})

  if (cuda_objs)
    target_link_libraries(${the_module} ${cuda_objs})
  endif()

  # TODO: is it needed?
  if (sub_links)
    ocv_list_filterout(sub_links "^opencv_")
    ocv_list_unique(sub_links)
    target_link_libraries(${the_module} ${sub_links})
  endif()

  unset(sub_objs)
  unset(sub_links)
  unset(cuda_objs)

  set_target_properties(${the_module} PROPERTIES LABELS "${OPENCV_MODULE_${the_module}_LABEL};Module")
  set_source_files_properties(${OPENCV_MODULE_${the_module}_HEADERS} ${OPENCV_MODULE_${the_module}_SOURCES} ${${the_module}_pch}
    PROPERTIES LABELS "${OPENCV_MODULE_${the_module}_LABEL};Module")

  if(NOT BUILD_SHARED_LIBS OR NOT INSTALL_CREATE_DISTRIB)
    ocv_target_link_libraries(${the_module} ${OPENCV_MODULE_${the_module}_DEPS_TO_LINK})
    ocv_target_link_libraries(${the_module} LINK_INTERFACE_LIBRARIES ${OPENCV_MODULE_${the_module}_DEPS_TO_LINK})
    ocv_target_link_libraries(${the_module} ${OPENCV_MODULE_${the_module}_DEPS_EXT} ${OPENCV_LINKER_LIBS} ${IPP_LIBS} ${ARGN})
    if (HAVE_CUDA)
      ocv_target_link_libraries(${the_module} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
    endif()
  else()
    ocv_target_link_libraries(${the_module} LINK_PRIVATE ${OPENCV_MODULE_${the_module}_DEPS_TO_LINK})
    ocv_target_link_libraries(${the_module} LINK_PRIVATE ${OPENCV_MODULE_${the_module}_DEPS_TO_LINK})
    ocv_target_link_libraries(${the_module} LINK_PRIVATE ${OPENCV_MODULE_${the_module}_DEPS_EXT} ${OPENCV_LINKER_LIBS} ${IPP_LIBS} ${ARGN})
    if (HAVE_CUDA)
      ocv_target_link_libraries(${the_module} LINK_PRIVATE ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
    endif()
  endif()

  add_dependencies(opencv_modules ${the_module})

  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_module} PROPERTIES FOLDER "modules")
  endif()

  set_target_properties(${the_module} PROPERTIES
    OUTPUT_NAME "${the_module}${OPENCV_DLLVERSION}"
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    COMPILE_PDB_NAME "${the_module}${OPENCV_DLLVERSION}"
    COMPILE_PDB_NAME_DEBUG "${the_module}${OPENCV_DLLVERSION}${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    COMPILE_PDB_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
  )

  # For dynamic link numbering convenions
  if(NOT ANDROID)
    # Android SDK build scripts can include only .so files into final .apk
    # As result we should not set version properties for Android
    set_target_properties(${the_module} PROPERTIES
      VERSION ${OPENCV_LIBVERSION}
      SOVERSION ${OPENCV_SOVERSION}
    )
  endif()

  if((NOT DEFINED OPENCV_MODULE_TYPE AND BUILD_SHARED_LIBS)
      OR (DEFINED OPENCV_MODULE_TYPE AND OPENCV_MODULE_TYPE STREQUAL SHARED))
    set_target_properties(${the_module} PROPERTIES COMPILE_DEFINITIONS CVAPI_EXPORTS)
    set_target_properties(${the_module} PROPERTIES DEFINE_SYMBOL CVAPI_EXPORTS)
  endif()

  if(MSVC)
    if(CMAKE_CROSSCOMPILING)
      set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:secchk")
    endif()
    set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libc /DEBUG")
  endif()

  get_target_property(_target_type ${the_module} TYPE)
  if(OPENCV_MODULE_${the_module}_CLASS STREQUAL "PUBLIC" AND
      ("${_target_type}" STREQUAL "SHARED_LIBRARY" OR (NOT BUILD_SHARED_LIBS OR NOT INSTALL_CREATE_DISTRIB)))
    ocv_install_target(${the_module} EXPORT OpenCVModules OPTIONAL
      RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT libs
      LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT libs NAMELINK_SKIP
      ARCHIVE DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT dev
      )
  endif()
  if("${_target_type}" STREQUAL "SHARED_LIBRARY")
    install(TARGETS ${the_module}
      LIBRARY DESTINATION ${OPENCV_LIB_INSTALL_PATH} COMPONENT dev NAMELINK_ONLY)
  endif()

  foreach(m ${OPENCV_MODULE_${the_module}_CHILDREN} ${the_module})
    # only "public" headers need to be installed
    if(OPENCV_MODULE_${m}_HEADERS AND ";${OPENCV_MODULES_PUBLIC};" MATCHES ";${m};")
      foreach(hdr ${OPENCV_MODULE_${m}_HEADERS})
        string(REGEX REPLACE "^.*opencv2/" "opencv2/" hdr2 "${hdr}")
        if(NOT hdr2 MATCHES "opencv2/${m}/private.*" AND hdr2 MATCHES "^(opencv2/?.*)/[^/]+.h(..)?$" )
          install(FILES ${hdr} OPTIONAL DESTINATION "${OPENCV_INCLUDE_INSTALL_PATH}/${CMAKE_MATCH_1}" COMPONENT dev)
        endif()
      endforeach()
    endif()
  endforeach()

  _ocv_add_precompiled_headers(${the_module})

  if (TARGET ${the_module}_object)
    # copy COMPILE_DEFINITIONS
    get_target_property(main_defs ${the_module} COMPILE_DEFINITIONS)
    if (main_defs)
      set_target_properties(${the_module}_object PROPERTIES COMPILE_DEFINITIONS ${main_defs})
    endif()
    # use same PCH
    if (TARGET pch_Generate_${the_module})
      add_dependencies(${the_module}_object pch_Generate_${the_module} )
    endif()
  endif()
endmacro()

# opencv precompiled headers macro (can add pch to modules and tests)
# this macro must be called after any "add_definitions" commands, otherwise precompiled headers will not work
# Usage:
# ocv_add_precompiled_headers(${the_module})
macro(_ocv_add_precompiled_headers the_target)
  ocv_debug_message("ocv_add_precompiled_headers(" ${the_target} ${ARGN} ")")

  if("${the_target}" MATCHES "^opencv_test_.*$")
    SET(pch_path "test/test_")
  elseif("${the_target}" MATCHES "^opencv_perf_.*$")
    SET(pch_path "perf/perf_")
  else()
    SET(pch_path "src/")
  endif()
  ocv_add_precompiled_header_to_target(${the_target} "${CMAKE_CURRENT_SOURCE_DIR}/${pch_path}precomp.hpp")
  unset(pch_path)
endmacro()

# short command for adding simple OpenCV module
# see ocv_add_module for argument details
# Usage:
# ocv_define_module(module_name  [INTERNAL] [EXCLUDE_CUDA] [REQUIRED] [<list of dependencies>] [OPTIONAL <list of optional dependencies>] [WRAP <list of wrappers>])
macro(ocv_define_module module_name)
  ocv_debug_message("ocv_define_module(" ${module_name} ${ARGN} ")")
  set(_argn ${ARGN})
  set(exclude_cuda "")
  foreach(arg ${_argn})
    if("${arg}" STREQUAL "EXCLUDE_CUDA")
      set(exclude_cuda "${arg}")
      list(REMOVE_ITEM _argn ${arg})
    endif()
  endforeach()

  ocv_add_module(${module_name} ${_argn})
  ocv_glob_module_sources(${exclude_cuda})
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

# auxiliary macro to parse arguments of ocv_add_accuracy_tests and ocv_add_perf_tests commands
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
      set(__currentvar "OPENCV_${tests_type}_${the_module}_DEPS")
    elseif(" ${__currentvar}" STREQUAL " __file_group_sources" AND NOT __file_group_name) # spaces to avoid CMP0054
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
function(ocv_add_perf_tests)
  ocv_debug_message("ocv_add_perf_tests(" ${ARGN} ")")

  if(WINRT)
    set(OPENCV_DEBUG_POSTFIX "")
  endif()

  set(perf_path "${CMAKE_CURRENT_LIST_DIR}/perf")
  if(BUILD_PERF_TESTS AND EXISTS "${perf_path}")
    __ocv_parse_test_sources(PERF ${ARGN})

    # opencv_imgcodecs is required for imread/imwrite
    set(perf_deps opencv_ts ${the_module} opencv_imgcodecs ${OPENCV_MODULE_${the_module}_DEPS} ${OPENCV_MODULE_opencv_ts_DEPS})
    ocv_check_dependencies(${perf_deps})

    if(OCV_DEPENDENCIES_FOUND)
      set(the_target "opencv_perf_${name}")
      # project(${the_target})

      if(NOT OPENCV_PERF_${the_module}_SOURCES)
        file(GLOB_RECURSE perf_srcs "${perf_path}/*.cpp")
        file(GLOB_RECURSE perf_hdrs "${perf_path}/*.hpp" "${perf_path}/*.h")
        ocv_source_group("Src" DIRBASE "${perf_path}" FILES ${perf_srcs})
        ocv_source_group("Include" DIRBASE "${perf_path}" FILES ${perf_hdrs})
        set(OPENCV_PERF_${the_module}_SOURCES ${perf_srcs} ${perf_hdrs})
      endif()

      if(NOT BUILD_opencv_world)
        get_native_precompiled_header(${the_target} perf_precomp.hpp)
      endif()

      ocv_add_executable(${the_target} ${OPENCV_PERF_${the_module}_SOURCES} ${${the_target}_pch})
      ocv_target_include_modules(${the_target} ${perf_deps} "${perf_path}")
      ocv_target_link_libraries(${the_target} ${perf_deps} ${OPENCV_MODULE_${the_module}_DEPS} ${OPENCV_LINKER_LIBS})
      add_dependencies(opencv_perf_tests ${the_target})

      set_target_properties(${the_target} PROPERTIES LABELS "${OPENCV_MODULE_${the_module}_LABEL};PerfTest")
      set_source_files_properties(${OPENCV_PERF_${the_module}_SOURCES} ${${the_target}_pch}
        PROPERTIES LABELS "${OPENCV_MODULE_${the_module}_LABEL};PerfTest")

      # Additional target properties
      set_target_properties(${the_target} PROPERTIES
        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
      )
      if(ENABLE_SOLUTION_FOLDERS)
        set_target_properties(${the_target} PROPERTIES FOLDER "tests performance")
      endif()

      if(WINRT)
        # removing APPCONTAINER from tests to run from console
        # look for detailed description inside of ocv_create_module macro above
        add_custom_command(TARGET "opencv_perf_${name}"
                           POST_BUILD
                           COMMAND link.exe /edit /APPCONTAINER:NO $(TargetPath))
      endif()

      if(NOT BUILD_opencv_world)
        _ocv_add_precompiled_headers(${the_target})
      endif()

      ocv_add_test_from_target("${the_target}" "Performance" "${the_target}")
      ocv_add_test_from_target("opencv_sanity_${name}" "Sanity" "${the_target}"
                               "--perf_min_samples=1"
                               "--perf_force_samples=1"
                               "--perf_verify_sanity")
    else(OCV_DEPENDENCIES_FOUND)
      # TODO: warn about unsatisfied dependencies
    endif(OCV_DEPENDENCIES_FOUND)
    if(INSTALL_TESTS)
      install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_TEST_INSTALL_PATH} COMPONENT tests)
    endif()
  endif()
endfunction()

# this is a command for adding OpenCV accuracy/regression tests to the module
# ocv_add_accuracy_tests([FILES <source group name> <list of sources>] [DEPENDS_ON] <list of extra dependencies>)
function(ocv_add_accuracy_tests)
  ocv_debug_message("ocv_add_accuracy_tests(" ${ARGN} ")")

  if(WINRT)
    set(OPENCV_DEBUG_POSTFIX "")
  endif()

  set(test_path "${CMAKE_CURRENT_LIST_DIR}/test")
  if(BUILD_TESTS AND EXISTS "${test_path}")
    __ocv_parse_test_sources(TEST ${ARGN})

    # opencv_imgcodecs is required for imread/imwrite
    set(test_deps opencv_ts ${the_module} opencv_imgcodecs opencv_videoio ${OPENCV_MODULE_${the_module}_DEPS} ${OPENCV_MODULE_opencv_ts_DEPS})
    ocv_check_dependencies(${test_deps})
    if(OCV_DEPENDENCIES_FOUND)
      set(the_target "opencv_test_${name}")
      # project(${the_target})

      if(NOT OPENCV_TEST_${the_module}_SOURCES)
        file(GLOB_RECURSE test_srcs "${test_path}/*.cpp")
        file(GLOB_RECURSE test_hdrs "${test_path}/*.hpp" "${test_path}/*.h")
        ocv_source_group("Src" DIRBASE "${test_path}" FILES ${test_srcs})
        ocv_source_group("Include" DIRBASE "${test_path}" FILES ${test_hdrs})
        set(OPENCV_TEST_${the_module}_SOURCES ${test_srcs} ${test_hdrs})
      endif()

      if(NOT BUILD_opencv_world)
        get_native_precompiled_header(${the_target} test_precomp.hpp)
      endif()

      ocv_add_executable(${the_target} ${OPENCV_TEST_${the_module}_SOURCES} ${${the_target}_pch})
      ocv_target_include_modules(${the_target} ${test_deps} "${test_path}")
      ocv_target_link_libraries(${the_target} ${test_deps} ${OPENCV_MODULE_${the_module}_DEPS} ${OPENCV_LINKER_LIBS})
      add_dependencies(opencv_tests ${the_target})

      set_target_properties(${the_target} PROPERTIES LABELS "${OPENCV_MODULE_${the_module}_LABEL};AccuracyTest")
      set_source_files_properties(${OPENCV_TEST_${the_module}_SOURCES} ${${the_target}_pch}
        PROPERTIES LABELS "${OPENCV_MODULE_${the_module}_LABEL};AccuracyTest")

      # Additional target properties
      set_target_properties(${the_target} PROPERTIES
        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
      )

      if(ENABLE_SOLUTION_FOLDERS)
        set_target_properties(${the_target} PROPERTIES FOLDER "tests accuracy")
      endif()

      if(OPENCV_TEST_BIGDATA)
        ocv_append_target_property(${the_target} COMPILE_DEFINITIONS "OPENCV_TEST_BIGDATA=1")
      endif()

      if(NOT BUILD_opencv_world)
        _ocv_add_precompiled_headers(${the_target})
      endif()

      ocv_add_test_from_target("${the_target}" "Accuracy" "${the_target}")
    else(OCV_DEPENDENCIES_FOUND)
      # TODO: warn about unsatisfied dependencies
    endif(OCV_DEPENDENCIES_FOUND)

    if(INSTALL_TESTS)
      install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_TEST_INSTALL_PATH} COMPONENT tests)
    endif()
  endif()
endfunction()

function(ocv_add_samples)
  ocv_debug_message("ocv_add_samples(" ${ARGN} ")")

  set(samples_path "${CMAKE_CURRENT_SOURCE_DIR}/samples")
  string(REGEX REPLACE "^opencv_" "" module_id ${the_module})

  if(BUILD_EXAMPLES AND EXISTS "${samples_path}")
    set(samples_deps ${the_module} ${OPENCV_MODULE_${the_module}_DEPS} opencv_imgcodecs opencv_videoio opencv_highgui ${ARGN})
    ocv_check_dependencies(${samples_deps})

    if(OCV_DEPENDENCIES_FOUND)
      file(GLOB sample_sources "${samples_path}/*.cpp")

      foreach(source ${sample_sources})
        get_filename_component(name "${source}" NAME_WE)
        set(the_target "example_${module_id}_${name}")

        ocv_add_executable(${the_target} "${source}")
        ocv_target_include_modules(${the_target} ${samples_deps})
        ocv_target_link_libraries(${the_target} ${samples_deps})
        set_target_properties(${the_target} PROPERTIES PROJECT_LABEL "(sample) ${name}")

        set_target_properties(${the_target} PROPERTIES LABELS "${OPENCV_MODULE_${the_module}_LABEL};Sample")
        set_source_files_properties("${source}"
          PROPERTIES LABELS "${OPENCV_MODULE_${the_module}_LABEL};Sample")

        if(ENABLE_SOLUTION_FOLDERS)
          set_target_properties(${the_target} PROPERTIES
            OUTPUT_NAME "${module_id}-example-${name}"
            FOLDER "samples/${module_id}")
        endif()

        if(WIN32)
          install(TARGETS ${the_target} RUNTIME DESTINATION "samples/${module_id}" COMPONENT samples)
        endif()
      endforeach()
    endif()
  endif()

  if(INSTALL_C_EXAMPLES AND NOT WIN32 AND EXISTS "${samples_path}")
  file(GLOB DEPLOY_FILES_AND_DIRS "${samples_path}/*")
    foreach(ITEM ${DEPLOY_FILES_AND_DIRS})
        IF( IS_DIRECTORY "${ITEM}" )
            LIST( APPEND sample_dirs "${ITEM}" )
        ELSE()
            LIST( APPEND sample_files "${ITEM}" )
        ENDIF()
    endforeach()
    install(FILES ${sample_files}
            DESTINATION ${OPENCV_SAMPLES_SRC_INSTALL_PATH}/${module_id}
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ COMPONENT samples)
    install(DIRECTORY ${sample_dirs}
            DESTINATION ${OPENCV_SAMPLES_SRC_INSTALL_PATH}/${module_id}
            USE_SOURCE_PERMISSIONS COMPONENT samples)
  endif()
endfunction()
