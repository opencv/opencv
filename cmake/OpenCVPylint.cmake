if(COMMAND ocv_pylint_add_target)
  return()
endif()

find_package(Pylint QUIET)
if(NOT PYLINT_FOUND OR NOT PYLINT_EXECUTABLE)
  include("${CMAKE_CURRENT_LIST_DIR}/FindPylint.cmake")
endif()

if(NOT PYLINT_FOUND)
  macro(ocv_pylint_add_target) # dummy
  endmacro()
  return()
endif()

macro(ocv_pylint_cleanup)
  foreach(__id ${PYLINT_TARGET_ID})
    ocv_clear_vars(
        PYLINT_TARGET_${__id}_CWD
        PYLINT_TARGET_${__id}_TARGET
        PYLINT_TARGET_${__id}_RCFILE
        PYLINT_TARGET_${__id}_OPTIONS
    )
  endforeach()
  ocv_clear_vars(PYLINT_TARGET_ID)
endmacro()
ocv_pylint_cleanup()

macro(ocv_pylint_add_target)
  cmake_parse_arguments(__pylint "" "CWD;TARGET;RCFILE;" "OPTIONS" ${ARGN})
  if(__pylint_UNPARSED_ARGUMENTS)
    message(WARNING "Unsupported arguments: ${__pylint_UNPARSED_ARGUMENTS}
(keep versions of opencv/opencv_contrib synchronized)
")
  endif()
  ocv_assert(__pylint_TARGET)
  set(__cwd ${__pylint_CWD})
  if(__cwd STREQUAL "default")
    get_filename_component(__cwd "${__pylint_TARGET}" DIRECTORY)
  endif()
  set(__rcfile ${__pylint_RCFILE})
  if(NOT __rcfile AND NOT __pylint_OPTIONS)
    if(__cwd)
      set(__path "${__cwd}")
    else()
      get_filename_component(__path "${__pylint_TARGET}" DIRECTORY)
    endif()
    while(__path MATCHES "^${CMAKE_SOURCE_DIR}")
      if(EXISTS "${__path}/pylintrc")
        set(__rcfile "${__path}/pylintrc")
        break()
      endif()
      if(EXISTS "${__path}/.pylintrc")
        set(__rcfile "${__path}/.pylintrc")
        break()
      endif()
      get_filename_component(__path "${__path}" DIRECTORY)
    endwhile()
    if(NOT __rcfile)
      set(__rcfile "${CMAKE_BINARY_DIR}/pylintrc")
    endif()
  endif()

  list(LENGTH PYLINT_TARGET_ID __id)
  list(APPEND PYLINT_TARGET_ID ${__id})
  set(PYLINT_TARGET_ID "${PYLINT_TARGET_ID}" CACHE INTERNAL "")
  set(PYLINT_TARGET_${__id}_CWD "${__cwd}" CACHE INTERNAL "")
  set(PYLINT_TARGET_${__id}_TARGET "${__pylint_TARGET}" CACHE INTERNAL "")
  set(PYLINT_TARGET_${__id}_RCFILE "${__rcfile}" CACHE INTERNAL "")
  set(PYLINT_TARGET_${__id}_OPTIONS "${__pylint_options}" CACHE INTERNAL "")
endmacro()

macro(ocv_pylint_add_directory_recurse __path)
  file(GLOB_RECURSE __python_scripts ${__path}/*.py)
  list(LENGTH __python_scripts __total)
  if(__total EQUAL 0)
    message(WARNING "Pylint: Python files are not found: ${__path}")
  endif()
  foreach(__script ${__python_scripts})
    ocv_pylint_add_target(TARGET ${__script} ${ARGN})
  endforeach()
endmacro()

macro(ocv_pylint_add_directory __path)
  file(GLOB __python_scripts ${__path}/*.py)
  list(LENGTH __python_scripts __total)
  if(__total EQUAL 0)
    message(WARNING "Pylint: Python files are not found: ${__path}")
  endif()
  foreach(__script ${__python_scripts})
    ocv_pylint_add_target(TARGET ${__script} ${ARGN})
  endforeach()
endmacro()

function(ocv_pylint_finalize)
  if(NOT PYLINT_FOUND)
    return()
  endif()

  add_custom_command(
      OUTPUT "${CMAKE_BINARY_DIR}/pylintrc"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/platforms/scripts/pylintrc" "${CMAKE_BINARY_DIR}/pylintrc"
      DEPENDS "${CMAKE_SOURCE_DIR}/platforms/scripts/pylintrc"
  )

  set(PYLINT_CONFIG_SCRIPT "")
  ocv_cmake_script_append_var(PYLINT_CONFIG_SCRIPT
      PYLINT_EXECUTABLE
      PYLINT_TARGET_ID
  )
  set(__sources "")
  foreach(__id ${PYLINT_TARGET_ID})
    ocv_cmake_script_append_var(PYLINT_CONFIG_SCRIPT
        PYLINT_TARGET_${__id}_CWD
        PYLINT_TARGET_${__id}_TARGET
        PYLINT_TARGET_${__id}_RCFILE
        PYLINT_TARGET_${__id}_OPTIONS
    )
    list(APPEND __sources ${PYLINT_TARGET_${__id}_TARGET} ${PYLINT_TARGET_${__id}_RCFILE})
  endforeach()
  list(REMOVE_DUPLICATES __sources)

  list(LENGTH PYLINT_TARGET_ID __total)
  set(PYLINT_TOTAL_TARGETS "${__total}" CACHE INTERNAL "")
  message(STATUS "Pylint: registered ${__total} targets. Build 'check_pylint' target to run checks (\"cmake --build . --target check_pylint\" or \"make check_pylint\")")
  configure_file("${OpenCV_SOURCE_DIR}/cmake/templates/pylint.cmake.in" "${CMAKE_BINARY_DIR}/pylint.cmake" @ONLY)

  add_custom_target(check_pylint
      COMMAND ${CMAKE_COMMAND} -P "${CMAKE_BINARY_DIR}/pylint.cmake"
      COMMENT "Running pylint"
      DEPENDS ${__sources}
      SOURCES ${__sources}
  )
endfunction()
