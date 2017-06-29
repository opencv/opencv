# ----------------------------------------------------------------------------
#  CMake file to run pylint on the tutorial python files.
#
# ----------------------------------------------------------------------------
include(${CMAKE_SOURCE_DIR}/cmake/FindPylint.cmake)

project(run_pylint_on_tutorials)

if(PYLINT_FOUND)
  message(STATUS "pylint version: ${PYLINT_VERSION}")
  set(curdir "${CMAKE_CURRENT_SOURCE_DIR}")
  file(GLOB_RECURSE PYTHON_SCRIPTS ${curdir}/*.py)
  add_custom_target("${PROJECT_NAME}")

  foreach(SCRIPT ${PYTHON_SCRIPTS})
    get_filename_component(SCRIPT_NAME ${SCRIPT} NAME_WE)
    add_custom_command(TARGET "${PROJECT_NAME}"
                       COMMAND ${PYLINT_EXECUTABLE} ${SCRIPT}
                            --rcfile=${curdir}/pylintrc
                       COMMENT "Running pylint on : ${SCRIPT_NAME}.py"
                      )
  endforeach()
endif()
