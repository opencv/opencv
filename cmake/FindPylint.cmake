# - Find Pylint
# Find the Pylint executable and extract the version number
#
# OUTPUT Variables
#
#   PYLINT_FOUND
#       True if the pylint package was found
#   PYLINT_EXECUTABLE
#       The pylint executable location
#   PYLINT_VERSION
#       A string denoting the version of pylint that has been found

find_host_program(PYLINT_EXECUTABLE pylint PATHS /usr/bin)

if(PYLINT_EXECUTABLE AND NOT DEFINED PYLINT_VERSION)
  execute_process(COMMAND ${PYLINT_EXECUTABLE} --version RESULT_VARIABLE _result OUTPUT_VARIABLE PYLINT_VERSION_RAW)
  if(NOT _result EQUAL 0)
    ocv_clear_vars(PYLINT_EXECUTABLE PYLINT_VERSION)
  elseif(PYLINT_VERSION_RAW MATCHES "pylint([^,\n]*) ([0-9\\.]+[0-9])")
    set(PYLINT_VERSION "${CMAKE_MATCH_2}")
  else()
    set(PYLINT_VERSION "unknown")
  endif()
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Pylint
    REQUIRED_VARS PYLINT_EXECUTABLE
    VERSION_VAR PYLINT_VERSION
)

mark_as_advanced(PYLINT_EXECUTABLE PYLINT_VERSION)
