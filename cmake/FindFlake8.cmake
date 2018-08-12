# - Find Flake8
# Find the Flake8 executable and extract the version number
#
# OUTPUT Variables
#
#   FLAKE8_FOUND
#       True if the flake8 package was found
#   FLAKE8_EXECUTABLE
#       The flake8 executable location
#   FLAKE8_VERSION
#       A string denoting the version of flake8 that has been found

find_host_program(FLAKE8_EXECUTABLE flake8 PATHS /usr/bin)

if(FLAKE8_EXECUTABLE)
  execute_process(COMMAND ${FLAKE8_EXECUTABLE} --version OUTPUT_VARIABLE FLAKE8_VERSION_RAW ERROR_QUIET)
  if(FLAKE8_VERSION_RAW MATCHES "^([0-9\\.]+[0-9])")
    set(FLAKE8_VERSION "${CMAKE_MATCH_1}")
  else()
    set(FLAKE8_VERSION "unknown")
  endif()
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Flake8 DEFAULT_MSG FLAKE8_EXECUTABLE)

mark_as_advanced(FLAKE8_EXECUTABLE FLAKE8_VERSION)
