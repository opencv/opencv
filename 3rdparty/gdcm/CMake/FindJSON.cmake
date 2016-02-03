# - Find json
# Find the native JSON headers and libraries.
# This module defines
#  JSON_INCLUDE_DIRS - the json include directory
#  JSON_LIBRARIES    - the libraries needed to use json
#  JSON_FOUND        - system has the json library
#
#  Copyright (c) 2013 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.

# See:
# https://github.com/json-c/json-c/wiki
# $ sudo apt-get install libjson0-dev
# in sid:
# $ sudo apt-get install libjson-c-dev

find_path(JSON_INCLUDE_DIR NAMES json-c/json.h json/json.h)
find_library(JSON_LIBRARY NAMES json-c json)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JSON DEFAULT_MSG
 JSON_LIBRARY
 JSON_INCLUDE_DIR
)

if(JSON_FOUND)
  set(JSON_LIBRARIES ${JSON_LIBRARY})
  # hack to get old and new layout working:
  set(JSON_INCLUDE_DIRS ${JSON_INCLUDE_DIR}/json-c
    ${JSON_INCLUDE_DIR}/json)
endif()

mark_as_advanced(
  JSON_LIBRARY
  JSON_INCLUDE_DIR
)
