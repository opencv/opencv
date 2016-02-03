# - Find socket++
# Find the native socket++ headers and libraries.
# This module defines
#  SOCKETXX_INCLUDE_DIRS - the json include directory
#  SOCKETXX_LIBRARIES    - the libraries needed to use json
#  SOCKETXX_FOUND        - system has the json library
#
#  Copyright (c) 2013 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.

find_path(SOCKETXX_INCLUDE_DIR socket++.h
  /usr/include/socket++
  )
find_library(SOCKETXX_LIBRARY NAMES Papyrus3)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SOCKETXX DEFAULT_MSG
  SOCKETXX_LIBRARY
  SOCKETXX_INCLUDE_DIR
)

if(SOCKETXX_FOUND)
  set(SOCKETXX_LIBRARIES ${SOCKETXX_LIBRARY})
  set(SOCKETXX_INCLUDE_DIRS ${SOCKETXX_INCLUDE_DIR})
endif()

mark_as_advanced(
  SOCKETXX_LIBRARY
  SOCKETXX_INCLUDE_DIR
)
