# - Find papyrus3
# Find the native PAPYRUS3 headers and libraries.
# This module defines
#  PAPYRUS3_INCLUDE_DIRS - the json include directory
#  PAPYRUS3_LIBRARIES    - the libraries needed to use json
#  PAPYRUS3_FOUND        - system has the json library
#
#  Copyright (c) 2013 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.

set(PAPYRUS3_INCLUDE_DIRECTORIES
  /usr/include/
  /usr/include/Papyrus3
  )
find_path(PAPYRUS3_INCLUDE_DIR Papyrus3.h
  ${PAPYRUS3_INCLUDE_DIRECTORIES}
  )
find_library(PAPYRUS3_LIBRARY NAMES Papyrus3)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PAPYRUS3 DEFAULT_MSG
 PAPYRUS3_LIBRARY
 PAPYRUS3_INCLUDE_DIR
)

if(PAPYRUS3_FOUND)
  set(PAPYRUS3_LIBRARIES ${PAPYRUS3_LIBRARY})
  set(PAPYRUS3_INCLUDE_DIRS ${PAPYRUS3_INCLUDE_DIR})
endif()

mark_as_advanced(
  PAPYRUS3_LIBRARY
  PAPYRUS3_INCLUDE_DIR
)
