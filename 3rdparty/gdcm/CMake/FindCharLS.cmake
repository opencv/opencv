#
#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

find_path(CHARLS_INCLUDE_DIR CharLS/interface.h
/usr/local/include
/usr/include
)

find_library(CHARLS_LIBRARY
  NAMES CharLS
  PATHS /usr/lib /usr/local/lib
  )

if (CHARLS_LIBRARY AND CHARLS_INCLUDE_DIR)
    set(CHARLS_LIBRARIES    ${CHARLS_LIBRARY})
    set(CHARLS_INCLUDE_DIRS ${CHARLS_INCLUDE_DIR})
    set(CHARLS_FOUND "YES")
else ()
  set(CHARLS_FOUND "NO")
endif ()

if (CHARLS_FOUND)
   if (NOT CHARLS_FIND_QUIETLY)
      message(STATUS "Found CHARLS: ${CHARLS_LIBRARIES}")
   endif ()
else ()
   if (CHARLS_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find CHARLS library")
   endif ()
endif ()

mark_as_advanced(
  CHARLS_LIBRARY
  CHARLS_INCLUDE_DIR
  )
