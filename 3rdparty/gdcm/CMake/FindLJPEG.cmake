#
#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

find_path(LJPEG_INCLUDE_DIR ljpeg-62/jpeglib.h
/usr/local/include
/usr/include
)

find_library(LJPEG8_LIBRARY
  NAMES jpeg8
  PATHS /usr/lib /usr/local/lib
  )
find_library(LJPEG12_LIBRARY
  NAMES jpeg12
  PATHS /usr/lib /usr/local/lib
  )
find_library(LJPEG16_LIBRARY
  NAMES jpeg16
  PATHS /usr/lib /usr/local/lib
  )

if (LJPEG8_LIBRARY AND LJPEG_INCLUDE_DIR)
    set(LJPEG_LIBRARIES ${LJPEG8_LIBRARY} ${LJPEG12_LIBRARY} ${LJPEG16_LIBRARY})
    set(LJPEG_INCLUDE_DIRS ${LJPEG_INCLUDE_DIR})
    set(LJPEG_FOUND "YES")
else ()
  set(LJPEG_FOUND "NO")
endif ()


if (LJPEG_FOUND)
   if (NOT LJPEG_FIND_QUIETLY)
      message(STATUS "Found LJPEG: ${LJPEG_LIBRARIES}")
   endif ()
else ()
   if (LJPEG_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find LJPEG library")
   endif ()
endif ()

mark_as_advanced(
  LJPEG_LIBRARIES
  LJPEG_INCLUDE_DIR
  )
