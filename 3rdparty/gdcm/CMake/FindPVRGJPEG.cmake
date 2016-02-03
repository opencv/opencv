#
# this module looks for PVRG-JPEG
#
# PVRG_JPEG_EXECUTABLE - the full path to pvrg-jpeg
# PVRG_JPEG_FOUND      - If false, don't attempt to use pvrg-jpeg
#
#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

find_program(PVRGJPEG_EXECUTABLE
  pvrg-jpeg
  )


mark_as_advanced(
  PVRGJPEG_EXECUTABLE
  )


if (PVRGJPEG_EXECUTABLE)
    set(PVRGJPEG_FOUND "YES")
else ()
  set(PVRGJPEG_FOUND "NO")
endif ()

if (PVRGJPEG_FOUND)
   if (NOT PVRGJPEG_FIND_QUIETLY)
      message(STATUS "Found PVRGJPEG: ${PVRGJPEG_EXECUTABLE}")
   endif ()
else ()
   if (PVRGJPEG_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find PVRGJPEG exe")
   endif ()
endif ()
