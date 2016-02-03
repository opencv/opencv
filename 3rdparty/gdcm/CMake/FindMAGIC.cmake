#
#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

# $ sudo apt-get install libmagic-dev
# $ dpkg -L libmagic-dev
# ...
# /usr/include/magic.h
# /usr/lib/libmagic.so


find_path(MAGIC_INCLUDE_DIR magic.h
/usr/local/include
/usr/include
)

find_library(MAGIC_LIBRARY
  NAMES magic
  PATHS /usr/lib /usr/local/lib
  )

if (MAGIC_LIBRARY AND MAGIC_INCLUDE_DIR)
    set(MAGIC_LIBRARIES ${MAGIC_LIBRARY})
    set(MAGIC_INCLUDE_DIRS ${MAGIC_INCLUDE_DIR})
    set(MAGIC_FOUND "YES")
else ()
  set(MAGIC_FOUND "NO")
endif ()


if (MAGIC_FOUND)
   if (NOT MAGIC_FIND_QUIETLY)
      message(STATUS "Found MAGIC: ${MAGIC_LIBRARIES} ${MAGIC_INCLUDE_DIR}")
   endif ()
else ()
   if (MAGIC_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find MAGIC library")
   endif ()
endif ()

mark_as_advanced(
  MAGIC_LIBRARY
  MAGIC_INCLUDE_DIR
  )
