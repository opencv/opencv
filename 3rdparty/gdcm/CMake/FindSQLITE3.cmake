#
#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

find_path(SQLITE3_INCLUDE_DIR sqlite3.h)

find_library(SQLITE3_LIBRARY NAMES sqlite3)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SQLITE3 DEFAULT_MSG SQLITE3_LIBRARY SQLITE3_INCLUDE_DIR)

set(SQLITE3_LIBRARIES ${SQLITE3_LIBRARY})
set(SQLITE3_INCLUDE_DIRS ${SQLITE3_INCLUDE_DIR})

mark_as_advanced(SQLITE3_LIBRARY SQLITE3_INCLUDE_DIR )
