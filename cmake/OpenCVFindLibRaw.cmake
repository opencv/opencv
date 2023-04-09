# - Find LibRaw
# Find the LibRaw library <http://www.libraw.org>
# This module defines
#  LibRaw_VERSION_STRING, the version string of LibRaw
#  LibRaw_INCLUDE_DIR, where to find libraw.h
#  LibRaw_LIBRARIES, the libraries needed to use LibRaw (non-thread-safe)
#  LibRaw_r_LIBRARIES, the libraries needed to use LibRaw (thread-safe)
#  LibRaw_DEFINITIONS, the definitions needed to use LibRaw (non-thread-safe)
#  LibRaw_r_DEFINITIONS, the definitions needed to use LibRaw (thread-safe)
#
# Copyright (c) 2013, Pino Toscano <pino at kde dot org>
# Copyright (c) 2013, Gilles Caulier <caulier dot gilles at gmail dot com>
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying LICENSE file.

FIND_PACKAGE(PkgConfig)

IF(PKG_CONFIG_FOUND)
   PKG_CHECK_MODULES(PC_LIBRAW libraw)
   SET(LibRaw_DEFINITIONS ${PC_LIBRAW_CFLAGS_OTHER})

   PKG_CHECK_MODULES(PC_LIBRAW_R libraw_r)
   SET(LibRaw_r_DEFINITIONS ${PC_LIBRAW_R_CFLAGS_OTHER})
ENDIF()

FIND_PATH(LibRaw_INCLUDE_DIR libraw.h
          HINTS
          ${PC_LIBRAW_INCLUDEDIR}
          ${PC_LibRaw_INCLUDE_DIRS}
          PATH_SUFFIXES libraw
         )

FIND_LIBRARY(LibRaw_LIBRARY_RELEASE NAMES raw
             HINTS
             ${PC_LIBRAW_LIBDIR}
             ${PC_LIBRAW_LIBRARY_DIRS}
            )

FIND_LIBRARY(LibRaw_LIBRARY_DEBUG NAMES rawd
             HINTS
             ${PC_LIBRAW_LIBDIR}
             ${PC_LIBRAW_LIBRARY_DIRS}
            )

include(SelectLibraryConfigurations)
select_library_configurations(LibRaw)

FIND_LIBRARY(LibRaw_r_LIBRARY_RELEASE NAMES raw_r
             HINTS
             ${PC_LIBRAW_R_LIBDIR}
             ${PC_LIBRAW_R_LIBRARY_DIRS}
            )

FIND_LIBRARY(LibRaw_r_LIBRARY_DEBUG NAMES raw_rd
             HINTS
             ${PC_LIBRAW_R_LIBDIR}
             ${PC_LIBRAW_R_LIBRARY_DIRS}
            )

select_library_configurations(LibRaw_r)

IF(LibRaw_INCLUDE_DIR)
   FILE(READ ${LibRaw_INCLUDE_DIR}/libraw_version.h _libraw_version_content)

   STRING(REGEX MATCH "#define LIBRAW_MAJOR_VERSION[ \t]*([0-9]*)\n" _version_major_match ${_libraw_version_content})
   SET(_libraw_version_major "${CMAKE_MATCH_1}")

   STRING(REGEX MATCH "#define LIBRAW_MINOR_VERSION[ \t]*([0-9]*)\n" _version_minor_match ${_libraw_version_content})
   SET(_libraw_version_minor "${CMAKE_MATCH_1}")

   STRING(REGEX MATCH "#define LIBRAW_PATCH_VERSION[ \t]*([0-9]*)\n" _version_patch_match ${_libraw_version_content})
   SET(_libraw_version_patch "${CMAKE_MATCH_1}")

   IF(_version_major_match AND _version_minor_match AND _version_patch_match)
      SET(LibRaw_VERSION_STRING "${_libraw_version_major}.${_libraw_version_minor}.${_libraw_version_patch}")
   ELSE()
      IF(NOT LibRaw_FIND_QUIETLY)
         MESSAGE(STATUS "Failed to get version information from ${LibRaw_INCLUDE_DIR}/libraw_version.h")
      ENDIF()
   ENDIF()
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibRaw
                                  REQUIRED_VARS LibRaw_LIBRARIES LibRaw_INCLUDE_DIR
                                  VERSION_VAR LibRaw_VERSION_STRING
                                 )
if (LibRaw_FOUND)
	set(HAVE_RAW ON)
ENDIF()
MARK_AS_ADVANCED(LibRaw_VERSION_STRING
                 LibRaw_INCLUDE_DIR
                 LibRaw_LIBRARIES
                 LibRaw_r_LIBRARIES
                 LibRaw_DEFINITIONS
                 LibRaw_r_DEFINITIONS
                 )
