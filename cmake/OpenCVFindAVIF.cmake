# The script is taken from https://webkit.googlesource.com/WebKit/+/master/Source/cmake/FindAVIF.cmake

# Copyright (C) 2021 Igalia S.L.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY APPLE INC. AND ITS CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL APPLE INC. OR ITS CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

#[=======================================================================[.rst:
FindAVIF
--------

Find libavif headers and libraries.

Imported Targets
^^^^^^^^^^^^^^^^

``AVIF::AVIF``
  The AVIF library, if found.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables in your project:

``AVIF_FOUND``
  true if (the requested version of) AVIF is available.
``AVIF_VERSION``
  the version of AVIF.
``AVIF_LIBRARIES``
  the libraries to link against to use AVIF.
``AVIF_INCLUDE_DIRS``
  where to find the AVIF headers.
``AVIF_COMPILE_OPTIONS``
  this should be passed to target_compile_options(), if the
  target is not used for linking

#]=======================================================================]

find_package(PkgConfig QUIET)
pkg_check_modules(PC_AVIF QUIET libavif)
set(AVIF_COMPILE_OPTIONS ${PC_AVIF_CFLAGS_OTHER})
set(AVIF_VERSION ${PC_AVIF_VERSION})

find_path(AVIF_INCLUDE_DIR
    NAMES avif.h
    HINTS ${PC_AVIF_INCLUDEDIR}
          ${PC_AVIF_INCLUDE_DIRS}
    PATH_SUFFIXES avif
)

find_library(AVIF_LIBRARY
    NAMES ${AVIF_NAMES} avif
    HINTS ${PC_AVIF_LIBDIR}
          ${PC_AVIF_LIBRARY_DIRS}
)

if (AVIF_INCLUDE_DIR AND NOT AVIF_VERSION)
    if (EXISTS "${AVIF_INCLUDE_DIR}/avif.h")
        file(READ "${AVIF_INCLUDE_DIR}/avif.h" AVIF_VERSION_CONTENT)

        string(REGEX MATCH "#define +AVIF_VERSION_MAJOR +([0-9]+)" _dummy "${AVIF_VERSION_CONTENT}")
        set(AVIF_VERSION_MAJOR "${CMAKE_MATCH_1}")

        string(REGEX MATCH "#define +AVIF_VERSION_MINOR +([0-9]+)" _dummy "${AVIF_VERSION_CONTENT}")
        set(AVIF_VERSION_MINOR "${CMAKE_MATCH_1}")

        string(REGEX MATCH "#define +AVIF_VERSION_PATCH +([0-9]+)" _dummy "${AVIF_VERSION_CONTENT}")
        set(AVIF_VERSION_PATCH "${CMAKE_MATCH_1}")

        set(AVIF_VERSION "${AVIF_VERSION_MAJOR}.${AVIF_VERSION_MINOR}.${AVIF_VERSION_PATCH}")
    endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AVIF
    FOUND_VAR AVIF_FOUND
    REQUIRED_VARS AVIF_INCLUDE_DIR AVIF_LIBRARY
    VERSION_VAR AVIF_VERSION
)

if (AVIF_LIBRARY AND NOT TARGET AVIF::AVIF)
    add_library(AVIF::AVIF UNKNOWN IMPORTED GLOBAL)
    set_target_properties(AVIF::AVIF PROPERTIES
        IMPORTED_LOCATION "${AVIF_LIBRARY}"
        INTERFACE_COMPILE_OPTIONS "${AVIF_COMPILE_OPTIONS}"
        INTERFACE_INCLUDE_DIRECTORIES "${AVIF_INCLUDE_DIR}"
    )
endif ()

mark_as_advanced(AVIF_INCLUDE_DIR AVIF_LIBRARY)

if (AVIF_FOUND)
    set(AVIF_LIBRARIES ${AVIF_LIBRARY})
    set(AVIF_INCLUDE_DIRS ${AVIF_INCLUDE_DIR})
endif ()
