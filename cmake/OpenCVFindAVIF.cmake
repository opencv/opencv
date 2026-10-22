#=============================================================================
# Find AVIF library
#=============================================================================
# Find the native AVIF headers and libraries.
#
#  AVIF_INCLUDE_DIRS - where to find avif/avif.h, etc.
#  AVIF_LIBRARIES    - List of libraries when using AVIF.
#  AVIF_FOUND        - True if AVIF is found.
#  AVIF_VERSION      - libavif version string (e.g. "1.4.0")
#=============================================================================

# Look for the header file.

unset(AVIF_FOUND)

find_package(libavif QUIET)

if(TARGET avif)
    MARK_AS_ADVANCED(AVIF_INCLUDE_DIR)
    MARK_AS_ADVANCED(AVIF_LIBRARY)

    SET(AVIF_FOUND TRUE)
    SET(AVIF_LIBRARY avif)
    GET_TARGET_PROPERTY(AVIF_INCLUDE_DIR1 avif INCLUDE_DIRECTORIES)
    GET_TARGET_PROPERTY(AVIF_INCLUDE_DIR2 avif INTERFACE_INCLUDE_DIRECTORIES)
    set(AVIF_INCLUDE_DIR)
    if(AVIF_INCLUDE_DIR1)
        LIST(APPEND AVIF_INCLUDE_DIR ${AVIF_INCLUDE_DIR1})
    endif()
    if(AVIF_INCLUDE_DIR2)
        LIST(APPEND AVIF_INCLUDE_DIR ${AVIF_INCLUDE_DIR2})
    endif()
else()
    FIND_PATH(AVIF_INCLUDE_DIR NAMES avif/avif.h)

     # Look for the library.
    FIND_LIBRARY(AVIF_LIBRARY NAMES avif)
    MARK_AS_ADVANCED(AVIF_LIBRARY)

    # handle the QUIETLY and REQUIRED arguments and set AVIF_FOUND to TRUE if
    # all listed variables are TRUE
    INCLUDE(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(AVIF DEFAULT_MSG AVIF_LIBRARY AVIF_INCLUDE_DIR)
endif()

if(NOT AVIF_FOUND)
  UNSET(AVIF_LIBRARIES)
  UNSET(AVIF_INCLUDE_DIRS)
  UNSET(AVIF_VERSION)
  return()
endif()

SET(AVIF_LIBRARIES ${AVIF_LIBRARY})
SET(AVIF_INCLUDE_DIRS ${AVIF_INCLUDE_DIR})

# unset(libavif_VERSION) # For debugging: uncomment to test fallback version extraction when find_package(libavif) cannot find libavif

if(NOT libavif_VERSION)
  find_file(AVIF_H NAMES avif/avif.h PATHS ${AVIF_INCLUDE_DIRS} NO_DEFAULT_PATH)

  if(AVIF_H)
    file(STRINGS "${AVIF_H}" AVIF_MAJOR_LINE REGEX "^#define[ \t]+AVIF_VERSION_MAJOR[ \t]+[0-9]+")
    file(STRINGS "${AVIF_H}" AVIF_MINOR_LINE REGEX "^#define[ \t]+AVIF_VERSION_MINOR[ \t]+[0-9]+")
    file(STRINGS "${AVIF_H}" AVIF_PATCH_LINE REGEX "^#define[ \t]+AVIF_VERSION_PATCH[ \t]+[0-9]+")

    string(REGEX REPLACE ".*AVIF_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" AVIF_MAJOR "${AVIF_MAJOR_LINE}")
    string(REGEX REPLACE ".*AVIF_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" AVIF_MINOR "${AVIF_MINOR_LINE}")
    string(REGEX REPLACE ".*AVIF_VERSION_PATCH[ \t]+([0-9]+).*" "\\1" AVIF_PATCH "${AVIF_PATCH_LINE}")

    if(AVIF_MAJOR MATCHES "^[0-9]+$" AND AVIF_MINOR MATCHES "^[0-9]+$" AND AVIF_PATCH MATCHES "^[0-9]+$")
      SET(libavif_VERSION "${AVIF_MAJOR}.${AVIF_MINOR}.${AVIF_PATCH}")
    endif()
  endif()

  UNSET(AVIF_H CACHE)
  UNSET(AVIF_H)
endif()

if(libavif_VERSION)
  SET(AVIF_VERSION "${libavif_VERSION}")
endif()
