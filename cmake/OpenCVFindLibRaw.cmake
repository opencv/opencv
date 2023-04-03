#=============================================================================
# Find Raw library
#=============================================================================
# Find the native RAW headers and libraries.
#
#  RAW_INCLUDE_DIRS - where to find libraw/libraw.h, etc.
#  RAW_LIBRARIES    - List of libraries when using raw.
#  RAW_FOUND        - True if raw is found.
#=============================================================================

# Look for the header file.

unset(RAW_FOUND)

FIND_PATH(RAW_INCLUDE_DIR NAMES libraw/libraw.h)
message("RAW Include ${RAW_INCLUDE_DIR}")

if(NOT LIBRAW_INCLUDE_DIR)
    unset(RAW_FOUND)
else()
    MARK_AS_ADVANCED(RAW_INCLUDE_DIR)

    # Look for the library.
    FIND_LIBRARY(RAW_LIBRARY NAMES libraw)
    MARK_AS_ADVANCED(RAW_LIBRARY)

    # handle the QUIETLY and REQUIRED arguments and set LIBRAW_FOUND to TRUE if
    # all listed variables are TRUE
    INCLUDE(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibRaw DEFAULT_MSG RAW_LIBRARY RAW_INCLUDE_DIR)

    SET(RAW_LIBRARIES ${RAW_LIBRARY})
    SET(RAW_INCLUDE_DIRS ${RAW_INCLUDE_DIR})
endif()
