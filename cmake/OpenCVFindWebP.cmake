#=============================================================================
# Find WebP library
#=============================================================================
# Find the native WebP headers and libraries.
#
#  WEBP_INCLUDE_DIRS - where to find webp/decode.h, etc.
#  WEBP_LIBRARIES    - List of libraries when using webp.
#  WEBP_FOUND        - True if webp is found.
#=============================================================================

# Look for the header file.

unset(WEBP_FOUND)

FIND_PATH(WEBP_INCLUDE_DIR NAMES webp/decode.h)

if(NOT WEBP_INCLUDE_DIR)
    unset(WEBP_FOUND)
else()
    MARK_AS_ADVANCED(WEBP_INCLUDE_DIR)

    # Look for the library.
    FIND_LIBRARY(WEBP_LIBRARY NAMES webp)
    MARK_AS_ADVANCED(WEBP_LIBRARY)

    # handle the QUIETLY and REQUIRED arguments and set WEBFOUND_FOUND to TRUE if
    # all listed variables are TRUE
    INCLUDE(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(WebP DEFAULT_MSG WEBP_LIBRARY WEBP_INCLUDE_DIR)

    SET(WEBP_LIBRARIES ${WEBP_LIBRARY})
    SET(WEBP_INCLUDE_DIRS ${WEBP_INCLUDE_DIR})
endif()
