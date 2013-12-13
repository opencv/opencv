# Main variables:
# INTELPERC_LIBRARY and INTELPERC_INCLUDES to link Intel Perceptial Computing SDK modules
# HAVE_INTELPERC for conditional compilation OpenCV with/without Intel Perceptial Computing SDK

if(NOT "${INTELPERC_LIB_DIR}" STREQUAL "${INTELPERC_LIB_DIR_INTERNAL}")
    unset(INTELPERC_LIBRARY CACHE)
    unset(INTELPERC_LIB_DIR CACHE)
endif()

if(NOT "${INTELPERC_INCLUDE_DIR}" STREQUAL "${INTELPERC_INCLUDE_DIR_INTERNAL}")
    unset(INTELPERC_INCLUDES CACHE)
    unset(INTELPERC_INCLUDE_DIR CACHE)
endif()

if(WIN32)
    if(NOT (MSVC64 OR MINGW64))
        find_file(INTELPERC_INCLUDES "pxcsession.h" PATHS "$ENV{PCSDK_DIR}include" DOC "Intel Perceptual Computing SDK interface header")
        find_library(INTELPERC_LIBRARY "libpxc.lib" PATHS "$ENV{PCSDK_DIR}lib/Win32" DOC "Intel Perceptual Computing SDK library")
    else()
        find_file(INTELPERC_INCLUDES "pxcsession.h" PATHS "$ENV{PCSDK_DIR}include" DOC "Intel Perceptual Computing SDK interface header")
        find_library(INTELPERC_LIBRARY "libpxc.lib" PATHS "$ENV{PCSDK_DIR}/lib/x64" DOC "Intel Perceptual Computing SDK library")
    endif()
endif()

if(INTELPERC_LIBRARY AND INTELPERC_INCLUDES)
    set(HAVE_INTELPERC TRUE)
endif() #if(INTELPERC_LIBRARY AND INTELPERC_INCLUDES)

get_filename_component(INTELPERC_LIB_DIR "${INTELPERC_LIBRARY}" PATH)
get_filename_component(INTELPERC_INCLUDE_DIR "${INTELPERC_INCLUDES}" PATH)

if(HAVE_INTELPERC)
  set(INTELPERC_LIB_DIR "${INTELPERC_LIB_DIR}" CACHE PATH "Path to Intel Perceptual Computing SDK interface libraries" FORCE)
  set(INTELPERC_INCLUDE_DIR "${INTELPERC_INCLUDE_DIR}" CACHE PATH "Path to Intel Perceptual Computing SDK interface headers" FORCE)
endif()

if(INTELPERC_LIBRARY)
    set(INTELPERC_LIB_DIR_INTERNAL "${INTELPERC_LIB_DIR}" CACHE INTERNAL "This is the value of the last time INTELPERC_LIB_DIR was set successfully." FORCE)
else()
    message( WARNING, " Intel Perceptual Computing SDK library directory (set by INTELPERC_LIB_DIR variable) is not found or does not have Intel Perceptual Computing SDK libraries." )
endif()

if(INTELPERC_INCLUDES)
    set(INTELPERC_INCLUDE_DIR_INTERNAL "${INTELPERC_INCLUDE_DIR}" CACHE INTERNAL "This is the value of the last time INTELPERC_INCLUDE_DIR was set successfully." FORCE)
else()
    message( WARNING, " Intel Perceptual Computing SDK include directory (set by INTELPERC_INCLUDE_DIR variable) is not found or does not have Intel Perceptual Computing SDK include files." )
endif()

mark_as_advanced(FORCE INTELPERC_LIBRARY)
mark_as_advanced(FORCE INTELPERC_INCLUDES)

