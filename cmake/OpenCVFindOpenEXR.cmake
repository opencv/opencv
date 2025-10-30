# The script is taken from http://code.google.com/p/nvidia-texture-tools/

#
# Try to find OpenEXR's libraries, and include path.
# Once done this will define:
#
# OPENEXR_FOUND = OpenEXR found.
# OPENEXR_INCLUDE_PATHS = OpenEXR include directories.
# OPENEXR_LIBRARIES = libraries that are needed to use OpenEXR.
#

if(NOT OPENCV_SKIP_OPENEXR_FIND_PACKAGE)
  find_package(OpenEXR 3 QUIET)
  #ocv_cmake_dump_vars(EXR)
  if(OpenEXR_FOUND)
    if(TARGET OpenEXR::OpenEXR)  # OpenEXR 3+
      set(OPENEXR_LIBRARIES OpenEXR::OpenEXR)
      set(OPENEXR_INCLUDE_PATHS "")
      set(OPENEXR_VERSION "${OpenEXR_VERSION}")
      set(OPENEXR_FOUND 1)
      return()
    else()
      message(STATUS "Unsupported find_package(OpenEXR) - missing OpenEXR::OpenEXR target (version ${OpenEXR_VERSION})")
    endif()
  endif()
endif()

SET(OPENEXR_LIBRARIES "")
SET(OPENEXR_LIBSEARCH_SUFFIXES "")
file(TO_CMAKE_PATH "$ENV{ProgramFiles}" ProgramFiles_ENV_PATH)

if(WIN32)
    SET(OPENEXR_ROOT "C:/Deploy" CACHE STRING "Path to the OpenEXR \"Deploy\" folder")
    if(X86_64)
        SET(OPENEXR_LIBSEARCH_SUFFIXES x64/Release x64 x64/Debug)
    elseif(MSVC)
        SET(OPENEXR_LIBSEARCH_SUFFIXES Win32/Release Win32 Win32/Debug)
    endif()
elseif(UNIX)
    SET(OPENEXR_LIBSEARCH_SUFFIXES ${CMAKE_LIBRARY_ARCHITECTURE})
endif()

SET(SEARCH_PATHS
    "${OPENEXR_ROOT}"
    /usr
    /usr/local
    /sw
    /opt
    "${ProgramFiles_ENV_PATH}/OpenEXR")

MACRO(FIND_OPENEXR_LIBRARY LIBRARY_NAME LIBRARY_SUFFIX)
    string(TOUPPER "${LIBRARY_NAME}" LIBRARY_NAME_UPPER)
    FIND_LIBRARY(OPENEXR_${LIBRARY_NAME_UPPER}_LIBRARY
        NAMES ${LIBRARY_NAME}${LIBRARY_SUFFIX}
        PATH_SUFFIXES ${OPENEXR_LIBSEARCH_SUFFIXES}
        NO_DEFAULT_PATH
        PATHS "${SEARCH_PATH}/lib" "${SEARCH_PATH}/lib/static")
ENDMACRO()

MACRO(ocv_find_openexr LIBRARY_SUFFIX)
    IF(NOT OPENEXR_FOUND)
        FIND_OPENEXR_LIBRARY("Half" "${LIBRARY_SUFFIX}")
        FIND_OPENEXR_LIBRARY("Iex" "${LIBRARY_SUFFIX}")
        FIND_OPENEXR_LIBRARY("Imath" "${LIBRARY_SUFFIX}")
        FIND_OPENEXR_LIBRARY("IlmImf" "${LIBRARY_SUFFIX}")
        FIND_OPENEXR_LIBRARY("IlmThread" "${LIBRARY_SUFFIX}")
        IF (OPENEXR_INCLUDE_PATH AND OPENEXR_IMATH_LIBRARY AND OPENEXR_ILMIMF_LIBRARY AND OPENEXR_IEX_LIBRARY AND OPENEXR_HALF_LIBRARY AND OPENEXR_ILMTHREAD_LIBRARY)
            SET(OPENEXR_FOUND TRUE)
        ELSE()
            UNSET(OPENEXR_IMATH_LIBRARY)
            UNSET(OPENEXR_ILMIMF_LIBRARY)
            UNSET(OPENEXR_IEX_LIBRARY)
            UNSET(OPENEXR_ILMTHREAD_LIBRARY)
            UNSET(OPENEXR_HALF_LIBRARY)
        ENDIF()
    ENDIF()
ENDMACRO()

FOREACH(SEARCH_PATH ${SEARCH_PATHS})
    FIND_PATH(OPENEXR_INCLUDE_PATH ImfRgbaFile.h
        PATH_SUFFIXES OpenEXR
        NO_DEFAULT_PATH
        PATHS
        "${SEARCH_PATH}/include")

    IF (OPENEXR_INCLUDE_PATH)
        SET(OPENEXR_VERSION_FILE "${OPENEXR_INCLUDE_PATH}/OpenEXRConfig.h")
        IF (EXISTS ${OPENEXR_VERSION_FILE})
            FILE (STRINGS ${OPENEXR_VERSION_FILE} contents REGEX "#define OPENEXR_VERSION_MAJOR ")
            IF (${contents} MATCHES "#define OPENEXR_VERSION_MAJOR ([0-9]+)")
                SET(OPENEXR_VERSION_MAJOR "${CMAKE_MATCH_1}")
            ENDIF ()
            FILE (STRINGS ${OPENEXR_VERSION_FILE} contents REGEX "#define OPENEXR_VERSION_MINOR ")
            IF (${contents} MATCHES "#define OPENEXR_VERSION_MINOR ([0-9]+)")
                SET(OPENEXR_VERSION_MINOR "${CMAKE_MATCH_1}")
            ENDIF ()
            FILE (STRINGS ${OPENEXR_VERSION_FILE} contents REGEX "#define OPENEXR_VERSION_PATCH ")
            IF (${contents} MATCHES "#define OPENEXR_VERSION_PATCH ([0-9]+)")
                SET(OPENEXR_VERSION_PATCH "${CMAKE_MATCH_1}")
            ENDIF ()
        ENDIF ()
    ENDIF ()

    set(OPENEXR_VERSION_MM "${OPENEXR_VERSION_MAJOR}_${OPENEXR_VERSION_MINOR}")
    set(OPENEXR_VERSION "${OPENEXR_VERSION_MAJOR}.${OPENEXR_VERSION_MINOR}.${OPENEXR_VERSION_PATCH}")

    ocv_find_openexr("-${OPENEXR_VERSION_MM}")
    ocv_find_openexr("-${OPENEXR_VERSION_MM}_s")
    ocv_find_openexr("-${OPENEXR_VERSION_MM}_d")
    ocv_find_openexr("-${OPENEXR_VERSION_MM}_s_d")
    ocv_find_openexr("")
    ocv_find_openexr("_s")
    ocv_find_openexr("_d")
    ocv_find_openexr("_s_d")

    IF (OPENEXR_FOUND)
        BREAK()
    ENDIF()

    UNSET(OPENEXR_INCLUDE_PATH)
    UNSET(OPENEXR_VERSION_FILE)
    UNSET(OPENEXR_VERSION_MAJOR)
    UNSET(OPENEXR_VERSION_MINOR)
    UNSET(OPENEXR_VERSION_MM)
    UNSET(OPENEXR_VERSION)
ENDFOREACH()

IF (OPENEXR_FOUND)
    SET(OPENEXR_INCLUDE_PATHS ${OPENEXR_INCLUDE_PATH} CACHE PATH "The include paths needed to use OpenEXR")
    SET(OPENEXR_LIBRARIES ${OPENEXR_IMATH_LIBRARY} ${OPENEXR_ILMIMF_LIBRARY} ${OPENEXR_IEX_LIBRARY} ${OPENEXR_HALF_LIBRARY} ${OPENEXR_ILMTHREAD_LIBRARY} CACHE STRING "The libraries needed to use OpenEXR" FORCE)
ENDIF ()

IF(OPENEXR_FOUND)
  IF(NOT OPENEXR_FIND_QUIETLY)
    MESSAGE(STATUS "Found OpenEXR: ${OPENEXR_ILMIMF_LIBRARY}")
  ENDIF()
  if(PKG_CONFIG_FOUND AND NOT OPENEXR_VERSION)
    get_filename_component(OPENEXR_LIB_PATH "${OPENEXR_ILMIMF_LIBRARY}" PATH)
    if(EXISTS "${OPENEXR_LIB_PATH}/pkgconfig/OpenEXR.pc")
      execute_process(COMMAND ${PKG_CONFIG_EXECUTABLE} --modversion "${OPENEXR_LIB_PATH}/pkgconfig/OpenEXR.pc"
                      RESULT_VARIABLE PKG_CONFIG_PROCESS
                      OUTPUT_VARIABLE OPENEXR_VERSION
                      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      if(NOT PKG_CONFIG_PROCESS EQUAL 0)
        SET(OPENEXR_VERSION "Unknown")
      endif()
    endif()
  endif()
  if(NOT OPENEXR_VERSION)
    SET(OPENEXR_VERSION "Unknown")
  else()
    if(HAVE_CXX17 AND OPENEXR_VERSION VERSION_LESS "2.3.0")
      message(STATUS "  OpenEXR(ver ${OPENEXR_VERSION}) doesn't support C++17 and higher. Updating OpenEXR 2.3.0+ is required.")
      SET(OPENEXR_FOUND FALSE)
    endif()
  endif()
ELSE()
  IF(OPENEXR_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find OpenEXR library")
  ENDIF()
ENDIF()

MARK_AS_ADVANCED(
    OPENEXR_INCLUDE_PATHS
    OPENEXR_LIBRARIES
    OPENEXR_ILMIMF_LIBRARY
    OPENEXR_IMATH_LIBRARY
    OPENEXR_IEX_LIBRARY
    OPENEXR_HALF_LIBRARY
    OPENEXR_ILMTHREAD_LIBRARY)
