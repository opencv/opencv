if(NOT AARCH64 AND NOT ARM64 AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    return()
endif()

if(NOT WITH_ARMPL)
    return()
endif()

set(ARMPL_ROOT_DIR "" CACHE PATH "Path to ARM Performance Libraries root directory")

if(NOT ARMPL_ROOT_DIR)
  if(DEFINED ENV{ARMPL_DIR})
    set(ARMPL_ROOT_DIR "$ENV{ARMPL_DIR}")
  endif()
endif()

find_path(ARMPL_INCLUDE_DIR
  NAMES armpl.h
  HINTS
    "${ARMPL_ROOT_DIR}/include"
    "${ARMPL_ROOT_DIR}/include_lp64"
  PATHS
    /opt/arm/armpl/include
    /usr/include/armpl
    ENV ARMPL_DIR
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)

if(WITH_OPENMP AND OpenMP_CXX_FOUND)
  set(ARMPL_USE_OPENMP TRUE)
  set(ARMPL_LIB_CANDIDATES
    armpl_lp64_mp
    armpl_ilp64_mp
  )
else()
  set(ARMPL_USE_OPENMP FALSE)
  set(ARMPL_LIB_CANDIDATES
    armpl_lp64
    armpl_ilp64
  )
endif()

set(ARMPL_LIB_FOUND FALSE)
set(ARMPL_LIB_NAME "")
set(ARMPL_LIB_FILE "")

foreach(lib_candidate ${ARMPL_LIB_CANDIDATES})
  if(WIN32)
    set(ARMPL_LIB_FILE_DLL "${ARMPL_ROOT_DIR}/lib/${lib_candidate}.dll.lib")
    set(ARMPL_LIB_FILE_LIB "${ARMPL_ROOT_DIR}/lib/${lib_candidate}.lib")
    if(EXISTS "${ARMPL_LIB_FILE_DLL}")
      set(ARMPL_LIB_FILE "${ARMPL_LIB_FILE_DLL}")
      set(ARMPL_LIB_NAME "${lib_candidate}")
      set(ARMPL_LIB_FOUND TRUE)
      break()
    elseif(EXISTS "${ARMPL_LIB_FILE_LIB}")
      set(ARMPL_LIB_FILE "${ARMPL_LIB_FILE_LIB}")
      set(ARMPL_LIB_NAME "${lib_candidate}")
      set(ARMPL_LIB_FOUND TRUE)
      break()
    endif()
  else()
    set(ARMPL_LIB_FILE "${ARMPL_ROOT_DIR}/lib/lib${lib_candidate}.a")
    if(EXISTS "${ARMPL_LIB_FILE}")
      set(ARMPL_LIB_NAME "${lib_candidate}")
      set(ARMPL_LIB_FOUND TRUE)
      break()
    endif()
  endif()
endforeach()

if(NOT ARMPL_LIB_FOUND)
  find_library(ARMPL_LIBRARY_FALLBACK
    NAMES ${ARMPL_LIB_CANDIDATES}
    HINTS "${ARMPL_ROOT_DIR}/lib"
    PATHS
      /opt/arm/armpl/lib
      /usr/lib/armpl
      ENV ARMPL_DIR
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
  )
  if(ARMPL_LIBRARY_FALLBACK)
    set(ARMPL_LIB_FILE "${ARMPL_LIBRARY_FALLBACK}")
    get_filename_component(ARMPL_LIB_NAME "${ARMPL_LIBRARY_FALLBACK}" NAME_WE)
    string(REGEX REPLACE "^lib" "" ARMPL_LIB_NAME "${ARMPL_LIB_NAME}")
    set(ARMPL_LIB_FOUND TRUE)
  endif()
endif()

if(NOT ARMPL_INCLUDE_DIR OR NOT ARMPL_LIB_FOUND)
  message(WARNING
    "ARM Performance Libraries: NOT FOUND. "
    "Please install ArmPL manually and set -DARMPL_ROOT_DIR=<path>. "
    "Download from: https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Libraries"
  )
  return()
endif()

set(ARMPL_VERSION_STR "unknown")
if(EXISTS "${ARMPL_INCLUDE_DIR}/armpl.h")
  file(STRINGS "${ARMPL_INCLUDE_DIR}/armpl.h" ARMPL_VERSION_MAJOR_LINE
    REGEX "#define ARMPL_VERSION_MAJOR")
  file(STRINGS "${ARMPL_INCLUDE_DIR}/armpl.h" ARMPL_VERSION_MINOR_LINE
    REGEX "#define ARMPL_VERSION_MINOR")
  if(ARMPL_VERSION_MAJOR_LINE AND ARMPL_VERSION_MINOR_LINE)
    string(REGEX REPLACE ".*ARMPL_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1"
      ARMPL_VERSION_MAJOR "${ARMPL_VERSION_MAJOR_LINE}")
    string(REGEX REPLACE ".*ARMPL_VERSION_MINOR[ \t]+([0-9]+).*" "\\1"
      ARMPL_VERSION_MINOR "${ARMPL_VERSION_MINOR_LINE}")
    set(ARMPL_VERSION_STR "${ARMPL_VERSION_MAJOR}.${ARMPL_VERSION_MINOR}")
  else()
    file(STRINGS "${ARMPL_INCLUDE_DIR}/armpl.h" ARMPL_BUILD_LINE
      REGEX "#define ARMPL_BUILD")
    if(ARMPL_BUILD_LINE)
      string(REGEX REPLACE ".*ARMPL_BUILD[ \t]+([0-9]+).*" "\\1"
        ARMPL_VERSION_STR "${ARMPL_BUILD_LINE}")
    else()
      string(REGEX MATCH "armpl_([0-9]+\\.[0-9]+)" ARMPL_VERSION_MATCH "${ARMPL_ROOT_DIR}")
      if(CMAKE_MATCH_1)
        set(ARMPL_VERSION_STR "${CMAKE_MATCH_1}")
      endif()
    endif()
  endif()
endif()

if(ARMPL_USE_OPENMP)
  message(STATUS "ArmPL: OpenMP enabled, using parallel version (${ARMPL_LIB_NAME})")
else()
  message(WARNING
    "ArmPL: OpenMP is not enabled. "
    "Using serial version of ArmPL (${ARMPL_LIB_NAME}). "
    "For better performance enable OpenMP with -DWITH_OPENMP=ON"
  )
endif()

if(NOT TARGET armpl)
  if(WIN32)
    add_library(armpl SHARED IMPORTED)
    find_file(ARMPL_DLL
      NAMES "${ARMPL_LIB_NAME}.dll"
      HINTS "${ARMPL_ROOT_DIR}/bin"
      NO_DEFAULT_PATH
    )
    set_target_properties(armpl PROPERTIES
      IMPORTED_IMPLIB               "${ARMPL_LIB_FILE}"
      IMPORTED_LOCATION             "${ARMPL_DLL}"
      INTERFACE_INCLUDE_DIRECTORIES "${ARMPL_INCLUDE_DIR}"
    )
  else()
    add_library(armpl UNKNOWN IMPORTED)
    set_target_properties(armpl PROPERTIES
      IMPORTED_LOCATION             "${ARMPL_LIB_FILE}"
      INTERFACE_INCLUDE_DIRECTORIES "${ARMPL_INCLUDE_DIR}"
    )
  endif()
  if(ARMPL_USE_OPENMP)
    set_target_properties(armpl PROPERTIES
      INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX
    )
  endif()
endif()

set(ARMPL_LIBRARIES    armpl                  CACHE INTERNAL "ArmPL libraries")
set(ARMPL_INCLUDE_DIRS "${ARMPL_INCLUDE_DIR}" CACHE INTERNAL "ArmPL include dirs")
set(ARMPL_INCLUDE_PATH "${ARMPL_INCLUDE_DIR}" CACHE INTERNAL "ArmPL include path")
set(ARMPL_LIBRARY      "${ARMPL_LIB_FILE}"    CACHE INTERNAL "ArmPL library path")
set(ARMPL_LIB_NAME     "${ARMPL_LIB_NAME}"    CACHE INTERNAL "ArmPL library variant")
set(ARMPL_VERSION_STR  "${ARMPL_VERSION_STR}" CACHE INTERNAL "ArmPL version")
set(ARMPL_ROOT_DIR     "${ARMPL_ROOT_DIR}"    CACHE PATH     "ArmPL root directory")
set(HAVE_ARMPL         TRUE                   CACHE BOOL     "ArmPL found and enabled" FORCE)
