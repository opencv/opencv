# --- OpenNI2 ---

if(NOT HAVE_OPENNI2)
  set(paths "${OPENNI2_DIR}")
  if(MSVC AND X86_64)
    list(APPEND paths ENV OPENNI2_INCLUDE64 ENV OPENNI2_LIB64 ENV OPENNI2_REDIST64)
  else()
    list(APPEND paths ENV OPENNI2_INCLUDE ENV OPENNI2_LIB ENV OPENNI2_REDIST)
  endif()

  # From SDK
  find_path(OPENNI2_INCLUDE "OpenNI.h"
    PATHS ${paths}
    PATH_SUFFIXES "Include"
    NO_DEFAULT_PATH)
  find_library(OPENNI2_LIBRARY "OpenNI2"
    PATHS ${paths}
    PATH_SUFFIXES "Redist" "Lib"
    NO_DEFAULT_PATH)

  if(OPENNI2_LIBRARY AND OPENNI2_INCLUDE)
    set(HAVE_OPENNI2 TRUE)
    set(OPENNI2_INCLUDE_DIRS "${OPENNI2_INCLUDE}")
    set(OPENNI2_LIBRARIES "${OPENNI2_LIBRARY}")
  endif()
endif()

if(NOT HAVE_OPENNI2)
  # From system
  find_path(OPENNI2_SYS_INCLUDE "OpenNI.h" PATH_SUFFIXES "openni2" "ni2")
  find_library(OPENNI2_SYS_LIBRARY "OpenNI2")

  if(OPENNI2_SYS_LIBRARY AND OPENNI2_SYS_INCLUDE)
    set(HAVE_OPENNI2 TRUE)
    set(OPENNI2_INCLUDE_DIRS "${OPENNI2_SYS_INCLUDE}")
    set(OPENNI2_LIBRARIES "${OPENNI2_SYS_LIBRARY}")
  endif()
endif()

if(HAVE_OPENNI2)
  file(STRINGS "${OPENNI2_INCLUDE_DIRS}/OniVersion.h" ver_strings REGEX "#define +ONI_VERSION_(MAJOR|MINOR|MAINTENANCE|BUILD).*")
  string(REGEX REPLACE ".*ONI_VERSION_MAJOR[^0-9]+([0-9]+).*" "\\1" ver_major "${ver_strings}")
  string(REGEX REPLACE ".*ONI_VERSION_MINOR[^0-9]+([0-9]+).*" "\\1" ver_minor "${ver_strings}")
  string(REGEX REPLACE ".*ONI_VERSION_MAINTENANCE[^0-9]+([0-9]+).*" "\\1" ver_maint "${ver_strings}")
  set(OPENNI2_VERSION "${ver_major}.${ver_minor}.${ver_maint}")  # informational
  ocv_add_external_target(openni2 "${OPENNI2_INCLUDE_DIRS}" "${OPENNI2_LIBRARIES}" "HAVE_OPENNI2")
endif()
