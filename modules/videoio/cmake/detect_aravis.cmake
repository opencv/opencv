# --- Aravis SDK ---
if(NOT HAVE_ARAVIS_API AND PKG_CONFIG_FOUND)
  ocv_check_modules(ARAVIS aravis-0.6 QUIET)
  if(ARAVIS_FOUND)
    set(HAVE_ARAVIS_API TRUE)
  endif()
endif()

if(NOT HAVE_ARAVIS_API)
  find_path(ARAVIS_INCLUDE "arv.h"
    PATHS "${ARAVIS_ROOT}" ENV ARAVIS_ROOT
    PATH_SUFFIXES "include/aravis-0.6"
    NO_DEFAULT_PATH)
  find_library(ARAVIS_LIBRARY "aravis-0.6"
    PATHS "${ARAVIS_ROOT}" ENV ARAVIS_ROOT
    PATH_SUFFIXES "lib"
    NO_DEFAULT_PATH)
  if(ARAVIS_INCLUDE AND ARAVIS_LIBRARY)
    set(HAVE_ARAVIS_API TRUE)
    file(STRINGS "${ARAVIS_INCLUDE}/arvversion.h" ver_strings REGEX "#define +ARAVIS_(MAJOR|MINOR|MICRO)_VERSION.*")
    string(REGEX REPLACE ".*ARAVIS_MAJOR_VERSION[^0-9]+([0-9]+).*" "\\1" ver_major "${ver_strings}")
    string(REGEX REPLACE ".*ARAVIS_MINOR_VERSION[^0-9]+([0-9]+).*" "\\1" ver_minor "${ver_strings}")
    string(REGEX REPLACE ".*ARAVIS_MICRO_VERSION[^0-9]+([0-9]+).*" "\\1" ver_micro "${ver_strings}")
    set(ARAVIS_VERSION "${ver_major}.${ver_minor}.${ver_micro}")  # informational
    set(ARAVIS_INCLUDE_DIRS "${ARAVIS_INCLUDE}")
    set(ARAVIS_LIBRARIES "${ARAVIS_LIBRARY}")
  endif()
endif()

if(HAVE_ARAVIS_API)
  ocv_add_external_target(aravis "${ARAVIS_INCLUDE_DIRS}" "${ARAVIS_LIBRARIES}" "HAVE_ARAVIS_API")
endif()
