if(NOT HAVE_UEYE)
  if(WIN32)
    if(X86_64)
      set(_WIN_LIB_SUFFIX "_64")
    endif()
  endif()
  find_path(UEYE_INCLUDE "ueye.h"
    PATHS "${UEYE_ROOT}" ENV UEYE_ROOT "/usr" "C:/Program Files/IDS/uEye/Develop"
    HINTS "${regpath}"
    PATH_SUFFIXES "include")
  find_library(UEYE_LIBRARY ueye_api${_WIN_LIB_SUFFIX}
    PATHS "${UEYE_ROOT}" ENV UEYE_ROOT "/usr" "C:/Program Files/IDS/uEye/Develop"
    HINTS "${regpath}"
    PATH_SUFFIXES "lib")
  if(UEYE_INCLUDE AND UEYE_LIBRARY)
    set(HAVE_UEYE TRUE)
  endif()
endif()
unset(_WIN_LIB_SUFFIX)

if(HAVE_UEYE)
  ocv_add_external_target(ueye "${UEYE_INCLUDE}" "${UEYE_LIBRARY}" "HAVE_UEYE")
endif()
