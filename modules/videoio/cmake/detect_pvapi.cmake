# --- PvApi ---
if(NOT HAVE_PVAPI)
  if(X86_64)
    set(arch x64)
  else()
    set(arch x86)
  endif()
  find_path(PVAPI_INCLUDE "PvApi.h"
    PATHS "${PVAPI_ROOT}" ENV PVAPI_ROOT
    PATH_SUFFIXES "inc-pc")
  find_library(PVAPI_LIBRARY "PvAPI"
    PATHS "${PVAPI_ROOT}" ENV PVAPI_ROOT
    PATH_SUFFIXES "bin-pc/${arch}/${gcc}")
  if(PVAPI_INCLUDE AND PVAPI_LIBRARY)
    set(HAVE_PVAPI TRUE)
  endif()
endif()

if(HAVE_PVAPI)
  ocv_add_external_target(pvapi "${PVAPI_INCLUDE}" "${PVAPI_LIBRARY}" "HAVE_PVAPI")
endif()
