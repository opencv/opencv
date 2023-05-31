# --- VideoInput/DirectShow ---
if(NOT HAVE_DSHOW AND MSVC AND NOT MSVC_VERSION LESS 1500)
  set(HAVE_DSHOW TRUE)
endif()

if(NOT HAVE_DSHOW)
  check_include_file(dshow.h HAVE_DSHOW)
endif()

if(HAVE_DSHOW)
  ocv_add_external_target(dshow "" "" "HAVE_DSHOW")
endif()
