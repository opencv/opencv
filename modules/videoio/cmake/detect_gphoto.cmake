# --- gPhoto2 ---
if(NOT HAVE_GPHOTO2 AND PKG_CONFIG_FOUND)
  pkg_check_modules(GPHOTO2 libgphoto2 QUIET)
  if(GPHOTO2_FOUND)
    set(HAVE_GPHOTO2 TRUE)
  endif()
endif()

if(HAVE_GPHOTO2)
  ocv_add_external_target(gphoto2 "${GPHOTO2_INCLUDE_DIRS}" "${GPHOTO2_LIBRARIES}" "HAVE_GPHOTO2")
endif()

set(HAVE_GPHOTO2 ${HAVE_GPHOTO2} PARENT_SCOPE)
