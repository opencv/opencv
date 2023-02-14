# --- GTK ---
ocv_clear_vars(HAVE_GTK HAVE_GTK3 HAVE_GTHREAD)
if(WITH_GTK)
  ocv_check_modules(GTK3 gtk+-3.0>=3.4 REQUIRED)
  ocv_add_external_target(gtk3 "${GTK3_INCLUDE_DIRS}" "${GTK3_LIBRARIES}" "HAVE_GTK3;HAVE_GTK")
  set(HAVE_GTK TRUE)
  ocv_check_modules(GTHREAD gthread-2.0 REQUIRED)
  ocv_add_external_target(gthread "${GTHREAD_INCLUDE_DIRS}" "${GTHREAD_LIBRARIES}" "HAVE_GTHREAD")
endif()
