#--- Win32 UI ---
ocv_clear_vars(HAVE_WIN32UI)
if(WITH_WIN32UI)
  try_compile(HAVE_WIN32UI
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/win32uitest.cpp"
    CMAKE_FLAGS "-DLINK_LIBRARIES:STRING=user32;gdi32")
  if(HAVE_WIN32UI)
    set(__libs "user32" "gdi32")
    if(OpenCV_ARCH STREQUAL "ARM64")
      list(APPEND __libs "comdlg32" "advapi32")
    endif()
    ocv_add_external_target(win32ui "" "${__libs}" "HAVE_WIN32UI")
  endif()
endif()
