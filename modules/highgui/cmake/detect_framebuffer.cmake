# --- FB ---
ocv_clear_vars(HAVE_FRAMEBUFFER HAVE_FRAMEBUFFER_XVFB)
if(WITH_FRAMEBUFFER OR WITH_FRAMEBUFFER_XVFB)
  if(WITH_FRAMEBUFFER_XVFB)
    try_compile(HAVE_FRAMEBUFFER_XVFB
      "${CMAKE_CURRENT_BINARY_DIR}"
      "${OpenCV_SOURCE_DIR}/cmake/checks/framebuffer.cpp")
    if(HAVE_FRAMEBUFFER_XVFB)
      message(STATUS "Check virtual framebuffer - done")
    else()
      message(STATUS
        "Check virtual framebuffer - faild\n"
        "Please install the xorg-x11-proto-devel or x11proto-dev package\n")
    endif()
  endif()
  set(HAVE_FRAMEBUFFER ON)
endif()
