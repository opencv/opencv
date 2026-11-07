# --- FB ---
set(HAVE_FRAMEBUFFER ON)
if(WITH_FRAMEBUFFER_XVFB)
  try_compile(HAVE_FRAMEBUFFER_XVFB
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/framebuffer.cpp")
  if(HAVE_FRAMEBUFFER_XVFB)
    message(STATUS "Check virtual framebuffer - done")
  else()
    message(STATUS
      "Check virtual framebuffer - failed\n"
      "Please install the xorg-x11-proto-devel or x11proto-dev package\n")
  endif()
endif()
