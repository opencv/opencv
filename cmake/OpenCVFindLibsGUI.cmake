# ----------------------------------------------------------------------------
#  Detect 3rd-party GUI libraries
# ----------------------------------------------------------------------------

#--- Win32 UI ---
ocv_clear_vars(HAVE_WIN32UI)
if(WITH_WIN32UI)
  try_compile(HAVE_WIN32UI
    "${OpenCV_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/win32uitest.cpp"
    CMAKE_FLAGS "-DLINK_LIBRARIES:STRING=user32;gdi32")
endif()

# --- QT4/5 ---
ocv_clear_vars(HAVE_QT HAVE_QT5)
if(WITH_QT)
  if(NOT WITH_QT EQUAL 4)
    find_package(Qt5 COMPONENTS Core Gui Widgets Test Concurrent REQUIRED NO_MODULE)
    if(Qt5_FOUND)
      set(HAVE_QT5 ON)
      set(HAVE_QT  ON)
      find_package(Qt5 COMPONENTS OpenGL QUIET)
      if(Qt5OpenGL_FOUND)
        set(QT_QTOPENGL_FOUND ON)
      endif()
    endif()
  endif()

  if(NOT HAVE_QT)
    find_package(Qt4 REQUIRED QtCore QtGui QtTest)
    if(QT4_FOUND)
      set(HAVE_QT TRUE)
    endif()
  endif()
endif()

# --- OpenGl ---
ocv_clear_vars(HAVE_OPENGL HAVE_QT_OPENGL)
if(WITH_OPENGL)
  if(WITH_WIN32UI OR (HAVE_QT AND QT_QTOPENGL_FOUND) OR HAVE_GTKGLEXT)
    find_package (OpenGL QUIET)
    if(OPENGL_FOUND)
      set(HAVE_OPENGL TRUE)
      list(APPEND OPENCV_LINKER_LIBS ${OPENGL_LIBRARIES})
      if(QT_QTOPENGL_FOUND)
        set(HAVE_QT_OPENGL TRUE)
      else()
        ocv_include_directories(${OPENGL_INCLUDE_DIR})
      endif()
    endif()
  endif()
endif(WITH_OPENGL)

# --- Cocoa ---
if(APPLE)
  if(NOT IOS AND CV_CLANG)
    set(HAVE_COCOA YES)
  endif()
endif()
