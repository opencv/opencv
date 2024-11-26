# --- Libcamera ---
<<<<<<< HEAD

if(NOT HAVE_LIBCAMERA AND PKG_CONFIG_FOUND)
  ocv_check_modules(LIBCAMERA libcamera)
  if(LIBCAMERA_FOUND)
    set(HAVE_LIBCAMERA TRUE)
  endif()
endif()

if(HAVE_LIBCAMERA)
  if((CMAKE_CXX_STANDARD EQUAL 98) OR (CMAKE_CXX_STANDARD LESS 17))
    message(STATUS "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} is too old to support libcamera. Use C++17 or later. Turning HAVE_LIBCAMERA off")
    set(HAVE_LIBCAMERA FALSE)
  endif()
endif()

if(HAVE_LIBCAMERA)
  ocv_add_external_target(libcamera "${LIBCAMERA_INCLUDE_DIRS}" "${LIBCAMERA_LINK_LIBRARIES}" "HAVE_LIBCAMERA")
endif()
=======
if(NOT HAVE_LIBCAMERA)
  set(CMAKE_REQUIRED_QUIET TRUE) # for check_include_file
  check_include_file(libcamera/libcamera/libcamera.h HAVE_LIBCAMERA)
  check_include_file(sys/videoio.h HAVE_VIDEOIO)
  if(HAVE_LIBCAMERA OR HAVE_VIDEOIO)
    set(HAVE_LIBCAMERA TRUE)
    set(defs)
    if(HAVE_LIBCAMERA)
      list(APPEND defs "HAVE_LIBCAMERA")
    endif()
    if(HAVE_VIDEOIO)
      list(APPEND defs "HAVE_VIDEOIO")
    endif()
    ocv_add_external_target(libcamera "" "" "${defs}")
  endif()
endif()
>>>>>>> 3b961731d6 (Modify cmake files)
