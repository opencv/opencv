# --- Libcamera ---

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
