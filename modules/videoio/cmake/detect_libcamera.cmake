# --- Libcamera ---

if(NOT HAVE_LIBCAMERA AND PKG_CONFIG_FOUND)
  ocv_check_modules(LIBCAMERA libcamera)
  if(LIBCAMERA_FOUND)
    set(HAVE_LIBCAMERA TRUE)
  endif()
endif()


# libcamera requires C++17. Check if C++17 is available.
if(HAVE_LIBCAMERA)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
  if(NOT COMPILER_SUPPORTS_CXX17)
    message(FATAL_ERROR "libcamera plugin requires C++17, but the compiler does not support it.")
  endif()

  ocv_add_external_target(libcamera "${LIBCAMERA_INCLUDE_DIRS}" "${LIBCAMERA_LINK_LIBRARIES}" "HAVE_LIBCAMERA")
endif()
