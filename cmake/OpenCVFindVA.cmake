# Output:
#   HAVE_VA - libva is available
#   HAVE_VA_INTEL - OpenCL/libva Intel interoperability extension is available

if(UNIX AND NOT ANDROID)
    find_path(
    VA_INCLUDE_DIR
    NAMES va/va.h
    PATHS "/usr/include"
    PATH_SUFFIXES include
    DOC "Path to libva headers")
endif()

if(VA_INCLUDE_DIR)
    set(HAVE_VA TRUE)
    if(NOT DEFINED VA_LIBRARIES)
      set(VA_LIBRARIES "va" "va-drm")
    endif()
else()
    set(HAVE_VA FALSE)
    message(WARNING "libva installation is not found.")
endif()
