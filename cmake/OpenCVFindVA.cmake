# Output:
#   HAVE_VA - libva is available
#   HAVE_VA_INTEL - OpenCL/libva Intel interoperability extension is available

find_path(
    VA_INCLUDE_DIR
    NAMES va/va.h
    PATHS ${VA_ROOT_DIR}
    PATH_SUFFIXES include
    DOC "Path to libva headers"
)

if(VA_INCLUDE_DIR)
    set(HAVE_VA TRUE)
    if(NOT DEFINED VA_LIBRARIES AND NOT OPENCV_LIBVA_LINK)
      set(VA_LIBRARIES "va" "va-drm")
    endif()
else()
    set(HAVE_VA FALSE)
    message(STATUS "libva: missing va.h header (VA_INCLUDE_DIR)")
endif()
