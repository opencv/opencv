# Main variables:
# VAAPI_MSDK_INCLUDE_DIR and VAAPI_IOCL_INCLUDE_DIR to use VAAPI
# HAVE_VAAPI for conditional compilation OpenCV with/without VAAPI

# VAAPI_MSDK_ROOT - root of Intel MSDK installation
# VAAPI_IOCL_ROOT - root of Intel OCL installation

if(UNIX AND NOT ANDROID)
    if($ENV{VAAPI_MSDK_ROOT})
        set(VAAPI_MSDK_ROOT $ENV{VAAPI_MSDK_ROOT})
    else()
        set(VAAPI_MSDK_ROOT "/opt/intel/mediasdk")
    endif()

    if($ENV{VAAPI_IOCL_ROOT})
        set(VAAPI_IOCL_ROOT $ENV{VAAPI_IOCL_ROOT})
    else()
        set(VAAPI_IOCL_ROOT "/opt/intel/opencl")
    endif()

    find_path(
    VAAPI_MSDK_INCLUDE_DIR
    NAMES mfxdefs.h
    PATHS ${VAAPI_MSDK_ROOT}
    PATH_SUFFIXES include
    DOC "Path to Intel MSDK headers")

    find_path(
    VAAPI_IOCL_INCLUDE_DIR
    NAMES CL/va_ext.h
    PATHS ${VAAPI_IOCL_ROOT}
    PATH_SUFFIXES include
    DOC "Path to Intel OpenCL headers")
endif()

if(VAAPI_MSDK_INCLUDE_DIR AND VAAPI_IOCL_INCLUDE_DIR)
    set(HAVE_VAAPI TRUE)
    set(VAAPI_EXTRA_LIBS "-lva" "-lva-drm")
else()
    set(HAVE_VAAPI FALSE)
    message(WARNING "Intel MSDK & OpenCL installation is not found.")
endif()

mark_as_advanced(FORCE VAAPI_MSDK_INCLUDE_DIR VAAPI_IOCL_INCLUDE_DIR)
