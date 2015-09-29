# Main variables:
# VA_INTEL_MSDK_INCLUDE_DIR and VA_INTEL_IOCL_INCLUDE_DIR to use VA_INTEL
# HAVE_VA_INTEL for conditional compilation OpenCV with/without VA_INTEL

# VA_INTEL_MSDK_ROOT - root of Intel MSDK installation
# VA_INTEL_IOCL_ROOT - root of Intel OCL installation

if(UNIX AND NOT ANDROID)
    if($ENV{VA_INTEL_MSDK_ROOT})
        set(VA_INTEL_MSDK_ROOT $ENV{VA_INTEL_MSDK_ROOT})
    else()
        set(VA_INTEL_MSDK_ROOT "/opt/intel/mediasdk")
    endif()

    if($ENV{VA_INTEL_IOCL_ROOT})
        set(VA_INTEL_IOCL_ROOT $ENV{VA_INTEL_IOCL_ROOT})
    else()
        set(VA_INTEL_IOCL_ROOT "/opt/intel/opencl")
    endif()

    find_path(
    VA_INTEL_MSDK_INCLUDE_DIR
    NAMES mfxdefs.h
    PATHS ${VA_INTEL_MSDK_ROOT}
    PATH_SUFFIXES include
    DOC "Path to Intel MSDK headers")

    find_path(
    VA_INTEL_IOCL_INCLUDE_DIR
    NAMES CL/va_ext.h
    PATHS ${VA_INTEL_IOCL_ROOT}
    PATH_SUFFIXES include
    DOC "Path to Intel OpenCL headers")
endif()

if(VA_INTEL_MSDK_INCLUDE_DIR AND VA_INTEL_IOCL_INCLUDE_DIR)
    set(HAVE_VA_INTEL TRUE)
    set(VA_INTEL_LIBRARIES "-lva" "-lva-drm")
else()
    set(HAVE_VA_INTEL FALSE)
    message(WARNING "Intel MSDK & OpenCL installation is not found.")
endif()

mark_as_advanced(FORCE VA_INTEL_MSDK_INCLUDE_DIR VA_INTEL_IOCL_INCLUDE_DIR)
