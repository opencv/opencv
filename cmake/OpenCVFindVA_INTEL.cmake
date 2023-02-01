# Main variables:
# VA_INTEL_IOCL_INCLUDE_DIR to use VA_INTEL
# HAVE_VA_INTEL for conditional compilation OpenCV with/without VA_INTEL

# VA_INTEL_IOCL_ROOT - root of Intel OCL installation

if(UNIX AND NOT ANDROID)
    ocv_check_environment_variables(VA_INTEL_IOCL_ROOT)
    if(NOT DEFINED VA_INTEL_IOCL_ROOT)
      set(VA_INTEL_IOCL_ROOT "/opt/intel/opencl")
    endif()

    find_path(
    VA_INTEL_IOCL_INCLUDE_DIR
    NAMES CL/va_ext.h
    PATHS ${VA_INTEL_IOCL_ROOT}
    PATH_SUFFIXES include
    DOC "Path to Intel OpenCL headers")
endif()

if(VA_INTEL_IOCL_INCLUDE_DIR)
    set(HAVE_VA_INTEL TRUE)
    if(NOT DEFINED VA_INTEL_LIBRARIES)
      set(VA_INTEL_LIBRARIES "va" "va-drm")
    endif()
else()
    set(HAVE_VA_INTEL FALSE)
    message(WARNING "Intel OpenCL installation is not found.")
endif()

mark_as_advanced(FORCE VA_INTEL_IOCL_INCLUDE_DIR)
