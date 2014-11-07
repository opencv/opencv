# Main variables:
# IPP_A_LIBRARIES and IPP_A_INCLUDE to use IPP Async
# HAVE_IPP_A for conditional compilation OpenCV with/without IPP Async

# IPP_ASYNC_ROOT - root of IPP Async installation

if(X86_64)
    find_path(
    IPP_A_INCLUDE_DIR
    NAMES ipp_async_defs.h
    PATHS $ENV{IPP_ASYNC_ROOT}
    PATH_SUFFIXES include
    DOC "Path to Intel IPP Async interface headers")

    find_file(
    IPP_A_LIBRARIES
    NAMES ipp_async_preview.lib
    PATHS $ENV{IPP_ASYNC_ROOT}
    PATH_SUFFIXES lib/intel64
    DOC "Path to Intel IPP Async interface libraries")

else()
    find_path(
    IPP_A_INCLUDE_DIR
    NAMES ipp_async_defs.h
    PATHS $ENV{IPP_ASYNC_ROOT}
    PATH_SUFFIXES include
    DOC "Path to Intel IPP Async interface headers")

    find_file(
    IPP_A_LIBRARIES
    NAMES ipp_async_preview.lib
    PATHS $ENV{IPP_ASYNC_ROOT}
    PATH_SUFFIXES lib/ia32
    DOC "Path to Intel IPP Async interface libraries")
endif()

if(IPP_A_INCLUDE_DIR AND IPP_A_LIBRARIES)
    set(HAVE_IPP_A TRUE)
else()
    set(HAVE_IPP_A FALSE)
    message(WARNING "Intel IPP Async library directory (set by IPP_A_LIBRARIES_DIR variable) is not found or does not have Intel IPP Async libraries.")
endif()

mark_as_advanced(FORCE IPP_A_LIBRARIES IPP_A_INCLUDE_DIR)