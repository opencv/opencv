# Force removal of code conditionally compiled with `#if
# HAVE_PTHREAD`.
ocv_update(HAVE_PTHREAD 0)

# There components are disabled because they require
# multi-threaded execution.
ocv_update(WITH_PROTOBUF OFF)
ocv_update(WITH_GSTREAMER OFF)
ocv_update(WITH_IPP OFF)
ocv_update(WITH_ITT OFF)
ocv_update(WITH_OPENCL OFF)
ocv_update(WITH_VA OFF)
ocv_update(WITH_VA_INTEL OFF)

# Disable bindings
ocv_update(BUILD_opencv_python2 OFF)
ocv_update(BUILD_opencv_python3 OFF)
ocv_update(BUILD_JAVA OFF)
ocv_update(BUILD_opencv_java OFF)

# These modules require `#include
# <[thread|mutex|condition_variable|future]>` and linkage into
# `libpthread` to work.
ocv_update(BUILD_opencv_objdetect OFF)
ocv_update(BUILD_opencv_gapi OFF)
ocv_update(BUILD_opencv_dnn OFF)

set(OPJ_USE_THREAD "OFF" CACHE INTERNAL "")
