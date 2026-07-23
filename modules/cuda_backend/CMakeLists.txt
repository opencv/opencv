# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.
# cuda_backend — GPU HAL CUDA backend plugin, loaded at runtime via dlopen.

# Only build if CUDA is available.
if(NOT HAVE_CUDA)
  message(STATUS "cuda_backend: CUDA not found — skipping module")
  return()
endif()

# cuda* deps are OPTIONAL — plugin still builds (fewer ops) if any are absent.
ocv_define_module(cuda_backend
    opencv_core
    OPTIONAL
        opencv_cudawarping
        opencv_cudafilters
        opencv_cudaimgproc
        opencv_cudaarithm
    WRAP
        # no Python wrapping — internal plugin
)

if(TARGET opencv_cuda_backend)
    set_target_properties(opencv_cuda_backend
        PROPERTIES
        OUTPUT_NAME "opencv_cuda_backend"
    )
endif()

# Install the plugin .so alongside other OpenCV libraries so dlopen finds it.
install(
    TARGETS opencv_cuda_backend
    LIBRARY DESTINATION ${OpenCV_INSTALL_PATH}/lib
    COMPONENT cuda_backend
)

message(STATUS
    "cuda_backend: CUDA plugin module configured")
