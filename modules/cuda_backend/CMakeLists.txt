# =============================================================
# cuda_backend — GPU HAL CUDA backend plugin
#
# Implements cv::hal::Backend for CUDA.
# Loaded at runtime by hal_backend.cpp via dlopen.
# Wraps existing cv::cuda:: functions internally.
# User code never calls this module directly.
# =============================================================

# Only build if CUDA is available
if(NOT HAVE_CUDA)
  message(STATUS
    "cuda_backend: CUDA not found — skipping module")
  return()
endif()

# -------------------------------------------------------------
# Define the module
# opencv_cudawarping  — for cuda::resize, warpAffine etc.
# opencv_cudafilters  — for cuda::GaussianBlur etc.
# opencv_cudaimgproc  — for cuda::cvtColor etc.
# opencv_cudaarithm   — for cuda::add, subtract etc.
# All cuda* deps are OPTIONAL — if a contrib module is
# absent the plugin still builds, just supports fewer ops.
# -------------------------------------------------------------
ocv_define_module(cuda_backend
    opencv_core
    OPTIONAL
        opencv_cudawarping
        opencv_cudafilters
        opencv_cudaimgproc
        opencv_cudaarithm
    WRAP
        # no Python wrapping needed — internal plugin
)

# -------------------------------------------------------------
# Must be a SHARED library — dlopen requires .so
# ocv_define_module builds STATIC by default for most modules.
# Override to SHARED explicitly.
# -------------------------------------------------------------
if(TARGET opencv_cuda_backend)
    set_target_properties(opencv_cuda_backend
        PROPERTIES
        # output name — what dlopen searches for
        OUTPUT_NAME "opencv_cuda_backend"

        # no version suffix on the .so file
        # dlopen uses exact filename
        VERSION ""
        SOVERSION ""
    )
endif()

# -------------------------------------------------------------
# Install the plugin .so alongside other OpenCV libraries
# so dlopen can find it without LD_LIBRARY_PATH
# -------------------------------------------------------------
install(
    TARGETS opencv_cuda_backend
    LIBRARY DESTINATION ${OpenCV_INSTALL_PATH}/lib
    COMPONENT cuda_backend
)

message(STATUS
    "cuda_backend: CUDA plugin module configured")
