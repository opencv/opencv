// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_NVCC_DEFS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_NVCC_DEFS_HPP

#include <cuda_runtime_api.h>

#ifdef __CUDACC__
#   define CUDA4DNN_HOST __host__
#   define CUDA4DNN_DEVICE __device__
#   define CUDA4DNN_HOST_DEVICE CUDA4DNN_HOST CUDA4DNN_DEVICE
#else
#   define CUDA4DNN_HOST
#   define CUDA4DNN_DEVICE
#   define CUDA4DNN_HOST_DEVICE
#endif

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_NVCC_DEFS_HPP */
