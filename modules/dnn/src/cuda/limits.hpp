// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_LIMITS_HPP
#define OPENCV_DNN_SRC_CUDA_LIMITS_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cfloat>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace device {

    template <class T>
    struct numeric_limits;

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <>
    struct numeric_limits<__half> {
        __device__ static __half min() { return 0.0000610; }
        __device__ static __half max() { return 65504.0; }
        __device__ static __half lowest() { return -65504.0; }
    };
#endif

    template <>
    struct numeric_limits<float> {
        __device__ static float min() { return FLT_MIN; }
        __device__ static float max() { return FLT_MAX; }
        __device__ static float lowest() { return -FLT_MAX; }
    };

}}}}} /* namespace cv::dnn::cuda4dnn::csl::device */

#endif /* OPENCV_DNN_SRC_CUDA_LIMITS_HPP */
