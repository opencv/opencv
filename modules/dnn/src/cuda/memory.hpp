// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_MEMORY_HPP
#define OPENCV_DNN_SRC_CUDA_MEMORY_HPP

#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace device {

template <class T>
__device__ T load_ldg(const T& src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(&src);
#else
    return src;
#endif
}

template <class T>
__device__ T load_ldg(const T* src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(src);
#else
    return *src;
#endif
}

}}}}} /* namespace cv::dnn::cuda4dnn::csl::device */

#endif /* OPENCV_DNN_SRC_CUDA_MEMORY_HPP */
