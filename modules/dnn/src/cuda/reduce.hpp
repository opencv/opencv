// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_REDUCE_HPP
#define OPENCV_DNN_SRC_CUDA_REDUCE_HPP

#include <cuda_runtime.h>

template <class T>
__device__ T warpReduceSum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

#endif /* OPENCV_DNN_SRC_CUDA_REDUCE_HPP */
