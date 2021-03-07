// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_INDEX_HELPERS_HPP
#define OPENCV_DNN_SRC_CUDA_INDEX_HELPERS_HPP

#include "types.hpp"

#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace device {

namespace detail {
    using dim3_member_type = decltype(dim3::x);
    using uint3_member_type = decltype(uint3::x);
}

template <int>  __device__ detail::dim3_member_type getGridDim();
template <> inline __device__ detail::dim3_member_type getGridDim<0>() { return gridDim.x; }
template <> inline __device__ detail::dim3_member_type getGridDim<1>() { return gridDim.y; }
template <> inline __device__ detail::dim3_member_type getGridDim<2>() { return gridDim.z; }

template <int> __device__ detail::dim3_member_type getBlockDim();
template <> inline __device__ detail::dim3_member_type getBlockDim<0>() { return blockDim.x; }
template <> inline __device__ detail::dim3_member_type getBlockDim<1>() { return blockDim.y; }
template <> inline __device__ detail::dim3_member_type getBlockDim<2>() { return blockDim.z; }

template <int> __device__ detail::uint3_member_type getBlockIdx();
template <> inline __device__ detail::uint3_member_type getBlockIdx<0>() { return blockIdx.x; }
template <> inline __device__ detail::uint3_member_type getBlockIdx<1>() { return blockIdx.y; }
template <> inline __device__ detail::uint3_member_type getBlockIdx<2>() { return blockIdx.z; }

template <int> __device__ detail::uint3_member_type getThreadIdx();
template <> inline __device__ detail::uint3_member_type getThreadIdx<0>() { return threadIdx.x; }
template <> inline __device__ detail::uint3_member_type getThreadIdx<1>() { return threadIdx.y; }
template <> inline __device__ detail::uint3_member_type getThreadIdx<2>() { return threadIdx.z; }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::device */

#endif /* OPENCV_DNN_SRC_CUDA_INDEX_HELPERS_HPP */
