// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CUDA_CUDA_COMPAT_HPP
#define OPENCV_CUDA_CUDA_COMPAT_HPP

#include <cuda.h>

namespace cv { namespace cuda { namespace device { namespace compat
{
#if CUDA_VERSION >= 13000
    using ulonglong4 = ::ulonglong4_16a;
    using double4 = ::double4_16a;
    __host__ __device__ __forceinline__
    double4 make_double4(double x, double y, double z, double w)
    {
        return ::make_double4_16a(x, y, z, w);
    }
#else
    using ulonglong4 = ::ulonglong4;
    using double4 = ::double4;
    __host__ __device__ __forceinline__
    double4 make_double4(double x, double y, double z, double w)
    {
        return ::make_double4(x, y, z, w);
    }
#endif
    using ulonglong4Compat = ulonglong4;
    using double4Compat = double4;
    __host__ __device__ __forceinline__
    double4Compat make_double4_compat(double x, double y, double z, double w)
    {
        return make_double4(x, y, z, w);
    }
}}}}

#endif // OPENCV_CUDA_CUDA_COMPAT_HPP