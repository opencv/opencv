// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_IMGPROC_WARP_COMMON_HPP__
#define __OPENCV_IMGPROC_WARP_COMMON_HPP__

#include "opencv2/core/hal/intrin.hpp"

#include "warp_common.vector.hpp"
#include "warp_common.scalar.hpp"

namespace cv {

typedef void (*ImgWarpFunc)(const float* x, const float* y, int len,
                            const void* src, size_t srcstep, Size size,
                            void* dst, const float* params,
                            int borderType, const void* borderVal);

ImgWarpFunc getImgWarpFunc(int type, int interpolation);

#if CV_NEON && CV_SIMD128
static inline void packRGBWarpWeights(const float* weights, v_float32x4* packed)
{
    packed[0] = v_float32x4(weights[0], weights[0], weights[0], weights[1]);
    packed[1] = v_float32x4(weights[1], weights[1], weights[2], weights[2]);
    packed[2] = v_float32x4(weights[2], weights[3], weights[3], weights[3]);
}

static inline v_float32x4 sumFourRGBWarpTaps(const v_float32x4* taps)
{
    const v_float32x4 tap1 = v_rotate_right<3>(taps[0], taps[1]);
    const v_float32x4 tap2 = v_rotate_right<2>(taps[1], taps[2]);
    const v_float32x4 tap3 = v_rotate_right<1>(taps[2]);
    return v_add(v_add(v_add(taps[0], tap1), tap2), tap3);
}

static inline void storeWarpRGB(float* dst, const v_float32x4& rgb)
{
    v_store_low(dst, rgb);
    dst[2] = v_extract_n<2>(rgb);
}
#endif

}

#endif // __OPENCV_IMGPROC_WARP_COMMON_HPP__
