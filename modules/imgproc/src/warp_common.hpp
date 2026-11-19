// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_IMGPROC_WARP_COMMON_HPP__
#define __OPENCV_IMGPROC_WARP_COMMON_HPP__

#include <cstring>

#include "opencv2/core/hal/intrin.hpp"

#include "warp_common.vector.hpp"
#include "warp_common.scalar.hpp"

namespace cv {

typedef void (*ImgWarpFunc)(const float* x, const float* y, int len,
                            const void* src, size_t srcstep, Size size,
                            void* dst, const float* params,
                            int borderType, const void* borderVal, int channels);

ImgWarpFunc getImgWarpFunc(int type, int interpolation);

// Interpolate two adjacent RGB pixels from each row, then store exactly three channels.
static inline void bilinearHalfRGB(const hfloat* top, const hfloat* bottom,
                                   float w0, float w1, float w2, float w3, hfloat* dst)
{
#if CV_NEON && CV_FP16 && CV_SIMD128
    // On NEON these four-half loads collectively read only the six samples in each row.
    const v_float32x4 tl = v_load_expand(top);
    const v_float32x4 tr = v_rotate_right<1>(v_load_expand(top + 2));
    const v_float32x4 bl = v_load_expand(bottom);
    const v_float32x4 br = v_rotate_right<1>(v_load_expand(bottom + 2));
    v_float32x4 sum = v_add(v_mul(tl, v_setall_f32(w0)), v_mul(tr, v_setall_f32(w1)));
    sum = v_add(sum, v_mul(bl, v_setall_f32(w2)));
    sum = v_add(sum, v_mul(br, v_setall_f32(w3)));
    hfloat rgb[4];
    v_pack_store(rgb, sum);
    std::memcpy(dst, rgb, 3*sizeof(hfloat));
#else
    for (int c = 0; c < 3; ++c)
        dst[c] = hfloat(float(top[c])*w0 + float(top[c + 3])*w1 +
                        float(bottom[c])*w2 + float(bottom[c + 3])*w3);
#endif
}

}

#endif // __OPENCV_IMGPROC_WARP_COMMON_HPP__
