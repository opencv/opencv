// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "convert.hpp"

namespace cv
{
#if CV_NEON
        const static int cVectorWidth = 4;
#else
        const static int cVectorWidth = 8;
#endif

template <>
int Cvt_SIMD<float16, float16>::operator() (const float16* src, float16* dst, int width) const {
    std::copy(src, src+width, dst);
    return width;
}

template <>
int Cvt_SIMD<float, float16>::operator() (const float* src, float16* dst, int width) const {
    int x = 0;
    for (; x <= width - cVectorWidth; x += cVectorWidth) {
#if CV_NEON

        float32x4_t v_src = vld1q_f32(src + x);

        float16x4_t v_dst = vcvt_f16_f32(v_src);

        cv_vst1_f16((__fp16*)dst + x, v_dst);
#else
        __m256 v_src = _mm256_loadu_ps(src + x);

        // round to nearest even
        __m128i v_dst = _mm256_cvtps_ph(v_src, 0);

        _mm_storeu_si128((__m128i *) (dst + x), v_dst);
#endif
    }
    return x;
}

template <>
int Cvt_SIMD<float16, float>::operator() (const float16* src, float* dst, int width) const {
    int x = 0;
    for (; x <= width - cVectorWidth; x += cVectorWidth) {
#if CV_NEON
        float16x4_t v_src = cv_vld1_f16((__fp16*)src + x);

        float32x4_t v_dst = vcvt_f32_f16(v_src);

        vst1q_f32(dst + x, v_dst);

#else
        __m128i v_src = _mm_loadu_si128((__m128i *) (src + x));

        __m256 v_dst = _mm256_cvtph_ps(v_src);

        _mm256_storeu_ps(dst + x, v_dst);
#endif
    }
    return x;
}

} // cv::
