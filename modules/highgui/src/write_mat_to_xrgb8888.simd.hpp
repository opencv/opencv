// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"


namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void write_g8_to_xrgb8888      (const uint8_t* src, uint8_t *dst, const int img_cols);
void write_bgr888_to_xrgb8888  (const uint8_t* src, uint8_t *dst, const int img_cols);
void write_bgra8888_to_xrgb8888(const uint8_t* src, uint8_t *dst, const int img_cols);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

// Convert from [g8] to [b8:g8:r8:x8]
void write_g8_to_xrgb8888(const uint8_t* src, uint8_t *dst, const int img_cols)
{
    int x = 0;
#if CV_SIMD
#if CV_SIMD256
    for (; x < img_cols - 32; x+=32, src+=32*1, dst+=32*4)
    {
        const cv::v_uint8x32 vG = cv::v256_load(src); // Gray
        cv::v_store_interleave (dst, vG, vG, vG, vG); // BGRx (x is any).
    }
#endif // CV_SIMD256
#if CV_SIMD128
    for (; x < img_cols - 16; x+=16, src+=16*1, dst+=16*4)
    {
        const cv::v_uint8x16 vG = cv::v_load(src);    // Gray
        cv::v_store_interleave (dst, vG, vG, vG, vG); // BGRx (x is any).
    }
#endif // CV_SIMD128
#endif // CV_SIMD

    // tail
    for (; x < img_cols; x++, src++, dst+=4)
    {
        const uint8_t g = src[0];
        dst[0] = g;
        dst[1] = g;
        dst[2] = g;
    }
}

// Convert from [b8:g8:r8] to [b8:g8:r8:x8]
void write_bgr888_to_xrgb8888(const uint8_t* src, uint8_t *dst, const int img_cols)
{
    int x = 0;

#if CV_SIMD
#if CV_SIMD256
    for (; x < img_cols - 32; x+=32, src+=32*3, dst+=32*4)
    {
        cv::v_uint8x32 vB, vG, vR;
        cv::v_load_deinterleave(src, vB, vG, vR);     // BGR
        cv::v_store_interleave (dst, vB, vG, vR, vR); // BGRx (x is any).
    }
#endif // CV_SIMD256
#if CV_SIMD128
    for (; x < img_cols - 16; x+=16, src+=16*3, dst+=16*4)
    {
        cv::v_uint8x16 vB, vG, vR;
        cv::v_load_deinterleave(src, vB, vG, vR);     // BGR
        cv::v_store_interleave (dst, vB, vG, vR, vR); // BGRx (x is any).
    }
#endif // CV_SIMD128
#endif // CV_SIMD

    // tail
    for (; x < img_cols; x++, src+=3, dst+=4)
    {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
    }
}

// Convert from [b8:g8:r8:a8] to [b8:g8:r8:x8]
void write_bgra8888_to_xrgb8888(const uint8_t* src, uint8_t *dst, const int img_cols)
{
    memcpy(dst, src, img_cols * 4);
}

#endif// CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
