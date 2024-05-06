// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"


namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void write_mat_to_xrgb8888(cv::Mat const &img, void *data);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

void write_mat_to_xrgb8888(cv::Mat const &img, void *data) {
    CV_CheckTrue(data != nullptr, "data must not be nullptr.");
    CV_CheckType(img.type(), (img.type() == CV_8UC1) || (img.type() == CV_8UC3) || (img.type() == CV_8UC4),
                 "Only 8UC1/8UC3/8UC4 images are supported.");

    int img_rows = img.rows;
    int img_cols = img.cols;
    uint8_t *dst = (uint8_t*)data;

    // to reduce calling img.ptr()
    if(img.isContinuous())
    {
        img_cols *= img_rows;
        img_rows  = 1;
    }

    switch(img.type())
    {
        case CV_8UC1:
        {
            // Convert from [g8] to [b8:g8:r8:x8]
            for (int y = 0; y < img_rows; y++)
            {
                const uint8_t* src = (uint8_t*)img.ptr(y);
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
        }
        break;

        case CV_8UC3:
        {
            // Convert from [b8:g8:r8] to [b8:g8:r8:x8]
            for (int y = 0; y < img_rows; y++)
            {
                const uint8_t* src = (uint8_t*)img.ptr(y);
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
        }
        break;

        case CV_8UC4:
        {
            // Convert from [b8:g8:r8:a8] to [b8:g8:r8:x8]
            for (int y = 0; y < img_rows; y++, dst+= img_cols * 4)
            {
                const uint8_t* src = (uint8_t*)img.ptr(y);
                memcpy(dst, src, img_cols * 4);
            }
        }
        break;

        default:
        // Do nothing.
        break;
    }
}

#endif// CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
