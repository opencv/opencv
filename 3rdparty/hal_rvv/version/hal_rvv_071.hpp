// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HAL_RVV_071_HPP_INCLUDED
#define OPENCV_HAL_RVV_071_HPP_INCLUDED

#include <riscv_vector.h>

#include <limits>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR cv::cv_hal_rvv::cvtBGRtoBGR

static const unsigned char index_array_32 [32]
                        { 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 18, 17, 16, 19, 22, 21, 20, 23, 26, 25, 24, 27, 30, 29, 28, 31  };

static const unsigned char index_array_24 [24]
                        { 2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21  };

static void vBGRtoBGR(const unsigned char* src, unsigned char * dst, const unsigned char * index, int n, int scn, int dcn, int vsize_pixels, const int vsize)
{
    vuint8m2_t vec_index = vle8_v_u8m2(index, vsize);

    int i = 0;

    for ( ; i <= n-vsize; i += vsize_pixels, src += vsize, dst += vsize)
    {
        vuint8m2_t vec_src = vle8_v_u8m2(src, vsize);
        vuint8m2_t vec_dst = vrgather_vv_u8m2(vec_src, vec_index, vsize);
        vse8_v_u8m2(dst, vec_dst, vsize);
    }

    for ( ; i < n; i++, src += scn, dst += dcn )
    {
        unsigned char t0 = src[0], t1 = src[1], t2 = src[2];
        dst[2] = t0;
        dst[1] = t1;
        dst[0] = t2;
        if(dcn == 4)
        {
            unsigned char d = src[3];
            dst[3] = d;
        }
    }
}

static void sBGRtoBGR(const unsigned char* src, unsigned char * dst, int n, int scn, int dcn, int bi)
{
    for (int i = 0; i < n; i++, src += scn, dst += dcn)
    {
        unsigned char t0 = src[0], t1 = src[1], t2 = src[2];
        dst[bi  ] = t0;
        dst[1]    = t1;
        dst[bi^2] = t2;
        if(dcn == 4)
        {
            unsigned char d = scn == 4 ? src[3] : std::numeric_limits<unsigned char>::max();
            dst[3] = d;
        }
    }
}

static int cvtBGRtoBGR(const unsigned char * src_data, size_t src_step, unsigned char * dst_data, size_t dst_step, int width, int height, int depth, int scn, int dcn, bool swapBlue)
{
    if (depth != CV_8U)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    const int blueIdx = swapBlue ? 2 : 0;
    if (scn == dcn)
    {
        if (!swapBlue)
        {
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        const int vsize_pixels = 8;

        if (scn == 4)
        {
            for (int i = 0; i < height; i++, src_data += src_step, dst_data += dst_step)
            {
                vBGRtoBGR(src_data, dst_data, index_array_32, width, scn, dcn, vsize_pixels, 32);
            }
        }
        else
        {
            for (int i = 0; i < height; i++, src_data += src_step, dst_data += dst_step)
            {
                vBGRtoBGR(src_data, dst_data, index_array_24, width, scn, dcn, vsize_pixels, 24);
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++, src_data += src_step, dst_data += dst_step)
            sBGRtoBGR(src_data, dst_data, width, scn, dcn, blueIdx);
    }

    return CV_HAL_ERROR_OK;
}

}}

#endif
