// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025.

#include "rvv_hal.hpp"
#include "common.hpp"

#define CV_DESCALE(x,n) (((x) + (1 << ((n)-1))) >> (n))

namespace cv { namespace rvv_hal { namespace imgproc {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

static const int G2Y = 9617;
static const int SHIFT = 14;

static int bayer2Gray_8u_rows(int start, int end,
                               const uchar* src_data, size_t src_step,
                               uchar* dst_data, size_t dst_step,
                               int width, int start_with_green,
                               int bcoeff, int rcoeff)
{
    int b_step = (int)src_step;
    int d_step = (int)dst_step;

    int swg = start_with_green ^ (start & 1);
    int bc = bcoeff, rc = rcoeff;
    if (start & 1)
        std::swap(bc, rc);

    for (int y = start; y < end; ++y)
    {
        const uchar* bayer = src_data + y * b_step;
        uchar* drow = dst_data + y * d_step;
        uchar* const drow_base = drow;

        const uchar* bayer_end = bayer + width;

        if (width <= 0)
        {
            drow[-1] = drow[width] = 0;
            swg = !swg; std::swap(bc, rc);
            continue;
        }

        if (swg)
        {
            unsigned t0 = (bayer[1] + bayer[b_step * 2 + 1]) * rc;
            unsigned t1 = (bayer[b_step] + bayer[b_step + 2]) * bc;
            unsigned t2 = bayer[b_step + 1] * (2 * G2Y);
            drow[0] = (uchar)CV_DESCALE(t0 + t1 + t2, SHIFT + 1);
            bayer++;
            drow++;
        }

        for (; bayer + 4 <= bayer_end; )
        {
            int remaining = (int)(bayer_end - bayer);
            size_t vl_src = __riscv_vsetvl_e16m1(remaining / 2 + 1);
            size_t vl = vl_src - 1;
            if (vl < 4) break;

            vuint8mf2_t r0_B_u8 = __riscv_vlse8_v_u8mf2(bayer, 2, vl_src);
            vuint8mf2_t r0_G_u8 = __riscv_vlse8_v_u8mf2(bayer + 1, 2, vl_src);
            vuint8mf2_t r1_G_u8 = __riscv_vlse8_v_u8mf2(bayer + b_step, 2, vl_src);
            vuint8mf2_t r1_R_u8 = __riscv_vlse8_v_u8mf2(bayer + b_step + 1, 2, vl_src);
            vuint8mf2_t r2_B_u8 = __riscv_vlse8_v_u8mf2(bayer + b_step * 2, 2, vl_src);
            vuint8mf2_t r2_G_u8 = __riscv_vlse8_v_u8mf2(bayer + b_step * 2 + 1, 2, vl_src);

            vuint16m1_t r0_B = __riscv_vwaddu_vx_u16m1(r0_B_u8, 0, vl_src);
            vuint16m1_t r0_G = __riscv_vwaddu_vx_u16m1(r0_G_u8, 0, vl_src);
            vuint16m1_t r1_G = __riscv_vwaddu_vx_u16m1(r1_G_u8, 0, vl_src);
            vuint16m1_t r1_R = __riscv_vwaddu_vx_u16m1(r1_R_u8, 0, vl_src);
            vuint16m1_t r2_B = __riscv_vwaddu_vx_u16m1(r2_B_u8, 0, vl_src);
            vuint16m1_t r2_G = __riscv_vwaddu_vx_u16m1(r2_G_u8, 0, vl_src);

            vuint16m1_t b1_ = __riscv_vadd_vv_u16m1(r0_B, r2_B, vl_src);
            vuint16m1_t b1  = __riscv_vslidedown_vx_u16m1(b1_, 1, vl_src);
            vuint16m1_t b0  = __riscv_vadd_vv_u16m1(b1_, b1, vl);

            vuint16m1_t r1_G_next = __riscv_vslidedown_vx_u16m1(r1_G, 1, vl_src);
            vuint16m1_t g0 = __riscv_vadd_vv_u16m1(r0_G, r2_G, vl);
            g0 = __riscv_vadd_vv_u16m1(g0, __riscv_vadd_vv_u16m1(r1_G, r1_G_next, vl), vl);
            vuint16m1_t g1 = __riscv_vsll_vx_u16m1(r1_G_next, 1, vl);

            vuint16m1_t r1_R_next = __riscv_vslidedown_vx_u16m1(r1_R, 1, vl_src);
            vuint16m1_t r0_ = r1_R;
            vuint16m1_t r1_ = __riscv_vadd_vv_u16m1(r1_R, r1_R_next, vl);
            r0_ = __riscv_vsll_vx_u16m1(r0_, 2, vl);

            vint16m1_t b0_s = __riscv_vreinterpret_v_u16m1_i16m1(b0);
            vint16m1_t b1_s = __riscv_vreinterpret_v_u16m1_i16m1(b1);
            vint16m1_t g0_s = __riscv_vreinterpret_v_u16m1_i16m1(g0);
            vint16m1_t g1_s = __riscv_vreinterpret_v_u16m1_i16m1(g1);
            vint16m1_t r0_s = __riscv_vreinterpret_v_u16m1_i16m1(r0_);
            vint16m1_t r1_s = __riscv_vreinterpret_v_u16m1_i16m1(r1_);

            vint32m2_t b0_32 = __riscv_vwadd_vx_i32m2(b0_s, 0, vl);
            vint32m2_t b1_32 = __riscv_vwadd_vx_i32m2(b1_s, 0, vl);
            vint32m2_t g0_32 = __riscv_vwadd_vx_i32m2(g0_s, 0, vl);
            vint32m2_t g1_32 = __riscv_vwadd_vx_i32m2(g1_s, 0, vl);
            vint32m2_t r0_32 = __riscv_vwadd_vx_i32m2(r0_s, 0, vl);
            vint32m2_t r1_32 = __riscv_vwadd_vx_i32m2(r1_s, 0, vl);

            vint32m2_t b0_mul = __riscv_vmul_vx_i32m2(b0_32, rc, vl);
            vint32m2_t b1_mul = __riscv_vmul_vx_i32m2(b1_32, rc, vl);
            vint32m2_t g0_mul = __riscv_vmul_vx_i32m2(g0_32, G2Y, vl);
            vint32m2_t g1_mul = __riscv_vmul_vx_i32m2(g1_32, G2Y, vl);
            vint32m2_t r0_mul = __riscv_vmul_vx_i32m2(r0_32, bc, vl);
            vint32m2_t r1_mul = __riscv_vmul_vx_i32m2(r1_32, bc, vl);

            vint32m2_t gray_even_32 = __riscv_vadd_vv_i32m2(
                __riscv_vadd_vv_i32m2(b0_mul, g0_mul, vl), r0_mul, vl);
            vint32m2_t gray_odd_32 = __riscv_vadd_vv_i32m2(
                __riscv_vadd_vv_i32m2(b1_mul, g1_mul, vl), r1_mul, vl);

            vint32m2_t gray_even_r = __riscv_vadd_vx_i32m2(gray_even_32, (1 << (SHIFT + 1)), vl);
            vint32m2_t gray_odd_r  = __riscv_vadd_vx_i32m2(gray_odd_32,  (1 << SHIFT), vl);
            vint16m1_t gray_even = __riscv_vnsra_wx_i16m1(gray_even_r, SHIFT + 2, vl);
            vint16m1_t gray_odd  = __riscv_vnsra_wx_i16m1(gray_odd_r,  SHIFT + 1, vl);

            vuint16m1_t gray_even_u16 = __riscv_vreinterpret_v_i16m1_u16m1(gray_even);
            vuint16m1_t gray_odd_u16  = __riscv_vreinterpret_v_i16m1_u16m1(gray_odd);
            vuint8mf2_t gray_even_u8 = __riscv_vnclipu_wx_u8mf2(gray_even_u16, 0, __RISCV_VXRM_RNU, vl);
            vuint8mf2_t gray_odd_u8  = __riscv_vnclipu_wx_u8mf2(gray_odd_u16,  0, __RISCV_VXRM_RNU, vl);

            __riscv_vsse8_v_u8mf2(drow, 2, gray_even_u8, vl);
            __riscv_vsse8_v_u8mf2(drow + 1, 2, gray_odd_u8, vl);

            bayer += vl * 2;
            drow += vl * 2;
        }

        for (; bayer <= bayer_end - 2; bayer += 2, drow += 2)
        {
            unsigned t0 = (bayer[0] + bayer[2] + bayer[b_step * 2] + bayer[b_step * 2 + 2]) * rc;
            unsigned t1 = (bayer[1] + bayer[b_step] + bayer[b_step + 2] + bayer[b_step * 2 + 1]) * G2Y;
            unsigned t2 = bayer[b_step + 1] * (4 * bc);
            drow[0] = (uchar)CV_DESCALE(t0 + t1 + t2, SHIFT + 2);

            t0 = (bayer[2] + bayer[b_step * 2 + 2]) * rc;
            t1 = (bayer[b_step + 1] + bayer[b_step + 3]) * bc;
            t2 = bayer[b_step + 2] * (2 * G2Y);
            drow[1] = (uchar)CV_DESCALE(t0 + t1 + t2, SHIFT + 1);
        }

        if (bayer < bayer_end)
        {
            unsigned t0 = (bayer[0] + bayer[2] + bayer[b_step * 2] + bayer[b_step * 2 + 2]) * rc;
            unsigned t1 = (bayer[1] + bayer[b_step] + bayer[b_step + 2] + bayer[b_step * 2 + 1]) * G2Y;
            unsigned t2 = bayer[b_step + 1] * (4 * bc);
            drow[0] = (uchar)CV_DESCALE(t0 + t1 + t2, SHIFT + 2);
            bayer++;
            drow++;
        }

        drow_base[-1] = drow_base[0];
        drow_base[width] = drow_base[width - 1];

        swg = !swg;
        std::swap(bc, rc);
    }

    return CV_HAL_ERROR_OK;
}

static int bayer2Gray_8u(const uchar* src, int src_step,
                          uchar* dst, int dst_step,
                          int src_width, int src_height,
                          int start_with_green, int bcoeff, int rcoeff)
{
    int ew = src_width - 2;
    int eh = src_height - 2;
    if (ew <= 0 || eh <= 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int b_step = src_step;
    int d_step = dst_step;

    const uchar* interior_src = src;
    uchar* interior_dst = dst + d_step + 1;

    common::invoke(eh, {bayer2Gray_8u_rows},
                   interior_src, (size_t)b_step,
                   interior_dst, (size_t)d_step,
                   ew, start_with_green, bcoeff, rcoeff);

    if (src_height > 2)
    {
        for (int i = 0; i < src_width; i++)
        {
            dst[i] = dst[i + d_step];
            dst[i + (src_height - 1) * d_step] = dst[i + (src_height - 2) * d_step];
        }
    }
    else
    {
        for (int i = 0; i < src_width; i++)
            dst[i] = dst[i + (src_height - 1) * d_step] = 0;
    }

    return CV_HAL_ERROR_OK;
}

static int bayer2BGR_8u(const uchar* src, int src_step,
                         uchar* dst, int dst_step,
                         int src_width, int src_height,
                         int blue, int start_with_green)
{
    CV_UNUSED(src); CV_UNUSED(src_step); CV_UNUSED(dst); CV_UNUSED(dst_step);
    CV_UNUSED(src_width); CV_UNUSED(src_height);
    CV_UNUSED(blue); CV_UNUSED(start_with_green);
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

static int bayer2BGRA_8u(const uchar* src, int src_step,
                          uchar* dst, int dst_step,
                          int src_width, int src_height,
                          int blue, int start_with_green)
{
    CV_UNUSED(src); CV_UNUSED(src_step); CV_UNUSED(dst); CV_UNUSED(dst_step);
    CV_UNUSED(src_width); CV_UNUSED(src_height);
    CV_UNUSED(blue); CV_UNUSED(start_with_green);
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

} // namespace

int demosaicing(const uchar* src_data, size_t src_step, int src_type,
                int src_width, int src_height,
                uchar* dst_data, size_t dst_step, int dst_type,
                int dcn, int blue, int start_with_green, int bcoeff, int rcoeff)
{
    int depth = CV_MAT_DEPTH(src_type);
    int ddepth = CV_MAT_DEPTH(dst_type);
    if (depth != CV_8U || ddepth != CV_8U || CV_MAT_CN(dst_type) != dcn)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int sstep = (int)src_step;
    int dstep = (int)dst_step;

    if (dcn == 1)
        return bayer2Gray_8u(src_data, sstep, dst_data, dstep,
                              src_width, src_height,
                              start_with_green, bcoeff, rcoeff);

    if (dcn == 3)
        return bayer2BGR_8u(src_data, sstep, dst_data, dstep,
                             src_width, src_height, blue, start_with_green);
    if (dcn == 4)
        return bayer2BGRA_8u(src_data, sstep, dst_data, dstep,
                              src_width, src_height, blue, start_with_green);

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::imgproc
