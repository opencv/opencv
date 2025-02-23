// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_WARPAFFINE_HPP_INCLUDED
#define OPENCV_HAL_RVV_WARPAFFINE_HPP_INCLUDED

#include <riscv_vector.h>
namespace cv { namespace cv_hal_rvv {
#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)

#undef cv_hal_warpAffine
#define cv_hal_warpAffine cv::cv_hal_rvv::warpAffine
#undef cv_hal_warpAffineBlocklineNN
#define cv_hal_warpAffineBlocklineNN cv::cv_hal_rvv::warpAffineBlocklineNN
#undef cv_hal_warpAffineBlockline
#define cv_hal_warpAffineBlockline cv::cv_hal_rvv::warpAffineBlockline

static int warpAffine(int src_type,
                const uchar * src_data, size_t src_step, int src_width, int src_height,
                uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
                const double M[6], int interpolation, int borderType, const double borderValue[4]) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}   

static int warpAffineBlocklineNN(int *adelta, int *bdelta, short* xy, int X0, int Y0, int bw) {
    constexpr int AB_BITS = MAX(10, static_cast<int>(INTER_BITS));
    int vl, x1 = 0;
    for (; x1 < bw; x1 += vl) {
        // set vl
        vl = __riscv_vsetvl_e32m8(bw - x1);
        // load i32 for adelta
        auto v_adelta = __riscv_vle32_v_i32m8(adelta + x1, vl);
        auto v_bdelta = __riscv_vle32_v_i32m8(bdelta + x1, vl);
        // add i32
        v_adelta = __riscv_vadd_vx_i32m8(v_adelta, X0, vl);
        v_bdelta = __riscv_vadd_vx_i32m8(v_bdelta, Y0, vl);
        // shr i32
        v_adelta = __riscv_vsra_vx_i32m8(v_adelta, AB_BITS, vl);
        v_bdelta = __riscv_vsra_vx_i32m8(v_bdelta, AB_BITS, vl);
        // compact to i16
        auto v_adelta_i16 = __riscv_vnclip_wx_i16m4(v_adelta, 0, 0, vl);
        auto v_bdelta_i16 = __riscv_vnclip_wx_i16m4(v_bdelta, 0, 0, vl);
        // store ans i16 
        __riscv_vsse16_v_i16m4(xy + x1*2, sizeof(short)*2, v_adelta_i16, vl);
        __riscv_vsse16_v_i16m4(xy + x1*2 + 1, sizeof(short)*2, v_bdelta_i16, vl);
    }
    return CV_HAL_ERROR_OK;
}

static int warpAffineBlockline(int *adelta, int *bdelta, short* xy, short* alpha, int X0, int Y0, int bw) {
    const int AB_BITS = MAX(10, (int)INTER_BITS);
    int vl, x1 = 0;
    for (; x1 < bw; x1 += vl) {
        // set vl
        vl = __riscv_vsetvl_e32m8(bw - x1);
        // load i32 for adelta
        auto v_adelta = __riscv_vle32_v_i32m8(adelta + x1, vl);
        auto v_bdelta = __riscv_vle32_v_i32m8(bdelta + x1, vl);
        // add i32
        v_adelta = __riscv_vadd_vx_i32m8(v_adelta, X0, vl);
        v_bdelta = __riscv_vadd_vx_i32m8(v_bdelta, Y0, vl);
        // shr i32
        v_adelta = __riscv_vsra_vx_i32m8(v_adelta, AB_BITS - INTER_BITS, vl);
        v_bdelta = __riscv_vsra_vx_i32m8(v_bdelta, AB_BITS - INTER_BITS, vl);
        // compact to i16
        auto v_adelta_i16 = __riscv_vnclip_wx_i16m4(__riscv_vsra_vx_i32m8(v_adelta, INTER_BITS, vl), 0, 0, vl);
        auto v_bdelta_i16 = __riscv_vnclip_wx_i16m4(__riscv_vsra_vx_i32m8(v_bdelta, INTER_BITS, vl), 0, 0, vl);
        // store ans i16 
        __riscv_vsse16_v_i16m4(xy + x1*2, sizeof(short)*2, v_adelta_i16, vl);
        __riscv_vsse16_v_i16m4(xy + x1*2 + 1, sizeof(short)*2, v_bdelta_i16, vl);
        
        // alpha
        int mask = INTER_TAB_SIZE - 1;
        v_adelta = __riscv_vand_vx_i32m8(v_adelta, mask, vl);
        v_bdelta = __riscv_vand_vx_i32m8(v_bdelta, mask, vl);
        v_bdelta = __riscv_vsll_vx_i32m8(v_bdelta, INTER_BITS, vl);
        auto v_alpha = __riscv_vor_vv_i32m8(v_adelta, v_bdelta, vl);
        auto v_alpha_i16 = __riscv_vnclip_wx_i16m4(v_alpha, 0, 0, vl);
        __riscv_vsse16_v_i16m4(alpha + x1, sizeof(short), v_alpha_i16, vl);
    }
    return CV_HAL_ERROR_OK;
}
} // cv_hal_rvv::
} // cv::

#endif