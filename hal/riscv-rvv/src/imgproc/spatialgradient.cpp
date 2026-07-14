// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace imgproc {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

static inline vint16m2_t widen_u8_to_i16(vuint8m1_t v, size_t vl)
{
    vuint16m2_t widened = __riscv_vzext_vf2(v, vl);
    return __riscv_vreinterpret_v_u16m2_i16m2(widened);
}

static inline void spatialGradientKernel(int16_t& dx, int16_t& dy,
                                         int16_t v00, int16_t v01, int16_t v02,
                                         int16_t v10,              int16_t v12,
                                         int16_t v20, int16_t v21, int16_t v22)
{
    int16_t tmp_add = v22 - v00;
    int16_t tmp_sub = v02 - v20;
    int16_t tmp_x = v12 - v10;
    int16_t tmp_y = v21 - v01;

    dx = tmp_add + tmp_sub + tmp_x + tmp_x;
    dy = tmp_add - tmp_sub + tmp_y + tmp_y;
}

static inline void loadRow3(const uint8_t* row, int x, size_t vl,
                            vint16m2_t& m, vint16m2_t& n, vint16m2_t& p)
{
    m = widen_u8_to_i16(__riscv_vle8_v_u8m1(row + x - 1, vl), vl);
    n = widen_u8_to_i16(__riscv_vle8_v_u8m1(row + x,     vl), vl);
    p = widen_u8_to_i16(__riscv_vle8_v_u8m1(row + x + 1, vl), vl);
}

static inline void spatialGradientKernelVec(vint16m2_t& vdx, vint16m2_t& vdy,
                                            const vint16m2_t& v_s1m,
                                            const vint16m2_t& v_s1n,
                                            const vint16m2_t& v_s1p,
                                            const vint16m2_t& v_s2m,
                                            const vint16m2_t& v_s2p,
                                            const vint16m2_t& v_s3m,
                                            const vint16m2_t& v_s3n,
                                            const vint16m2_t& v_s3p,
                                            size_t vl)
{
    // vdx = (v_s3p - v_s1m) + (v_s1p - v_s3m) + 2 * (v_s2p - v_s2m)
    // vdy = (v_s3p - v_s1m) - (v_s1p - v_s3m) + 2 * (v_s3n - v_s1n)
    vint16m2_t tmp_add = __riscv_vsub_vv_i16m2(v_s3p, v_s1m, vl);
    vint16m2_t tmp_sub = __riscv_vsub_vv_i16m2(v_s1p, v_s3m, vl);
    vint16m2_t tmp_x = __riscv_vsub_vv_i16m2(v_s2p, v_s2m, vl);
    vint16m2_t tmp_y = __riscv_vsub_vv_i16m2(v_s3n, v_s1n, vl);

    vdx = __riscv_vadd_vv_i16m2(tmp_add, tmp_sub, vl);
    vdx = __riscv_vadd_vv_i16m2(vdx, tmp_x, vl);
    vdx = __riscv_vadd_vv_i16m2(vdx, tmp_x, vl);

    vdy = __riscv_vsub_vv_i16m2(tmp_add, tmp_sub, vl);
    vdy = __riscv_vadd_vv_i16m2(vdy, tmp_y, vl);
    vdy = __riscv_vadd_vv_i16m2(vdy, tmp_y, vl);
}

static inline void processBorderPixels(const uint8_t* p_src,
                                       const uint8_t* c_src,
                                       const uint8_t* n_src,
                                       int16_t* c_dx,
                                       int16_t* c_dy,
                                       int width,
                                       int j_offl,
                                       int j_offr)
{
    int j = 0;
    int j_p = j + j_offl;
    int j_n = 1;
    if (j_n >= width)
        j_n = j + j_offr;

    int16_t v00 = p_src[j_p], v01 = p_src[j], v02 = p_src[j_n];
    int16_t v10 = c_src[j_p],                 v12 = c_src[j_n];
    int16_t v20 = n_src[j_p], v21 = n_src[j], v22 = n_src[j_n];
    spatialGradientKernel(c_dx[j], c_dy[j], v00, v01, v02, v10, v12, v20, v21, v22);

    if (width > 1)
    {
        j = width - 1;
        j_p = j - 1;
        j_n = j + j_offr;

        v00 = p_src[j_p]; v01 = p_src[j]; v02 = p_src[j_n];
        v10 = c_src[j_p];                 v12 = c_src[j_n];
        v20 = n_src[j_p]; v21 = n_src[j]; v22 = n_src[j_n];
        spatialGradientKernel(c_dx[j], c_dy[j], v00, v01, v02, v10, v12, v20, v21, v22);
    }
}

// Characters in variable names have the following meanings:
// m: offset -1
// n: offset  0
// p: offset  1
static int spatialGradient_row(int start, int end,
                               const uint8_t* src_data, size_t src_step,
                               int16_t*       dx_data,  size_t dx_step,
                               int16_t*       dy_data,  size_t dy_step,
                               int width, int height,
                               int border_type)
{
    int i_top = 0;
    int i_bottom = height - 1;
    int j_offl = 0;
    int j_offr = 0;

    if (border_type == BORDER_DEFAULT)
    {
        if (height > 1)
        {
            i_top = 1;
            i_bottom = height - 2;
        }
        if (width > 1)
        {
            j_offl = 1;
            j_offr = -1;
        }
    }

    int y = start;
    for (; y + 1 < end; y += 2)
    {
        const uint8_t* p_src = src_data + (y == 0 ? i_top : y - 1) * src_step;
        const uint8_t* c_src = src_data + y * src_step;
        const uint8_t* n_src = src_data + (y + 1) * src_step;
        const uint8_t* m_src = src_data + (y == height - 2 ? i_bottom : y + 2) * src_step;

        int16_t* c_dx = reinterpret_cast<int16_t*>(
                            reinterpret_cast<uint8_t*>(dx_data) + y * dx_step);
        int16_t* c_dy = reinterpret_cast<int16_t*>(
                            reinterpret_cast<uint8_t*>(dy_data) + y * dy_step);
        int16_t* n_dx = reinterpret_cast<int16_t*>(
                            reinterpret_cast<uint8_t*>(dx_data) + (y + 1) * dx_step);
        int16_t* n_dy = reinterpret_cast<int16_t*>(
                            reinterpret_cast<uint8_t*>(dy_data) + (y + 1) * dy_step);

        processBorderPixels(p_src, c_src, n_src, c_dx, c_dy, width, j_offl, j_offr);
        processBorderPixels(c_src, n_src, m_src, n_dx, n_dy, width, j_offl, j_offr);

        // Process rest of columns
        int x = 1;
        const int last = width - 1;
        while (x < last)
        {
            size_t vl = __riscv_vsetvl_e8m1((size_t)(last - x));

            vint16m2_t v_s1m, v_s1n, v_s1p;
            vint16m2_t v_s2m, v_s2n, v_s2p;
            vint16m2_t v_s3m, v_s3n, v_s3p;
            loadRow3(p_src, x, vl, v_s1m, v_s1n, v_s1p);
            loadRow3(c_src, x, vl, v_s2m, v_s2n, v_s2p);
            loadRow3(n_src, x, vl, v_s3m, v_s3n, v_s3p);

            vint16m2_t vdx, vdy;
            spatialGradientKernelVec(vdx, vdy,
                                     v_s1m, v_s1n, v_s1p,
                                     v_s2m, v_s2p,
                                     v_s3m, v_s3n, v_s3p, vl);
            __riscv_vse16_v_i16m2(c_dx + x, vdx, vl);
            __riscv_vse16_v_i16m2(c_dy + x, vdy, vl);

            vint16m2_t v_s4m, v_s4n, v_s4p;
            loadRow3(m_src, x, vl, v_s4m, v_s4n, v_s4p);
            spatialGradientKernelVec(vdx, vdy,
                                     v_s2m, v_s2n, v_s2p,
                                     v_s3m, v_s3p,
                                     v_s4m, v_s4n, v_s4p, vl);
            __riscv_vse16_v_i16m2(n_dx + x, vdx, vl);
            __riscv_vse16_v_i16m2(n_dy + x, vdy, vl);

            x += (int)vl;
        }
    }

    for (; y < end; ++y)
    {
        const uint8_t* p_src = src_data + (y == 0 ? i_top : y - 1) * src_step;
        const uint8_t* c_src = src_data + y * src_step;
        const uint8_t* n_src = src_data + (y == height - 1 ? i_bottom : y + 1) * src_step;

        int16_t* c_dx = reinterpret_cast<int16_t*>(
                            reinterpret_cast<uint8_t*>(dx_data) + y * dx_step);
        int16_t* c_dy = reinterpret_cast<int16_t*>(
                            reinterpret_cast<uint8_t*>(dy_data) + y * dy_step);

        processBorderPixels(p_src, c_src, n_src, c_dx, c_dy, width, j_offl, j_offr);

        int x = 1;
        const int last = width - 1;
        while (x < last)
        {
            size_t vl = __riscv_vsetvl_e8m1((size_t)(last - x));
            vint16m2_t v_s1m, v_s1n, v_s1p;
            vint16m2_t v_s2m, v_s2p;
            vint16m2_t v_s3m, v_s3n, v_s3p;
            loadRow3(p_src, x, vl, v_s1m, v_s1n, v_s1p);
            v_s2m = widen_u8_to_i16(__riscv_vle8_v_u8m1(c_src + x - 1, vl), vl);
            v_s2p = widen_u8_to_i16(__riscv_vle8_v_u8m1(c_src + x + 1, vl), vl);
            loadRow3(n_src, x, vl, v_s3m, v_s3n, v_s3p);

            vint16m2_t vdx, vdy;
            spatialGradientKernelVec(vdx, vdy,
                                     v_s1m, v_s1n, v_s1p,
                                     v_s2m, v_s2p,
                                     v_s3m, v_s3n, v_s3p, vl);
            __riscv_vse16_v_i16m2(c_dx + x, vdx, vl);
            __riscv_vse16_v_i16m2(c_dy + x, vdy, vl);

            x += (int)vl;
        }
    }

    return CV_HAL_ERROR_OK;
}

} // anonymous namespace

int spatialGradient(const uint8_t* src_data, size_t src_step,
                    int16_t*       dx_data,  size_t dx_step,
                    int16_t*       dy_data,  size_t dy_step,
                    int width, int height,
                    int ksize, int border_type)
{
    if (ksize != 3)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (border_type != BORDER_REPLICATE && border_type != BORDER_DEFAULT)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (width < 1 || height < 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    return common::invoke(height, {spatialGradient_row},
                          src_data, src_step,
                          dx_data,  dx_step,
                          dy_data,  dy_step,
                          width, height, border_type);
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::imgproc
