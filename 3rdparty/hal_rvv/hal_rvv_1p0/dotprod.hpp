// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Third party copyrights are property of their respective owners.


#ifndef OPENCV_HAL_RVV_DOTPROD_HPP_INCLUDED
#define OPENCV_HAL_RVV_DOTPROD_HPP_INCLUDED

#include <riscv_vector.h>
#include <algorithm>

namespace cv { namespace cv_hal_rvv { namespace dotprod {

#undef cv_hal_dotProduct
#define cv_hal_dotProduct cv::cv_hal_rvv::dotprod::dotprod

namespace {

double dotProd_8u(const uchar *a, const uchar *b, int len) {
    constexpr int block_size0 = (1 << 15);

    double r = 0;
    int i = 0;
    while (i < len) {
        int block_size = std::min(block_size0, len - i);

        vuint32m1_t s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        int vl;
        for (int j = 0; j < block_size; j += vl) {
            vl = __riscv_vsetvl_e8m4(block_size - j);

            auto va = __riscv_vle8_v_u8m4(a + j, vl);
            auto vb = __riscv_vle8_v_u8m4(b + j, vl);

            s = __riscv_vwredsumu(__riscv_vwmulu(va, vb, vl), s, vl);
        }
        r += (double)__riscv_vmv_x(s);

        i += block_size;
        a += block_size;
        b += block_size;
    }

    return r;
}

double dotProd_8s(const schar *a, const schar *b, int len) {
    constexpr int block_size0 = (1 << 14);

    double r = 0;
    int i = 0;
    while (i < len) {
        int block_size = std::min(block_size0, len - i);

        vint32m1_t s = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
        int vl;
        for (int j = 0; j < block_size; j += vl) {
            vl = __riscv_vsetvl_e8m4(block_size - j);

            auto va = __riscv_vle8_v_i8m4(a + j, vl);
            auto vb = __riscv_vle8_v_i8m4(b + j, vl);

            s = __riscv_vwredsum(__riscv_vwmul(va, vb, vl), s, vl);
        }
        r += (double)__riscv_vmv_x(s);

        i += block_size;
        a += block_size;
        b += block_size;
    }

    return r;
}

} // anonymous

using DotProdFunc = double (*)(const uchar *a, const uchar *b, int len);
inline int dotprod(const uchar *a_data, size_t a_step, const uchar *b_data, size_t b_step,
                   int width, int height, int type, double *dot_val) {
    int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    static DotProdFunc dotprod_tab[CV_DEPTH_MAX] = {
        (DotProdFunc)dotProd_8u,  (DotProdFunc)dotProd_8s,
        nullptr, nullptr,
        nullptr, nullptr,
        nullptr, nullptr
    };
    DotProdFunc func = dotprod_tab[depth];
    if (func == nullptr) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    static const size_t elem_size_tab[CV_DEPTH_MAX] = {
        sizeof(uchar),   sizeof(schar),
        sizeof(ushort),  sizeof(short),
        sizeof(int),     sizeof(float),
        sizeof(int64_t), 0,
    };
    CV_Assert(elem_size_tab[depth]);

    bool a_continuous = (a_step == width * elem_size_tab[depth] * cn);
    bool b_continuous = (b_step == width * elem_size_tab[depth] * cn);
    size_t nplanes = 1;
    size_t len = width * height;
    if (!a_continuous || !b_continuous) {
        nplanes = height;
        len = width;
    }
    len *= cn;

    double r = 0;
    auto _a = a_data;
    auto _b = b_data;
    for (size_t i = 0; i < nplanes; i++) {
        if (!a_continuous || !b_continuous) {
            _a = a_data + a_step * i;
            _b = b_data + b_step * i;
        }
        r += func(_a, _b, len);
    }
    *dot_val = r;

    return CV_HAL_ERROR_OK;
}

}}} // cv::cv_hal_rvv::dotprod

#endif // OPENCV_HAL_RVV_DOTPROD_HPP_INCLUDED

