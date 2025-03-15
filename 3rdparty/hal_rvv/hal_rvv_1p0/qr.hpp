// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_QR_HPP_INCLUDED
#define OPENCV_HAL_RVV_QR_HPP_INCLUDED

#include <cfloat>
#include <cmath>
#include <typeinfo>
#include <vector>
#include <riscv_vector.h>
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv { namespace qr {

#undef cv_hal_QR32f
#define cv_hal_QR32f cv::cv_hal_rvv::qr::QR<cv::cv_hal_rvv::RVV_F32M4>
#undef cv_hal_QR64f
#define cv_hal_QR64f cv::cv_hal_rvv::qr::QR<cv::cv_hal_rvv::RVV_F64M4>

// the algorithm is copied from core/src/matrix_decomp.cpp,
// in the function template static int cv::QRImpl
template<typename RVV_T, typename T = typename RVV_T::ElemType>
inline int QR(T* src1, size_t src1_step, int m, int n, int k, T* src2, size_t src2_step, T* dst, int* info)
{
    T eps;
    if (typeid(T) == typeid(float))
        eps = FLT_EPSILON*10;
    else if (typeid(T) == typeid(double))
        eps = DBL_EPSILON*400;
    else
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    src1_step /= sizeof(T);
    src2_step /= sizeof(T);

    size_t buf_size = m ? m + n : dst != NULL;
    std::vector<T> buffer(buf_size);
    T* val = buffer.data();
    if (dst == NULL)
        dst = val + m;

    int vlmax = RVV_T::setvlmax(), vl;
    for (int l = 0; l < n; l++)
    {
        //generate val
        int vlSize = m - l;
        auto vec_sum = RVV_T::vmv(0, vlmax);
        for (int i = 0; i < vlSize; i += vl)
        {
            vl = RVV_T::setvl(vlSize - i);
            auto vec_src = RVV_T::vload_stride(src1 + (l + i) * src1_step + l, sizeof(T) * src1_step, vl);
            RVV_T::vstore(val + i, vec_src, vl);
            vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src, vec_src, vl);
        }
        T vlNorm = __riscv_vfmv_f(__riscv_vfredosum(vec_sum, RVV_BaseType<RVV_T>::vmv_s(0, vlmax), vlmax));
        T tmpV = val[0];
        val[0] = val[0] + (val[0] >= 0 ? 1 : -1) * std::sqrt(vlNorm);
        vlNorm = std::sqrt(vlNorm + val[0] * val[0] - tmpV*tmpV);
        for (int i = 0; i < vlSize; i += vl)
        {
            vl = RVV_T::setvl(vlSize - i);
            auto vec_src = RVV_T::vload(val + i, vl);
            vec_src = __riscv_vfdiv(vec_src, vlNorm, vl);
            RVV_T::vstore(val + i, vec_src, vl);
        }
        //multiply A_l*val
        for (int j = l; j < n; j++)
        {
            vec_sum = RVV_T::vmv(0, vlmax);
            for (int i = l; i < m; i += vl)
            {
                vl = RVV_T::setvl(m - i);
                auto vec_src1 = RVV_T::vload(val + i - l, vl);
                auto vec_src2 = RVV_T::vload_stride(src1 + i * src1_step + j, sizeof(T) * src1_step, vl);
                vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
            }
            T v_lA = 2 * __riscv_vfmv_f(__riscv_vfredosum(vec_sum, RVV_BaseType<RVV_T>::vmv_s(0, vlmax), vlmax));

            for (int i = l; i < m; i += vl)
            {
                vl = RVV_T::setvl(m - i);
                auto vec_src1 = RVV_T::vload(val + i - l, vl);
                auto vec_src2 = RVV_T::vload_stride(src1 + i * src1_step + j, sizeof(T) * src1_step, vl);
                vec_src2 = __riscv_vfnmsac(vec_src2, v_lA, vec_src1, vl);
                RVV_T::vstore_stride(src1 + i * src1_step + j, sizeof(T) * src1_step, vec_src2, vl);
            }
        }

        //save val and factors
        dst[l] = val[0] * val[0];
        for (int i = 1; i < vlSize; i += vl)
        {
            vl = RVV_T::setvl(vlSize - i);
            auto vec_src = RVV_T::vload(val + i, vl);
            vec_src = __riscv_vfdiv(vec_src, val[0], vl);
            RVV_T::vstore_stride(src1 + (l + i) * src1_step + l, sizeof(T) * src1_step, vec_src, vl);
        }
    }

    if (src2)
    {
        //generate new rhs
        for (int l = 0; l < n; l++)
        {
            //unpack val
            val[0] = (T)1;
            for (int j = 1; j < m - l; j += vl)
            {
                vl = RVV_T::setvl(m - l - j);
                auto vec_src = RVV_T::vload_stride(src1 + (j + l) * src1_step + l, sizeof(T) * src1_step, vl);
                RVV_T::vstore(val + j, vec_src, vl);
            }

            //h_l*x
            for (int j = 0; j < k; j++)
            {
                auto vec_sum = RVV_T::vmv(0, vlmax);
                for (int i = l; i < m; i += vl)
                {
                    vl = RVV_T::setvl(m - i);
                    auto vec_src1 = RVV_T::vload(val + i - l, vl);
                    auto vec_src2 = RVV_T::vload_stride(src2 + i * src2_step + j, sizeof(T) * src2_step, vl);
                    vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
                }
                T v_lB = 2 * dst[l] * __riscv_vfmv_f(__riscv_vfredosum(vec_sum, RVV_BaseType<RVV_T>::vmv_s(0, vlmax), vlmax));

                for (int i = l; i < m; i += vl)
                {
                    vl = RVV_T::setvl(m - i);
                    auto vec_src1 = RVV_T::vload(val + i - l, vl);
                    auto vec_src2 = RVV_T::vload_stride(src2 + i * src2_step + j, sizeof(T) * src2_step, vl);
                    vec_src2 = __riscv_vfnmsac(vec_src2, v_lB, vec_src1, vl);
                    RVV_T::vstore_stride(src2 + i * src2_step + j, sizeof(T) * src2_step, vec_src2, vl);
                }
            }
        }
        //do back substitution
        for (int i = n - 1; i >= 0; i--)
        {
            for (int j = n - 1; j > i; j--)
            {
                for (int p = 0; p < k; p += vl)
                {
                    vl = RVV_T::setvl(k - p);
                    auto vec_src1 = RVV_T::vload(src2 + i * src2_step + p, vl);
                    auto vec_src2 = RVV_T::vload(src2 + j * src2_step + p, vl);
                    vec_src1 = __riscv_vfnmsac(vec_src1, src1[i*src1_step + j], vec_src2, vl);
                    RVV_T::vstore(src2 + i * src2_step + p, vec_src1, vl);
                }
            }
            if (std::abs(src1[i*src1_step + i]) < eps)
            {
                *info = 0;
                return CV_HAL_ERROR_OK;
            }
            for (int p = 0; p < k; p += vl)
            {
                vl = RVV_T::setvl(k - p);
                auto vec_src = RVV_T::vload(src2 + i * src2_step + p, vl);
                vec_src = __riscv_vfdiv(vec_src, src1[i*src1_step + i], vl);
                RVV_T::vstore(src2 + i * src2_step + p, vec_src, vl);
            }
        }
    }

    *info = 1;
    return CV_HAL_ERROR_OK;
}

}}}

#endif
