// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_CHOLESKY_HPP_INCLUDED
#define OPENCV_HAL_RVV_CHOLESKY_HPP_INCLUDED

#include <cmath>
#include <limits>
#include <riscv_vector.h>
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv { namespace cholesky {

#undef cv_hal_Cholesky32f
#define cv_hal_Cholesky32f cv::cv_hal_rvv::cholesky::Cholesky<cv::cv_hal_rvv::RVV_F32M4>
#undef cv_hal_Cholesky64f
#define cv_hal_Cholesky64f cv::cv_hal_rvv::cholesky::Cholesky<cv::cv_hal_rvv::RVV_F64M4>

// the algorithm is copied from core/src/matrix_decomp.cpp,
// in the function template static int cv::CholImpl
template <typename RVV_T, typename T = typename RVV_T::ElemType>
inline int Cholesky(T* src1, size_t src1_step, int m, T* src2, size_t src2_step, int n, bool* info)
{
    int i, j, k;
    double s;
    src1_step /= sizeof(src1[0]);
    src2_step /= sizeof(src2[0]);

    int vlmax = RVV_T::setvlmax(), vl;
    for( i = 0; i < m; i++ )
    {
        for( j = 0; j < i; j++ )
        {
            auto vec_sum = RVV_T::vmv(0, vlmax);
            for( k = 0; k < j; k += vl )
            {
                vl = RVV_T::setvl(j - k);
                auto vec_src1 = RVV_T::vload(src1 + i * src1_step + k, vl);
                auto vec_src2 = RVV_T::vload(src1 + j * src1_step + k, vl);
                vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
            }
            s = src1[i*src1_step + j] - __riscv_vfmv_f(__riscv_vfredosum(vec_sum, RVV<T, LMUL_1>::vmv_s(0, vlmax), vlmax));
            src1[i*src1_step + j] = (T)(s*src1[j*src1_step + j]);
        }
        auto vec_sum = RVV_T::vmv(0, vlmax);
        for( k = 0; k < j; k += vl )
        {
            vl = RVV_T::setvl(j - k);
            auto vec_src = RVV_T::vload(src1 + i * src1_step + k, vl);
            vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src, vec_src, vl);
        }
        s = src1[i*src1_step + i] - __riscv_vfmv_f(__riscv_vfredosum(vec_sum, RVV<T, LMUL_1>::vmv_s(0, vlmax), vlmax));
        if( s < std::numeric_limits<T>::epsilon() )
        {
            *info = false;
            return CV_HAL_ERROR_OK;
        }
        src1[i*src1_step + i] = (T)(1./std::sqrt(s));
    }

    if (!src2)
    {
        for( i = 0; i < m; i += vl )
        {
            vl = RVV_T::setvl(m - i);
            auto vec_src = RVV_T::vload_stride(src1 + i * src1_step + i, sizeof(T) * (src1_step + 1), vl);
            vec_src = __riscv_vfrdiv(vec_src, 1, vl);
            RVV_T::vstore_stride(src1 + i * src1_step + i, sizeof(T) * (src1_step + 1), vec_src, vl);
        }
        *info = true;
        return CV_HAL_ERROR_OK;
    }

    for( i = 0; i < m; i++ )
    {
        for( j = 0; j < n; j++ )
        {
            auto vec_sum = RVV_T::vmv(0, vlmax);
            for( k = 0; k < i; k += vl )
            {
                vl = RVV_T::setvl(i - k);
                auto vec_src1 = RVV_T::vload(src1 + i * src1_step + k, vl);
                auto vec_src2 = RVV_T::vload_stride(src2 + k * src2_step + j, sizeof(T) * src2_step, vl);
                vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
            }
            s = src2[i*src2_step + j] - __riscv_vfmv_f(__riscv_vfredosum(vec_sum, RVV<T, LMUL_1>::vmv_s(0, vlmax), vlmax));
            src2[i*src2_step + j] = (T)(s*src1[i*src1_step + i]);
        }
    }

    for( i = m-1; i >= 0; i-- )
    {
        for( j = 0; j < n; j++ )
        {
            auto vec_sum = RVV_T::vmv(0, vlmax);
            for( k = i + 1; k < m; k += vl )
            {
                vl = RVV_T::setvl(m - k);
                auto vec_src1 = RVV_T::vload_stride(src1 + k * src1_step + i, sizeof(T) * src1_step, vl);
                auto vec_src2 = RVV_T::vload_stride(src2 + k * src2_step + j, sizeof(T) * src2_step, vl);
                vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
            }
            s = src2[i*src2_step + j] - __riscv_vfmv_f(__riscv_vfredosum(vec_sum, RVV<T, LMUL_1>::vmv_s(0, vlmax), vlmax));
            src2[i*src2_step + j] = (T)(s*src1[i*src1_step + i]);
        }
    }
    for( i = 0; i < m; i += vl )
    {
        vl = RVV_T::setvl(m - i);
        auto vec_src = RVV_T::vload_stride(src1 + i * src1_step + i, sizeof(T) * (src1_step + 1), vl);
        vec_src = __riscv_vfrdiv(vec_src, 1, vl);
        RVV_T::vstore_stride(src1 + i * src1_step + i, sizeof(T) * (src1_step + 1), vec_src, vl);
    }

    *info = true;
    return CV_HAL_ERROR_OK;
}

}}}

#endif
