// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include <cfloat>
#include <cmath>
#include <typeinfo>

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

// the algorithm is copied from core/src/matrix_decomp.cpp,
// in the function template static int cv::LUImpl
template<typename RVV_T, typename T = typename RVV_T::ElemType>
inline int LU(T* src1, size_t src1_step, int m, T* src2, size_t src2_step, int n, int* info)
{
    T eps;
    if( typeid(T) == typeid(float) )
        eps = FLT_EPSILON*10;
    else if( typeid(T) == typeid(double) )
        eps = DBL_EPSILON*100;
    else
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int i, j, k, p = 1;
    src1_step /= sizeof(src1[0]);
    src2_step /= sizeof(src2[0]);

    int vlmax = RVV_T::setvlmax(), vl;
    if( src2 )
    {
        for( i = 0; i < m; i++ )
        {
            k = i;

            for( j = i+1; j < m; j++ )
                if( std::abs(src1[j*src1_step + i]) > std::abs(src1[k*src1_step + i]) )
                    k = j;

            if( std::abs(src1[k*src1_step + i]) < eps )
            {
                *info = 0;
                return CV_HAL_ERROR_OK;
            }

            if( k != i )
            {
                for( j = i; j < m; j += vl )
                {
                    vl = RVV_T::setvl(m - j);
                    auto vec_src1 = RVV_T::vload(src1 + i * src1_step + j, vl);
                    auto vec_src2 = RVV_T::vload(src1 + k * src1_step + j, vl);
                    RVV_T::vstore(src1 + k * src1_step + j, vec_src1, vl);
                    RVV_T::vstore(src1 + i * src1_step + j, vec_src2, vl);
                }
                for( j = 0; j < n; j += vl )
                {
                    vl = RVV_T::setvl(n - j);
                    auto vec_src1 = RVV_T::vload(src2 + i * src2_step + j, vl);
                    auto vec_src2 = RVV_T::vload(src2 + k * src2_step + j, vl);
                    RVV_T::vstore(src2 + k * src2_step + j, vec_src1, vl);
                    RVV_T::vstore(src2 + i * src2_step + j, vec_src2, vl);
                }
                p = -p;
            }

            T d = -1/src1[i*src1_step + i];

            for( j = i+1; j < m; j++ )
            {
                T alpha = src1[j*src1_step + i]*d;

                for( k = i+1; k < m; k += vl )
                {
                    vl = RVV_T::setvl(m - k);
                    auto vec_src = RVV_T::vload(src1 + i * src1_step + k, vl);
                    auto vec_dst = RVV_T::vload(src1 + j * src1_step + k, vl);
                    vec_dst = __riscv_vfmacc(vec_dst, alpha, vec_src, vl);
                    RVV_T::vstore(src1 + j * src1_step + k, vec_dst, vl);
                }
                for( k = 0; k < n; k += vl )
                {
                    vl = RVV_T::setvl(n - k);
                    auto vec_src = RVV_T::vload(src2 + i * src2_step + k, vl);
                    auto vec_dst = RVV_T::vload(src2 + j * src2_step + k, vl);
                    vec_dst = __riscv_vfmacc(vec_dst, alpha, vec_src, vl);
                    RVV_T::vstore(src2 + j * src2_step + k, vec_dst, vl);
                }
            }
        }

        for( i = m-1; i >= 0; i-- )
            for( j = 0; j < n; j++ )
            {
                T s = src2[i*src2_step + j];
                auto vec_sum = RVV_T::vmv(0, vlmax);
                for( k = i+1; k < m; k += vl )
                {
                    vl = RVV_T::setvl(m - k);
                    auto vec_src1 = RVV_T::vload(src1 + i * src1_step + k, vl);
                    auto vec_src2 = RVV_T::vload_stride(src2 + k * src2_step + j, sizeof(T) * src2_step, vl);
                    vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
                }
                s -= __riscv_vfmv_f(__riscv_vfredosum(vec_sum, RVV_BaseType<RVV_T>::vmv_s(0, vlmax), vlmax));
                src2[i*src2_step + j] = s/src1[i*src1_step + i];
            }
    }
    else
    {
        for( i = 0; i < m; i++ )
        {
            k = i;

            for( j = i+1; j < m; j++ )
                if( std::abs(src1[j*src1_step + i]) > std::abs(src1[k*src1_step + i]) )
                    k = j;

            if( std::abs(src1[k*src1_step + i]) < eps )
            {
                *info = 0;
                return CV_HAL_ERROR_OK;
            }

            if( k != i )
            {
                for( j = i; j < m; j += vl )
                {
                    vl = RVV_T::setvl(m - j);
                    auto vec_src1 = RVV_T::vload(src1 + i * src1_step + j, vl);
                    auto vec_src2 = RVV_T::vload(src1 + k * src1_step + j, vl);
                    RVV_T::vstore(src1 + k * src1_step + j, vec_src1, vl);
                    RVV_T::vstore(src1 + i * src1_step + j, vec_src2, vl);
                }
                p = -p;
            }

            T d = -1/src1[i*src1_step + i];

            for( j = i+1; j < m; j++ )
            {
                T alpha = src1[j*src1_step + i]*d;

                for( k = i+1; k < m; k += vl )
                {
                    vl = RVV_T::setvl(m - k);
                    auto vec_src = RVV_T::vload(src1 + i * src1_step + k, vl);
                    auto vec_dst = RVV_T::vload(src1 + j * src1_step + k, vl);
                    vec_dst = __riscv_vfmacc(vec_dst, alpha, vec_src, vl);
                    RVV_T::vstore(src1 + j * src1_step + k, vec_dst, vl);
                }
            }
        }
    }

    *info = p;
    return CV_HAL_ERROR_OK;
}

} // anonymous

int LU32f(float* src1, size_t src1_step, int m, float* src2, size_t src2_step, int n, int* info) {
    return LU<RVV_F32M4>(src1, src1_step, m, src2, src2_step, n, info);
}
int LU64f(double* src1, size_t src1_step, int m, double* src2, size_t src2_step, int n, int* info) {
    return LU<RVV_F64M4>(src1, src1_step, m, src2, src2_step, n, info);
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::core
