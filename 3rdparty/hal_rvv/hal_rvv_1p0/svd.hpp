// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_SVD_HPP_INCLUDED
#define OPENCV_HAL_RVV_SVD_HPP_INCLUDED

#include <cfloat>
#include <cmath>
#include <typeinfo>
#include <vector>
#include <riscv_vector.h>
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv { namespace svd {

#undef cv_hal_SVD32f
#define cv_hal_SVD32f cv::cv_hal_rvv::svd::SVD<cv::cv_hal_rvv::RVV_F32M4>
#undef cv_hal_SVD64f
#define cv_hal_SVD64f cv::cv_hal_rvv::svd::SVD<cv::cv_hal_rvv::RVV_F64M4>

// the algorithm is copied from core/src/lapack.cpp,
// in the function template static void cv::JacobiSVDImpl_
template<typename RVV_T, typename T = typename RVV_T::ElemType>
inline int SVD(T* src, size_t src_step, T* w, T*, size_t, T* vt, size_t vt_step, int m, int n, int flags)
{
    T minval, eps;
    if( typeid(T) == typeid(float) )
    {
        minval = FLT_MIN;
        eps = FLT_EPSILON*2;
    }
    else if( typeid(T) == typeid(double) )
    {
        minval = DBL_MIN;
        eps = DBL_EPSILON*10;
    }
    else
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int n1;
    if( flags == CV_HAL_SVD_NO_UV )
        n1 = 0;
    else if( flags == (CV_HAL_SVD_SHORT_UV | CV_HAL_SVD_MODIFY_A) )
        n1 = n;
    else if( flags == (CV_HAL_SVD_FULL_UV | CV_HAL_SVD_MODIFY_A) )
        n1 = m;
    else
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    std::vector<double> Wbuf(n);
    double* W = Wbuf.data();
    int i, j, k, iter, max_iter = std::max(m, 30);
    T c, s;
    double sd;
    src_step /= sizeof(src[0]);
    vt_step /= sizeof(vt[0]);

    int vlmax = RVV_T::setvlmax(), vl;
    for( i = 0; i < n; i++ )
    {
        for( k = 0, sd = 0; k < m; k++ )
        {
            T t = src[i*src_step + k];
            sd += (double)t*t;
        }
        W[i] = sd;

        if( vt )
        {
            for( k = 0; k < n; k++ )
                vt[i*vt_step + k] = 0;
            vt[i*vt_step + i] = 1;
        }
    }

    for( iter = 0; iter < max_iter; iter++ )
    {
        bool changed = false;

        for( i = 0; i < n-1; i++ )
            for( j = i+1; j < n; j++ )
            {
                T *Ai = src + i*src_step, *Aj = src + j*src_step;
                double a = W[i], p = 0, b = W[j];

                auto vec_sum1 = RVV_T::vmv(0, vlmax);
                for( k = 0; k < m; k += vl )
                {
                    vl = RVV_T::setvl(m - k);
                    auto vec_src1 = RVV_T::vload(Ai + k, vl);
                    auto vec_src2 = RVV_T::vload(Aj + k, vl);
                    vec_sum1 = __riscv_vfmacc_tu(vec_sum1, vec_src1, vec_src2, vl);
                }
                p = __riscv_vfmv_f(__riscv_vfredosum(vec_sum1, RVV_BaseType<RVV_T>::vmv_s(0, vlmax), vlmax));

                if( std::abs(p) <= eps*std::sqrt((double)a*b) )
                    continue;

                p *= 2;
                double beta = a - b, gamma = hypot((double)p, beta);
                if( beta < 0 )
                {
                    double delta = (gamma - beta)*0.5;
                    s = (T)std::sqrt(delta/gamma);
                    c = (T)(p/(gamma*s*2));
                }
                else
                {
                    c = (T)std::sqrt((gamma + beta)/(gamma*2));
                    s = (T)(p/(gamma*c*2));
                }

                vec_sum1 = RVV_T::vmv(0, vlmax);
                auto vec_sum2 = RVV_T::vmv(0, vlmax);
                for( k = 0; k < m; k += vl )
                {
                    vl = RVV_T::setvl(m - k);
                    auto vec_src1 = RVV_T::vload(Ai + k, vl);
                    auto vec_src2 = RVV_T::vload(Aj + k, vl);
                    auto vec_t0 = __riscv_vfadd(__riscv_vfmul(vec_src1, c, vl), __riscv_vfmul(vec_src2, s, vl), vl);
                    auto vec_t1 = __riscv_vfsub(__riscv_vfmul(vec_src2, c, vl), __riscv_vfmul(vec_src1, s, vl), vl);
                    RVV_T::vstore(Ai + k, vec_t0, vl);
                    RVV_T::vstore(Aj + k, vec_t1, vl);
                    vec_sum1 = __riscv_vfmacc_tu(vec_sum1, vec_t0, vec_t0, vl);
                    vec_sum2 = __riscv_vfmacc_tu(vec_sum2, vec_t1, vec_t1, vl);
                }
                W[i] = __riscv_vfmv_f(__riscv_vfredosum(vec_sum1, RVV_BaseType<RVV_T>::vmv_s(0, vlmax), vlmax));
                W[j] = __riscv_vfmv_f(__riscv_vfredosum(vec_sum2, RVV_BaseType<RVV_T>::vmv_s(0, vlmax), vlmax));

                changed = true;

                if( vt )
                {
                    T *Vi = vt + i*vt_step, *Vj = vt + j*vt_step;
                    for( k = 0; k < n; k += vl )
                    {
                        vl = RVV_T::setvl(n - k);
                        auto vec_src1 = RVV_T::vload(Vi + k, vl);
                        auto vec_src2 = RVV_T::vload(Vj + k, vl);
                        auto vec_t0 = __riscv_vfadd(__riscv_vfmul(vec_src1, c, vl), __riscv_vfmul(vec_src2, s, vl), vl);
                        auto vec_t1 = __riscv_vfsub(__riscv_vfmul(vec_src2, c, vl), __riscv_vfmul(vec_src1, s, vl), vl);
                        RVV_T::vstore(Vi + k, vec_t0, vl);
                        RVV_T::vstore(Vj + k, vec_t1, vl);
                    }
                }
            }
        if( !changed )
            break;
    }

    for( i = 0; i < n; i++ )
    {
        auto vec_sum = RVV_T::vmv(0, vlmax);
        for( k = 0; k < m; k += vl )
        {
            vl = RVV_T::setvl(m - k);
            auto vec_src = RVV_T::vload(src + i * src_step + k, vl);
            vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src, vec_src, vl);
        }
        W[i] = std::sqrt(__riscv_vfmv_f(__riscv_vfredosum(vec_sum, RVV_BaseType<RVV_T>::vmv_s(0, vlmax), vlmax)));
    }

    for( i = 0; i < n-1; i++ )
    {
        j = i;
        for( k = i+1; k < n; k++ )
        {
            if( W[j] < W[k] )
                j = k;
        }
        if( i != j )
        {
            std::swap(W[i], W[j]);
            if( vt )
            {
                for( k = 0; k < m; k++ )
                    std::swap(src[i*src_step + k], src[j*src_step + k]);

                for( k = 0; k < n; k++ )
                    std::swap(vt[i*vt_step + k], vt[j*vt_step + k]);
            }
        }
    }

    for( i = 0; i < n; i++ )
        w[i] = (T)W[i];

    if( !vt )
        return CV_HAL_ERROR_OK;

    uint64 rng = 0x12345678;
    auto next = [&rng]{ return (unsigned)(rng = (uint64)(unsigned)rng * 4164903690U + (unsigned)(rng >> 32)); };
    for( i = 0; i < n1; i++ )
    {
        sd = i < n ? W[i] : 0;

        for( int ii = 0; ii < 100 && sd <= minval; ii++ )
        {
            // if we got a zero singular value, then in order to get the corresponding left singular vector
            // we generate a random vector, project it to the previously computed left singular vectors,
            // subtract the projection and normalize the difference.
            const T val0 = (T)(1./m);
            for( k = 0; k < m; k++ )
            {
                T val = (next() & 256) != 0 ? val0 : -val0;
                src[i*src_step + k] = val;
            }
            for( iter = 0; iter < 2; iter++ )
            {
                for( j = 0; j < i; j++ )
                {
                    auto vec_sum = RVV_T::vmv(0, vlmax);
                    for( k = 0; k < m; k += vl )
                    {
                        vl = RVV_T::setvl(m - k);
                        auto vec_src1 = RVV_T::vload(src + i * src_step + k, vl);
                        auto vec_src2 = RVV_T::vload(src + j * src_step + k, vl);
                        vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
                    }
                    sd = __riscv_vfmv_f(__riscv_vfredosum(vec_sum, RVV_BaseType<RVV_T>::vmv_s(0, vlmax), vlmax));

                    vec_sum = RVV_T::vmv(0, vlmax);
                    for( k = 0; k < m; k += vl )
                    {
                        vl = RVV_T::setvl(m - k);
                        auto vec_src1 = RVV_T::vload(src + i * src_step + k, vl);
                        auto vec_src2 = RVV_T::vload(src + j * src_step + k, vl);
                        vec_src1 = __riscv_vfnmsac(vec_src1, sd, vec_src2, vl);
                        RVV_T::vstore(src + i * src_step + k, vec_src1, vl);
                        vec_sum = __riscv_vfadd_tu(vec_sum, vec_sum, __riscv_vfabs(vec_src1, vl), vl);
                    }
                    T asum = __riscv_vfmv_f(__riscv_vfredosum(vec_sum, RVV_BaseType<RVV_T>::vmv_s(0, vlmax), vlmax));
                    asum = asum > eps*100 ? 1/asum : 0;
                    for( k = 0; k < m; k += vl )
                    {
                        vl = RVV_T::setvl(m - k);
                        auto vec_src = RVV_T::vload(src + i * src_step + k, vl);
                        vec_src = __riscv_vfmul(vec_src, asum, vl);
                        RVV_T::vstore(src + i * src_step + k, vec_src, vl);
                    }
                }
            }
            
            auto vec_sum = RVV_T::vmv(0, vlmax);
            for( k = 0; k < m; k += vl )
            {
                vl = RVV_T::setvl(m - k);
                auto vec_src1 = RVV_T::vload(src + i * src_step + k, vl);
                auto vec_src2 = RVV_T::vload(src + j * src_step + k, vl);
                vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
            }
            sd = std::sqrt(__riscv_vfmv_f(__riscv_vfredosum(vec_sum, RVV_BaseType<RVV_T>::vmv_s(0, vlmax), vlmax)));
        }

        s = (T)(sd > minval ? 1/sd : 0.);
        for( k = 0; k < m; k += vl )
        {
            vl = RVV_T::setvl(m - k);
            auto vec_src = RVV_T::vload(src + i * src_step + k, vl);
            vec_src = __riscv_vfmul(vec_src, s, vl);
            RVV_T::vstore(src + i * src_step + k, vec_src, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

}}}

#endif
