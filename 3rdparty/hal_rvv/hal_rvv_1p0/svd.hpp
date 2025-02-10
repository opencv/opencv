// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_SVD_HPP_INCLUDED
#define OPENCV_HAL_RVV_SVD_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_SVD32f
#define cv_hal_SVD32f cv::cv_hal_rvv::SVD
#undef cv_hal_SVD64f
#define cv_hal_SVD64f cv::cv_hal_rvv::SVD

namespace svd
{
    template<typename T> struct rvv;

    template<> struct rvv<float>
    {
        static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m4(); }
        static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m4(a); }
        static inline vfloat32m4_t vfmv_v_f(float a, size_t b) { return __riscv_vfmv_v_f_f32m4(a, b); }
        static inline vfloat32m1_t vfmv_s_f(float a, size_t b) { return __riscv_vfmv_s_f_f32m1(a, b); }
        static inline vfloat32m4_t vle(const float* a, size_t b) { return __riscv_vle32_v_f32m4(a, b); }
        static inline void vse(float* a, vfloat32m4_t b, size_t c) { __riscv_vse32(a, b, c); }
    };

    template<> struct rvv<double>
    {
        static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e64m4(); }
        static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e64m4(a); }
        static inline vfloat64m4_t vfmv_v_f(double a, size_t b) { return __riscv_vfmv_v_f_f64m4(a, b); }
        static inline vfloat64m1_t vfmv_s_f(double a, size_t b) { return __riscv_vfmv_s_f_f64m1(a, b); }
        static inline vfloat64m4_t vle(const double* a, size_t b) { return __riscv_vle64_v_f64m4(a, b); }
        static inline void vse(double* a, vfloat64m4_t b, size_t c) { __riscv_vse64(a, b, c); }
    };
}

// the algorithm is copied from core/src/lapack.cpp,
// in the function template static void cv::JacobiSVDImpl_
template<typename T>
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

    AutoBuffer<double> Wbuf(n);
    double* W = Wbuf.data();
    int i, j, k, iter, max_iter = std::max(m, 30);
    T c, s;
    double sd;
    src_step /= sizeof(src[0]);
    vt_step /= sizeof(vt[0]);

    int vlmax = svd::rvv<T>::vsetvlmax(), vl;
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

                auto vec_sum1 = svd::rvv<T>::vfmv_v_f(0, vlmax);
                for( k = 0; k < m; k += vl )
                {
                    vl = svd::rvv<T>::vsetvl(m - k);
                    auto vec_src1 = svd::rvv<T>::vle(Ai + k, vl);
                    auto vec_src2 = svd::rvv<T>::vle(Aj + k, vl);
                    vec_sum1 = __riscv_vfmacc_tu(vec_sum1, vec_src1, vec_src2, vl);
                }
                p = __riscv_vfmv_f(__riscv_vfredosum(vec_sum1, svd::rvv<T>::vfmv_s_f(0, vlmax), vlmax));

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

                vec_sum1 = svd::rvv<T>::vfmv_v_f(0, vlmax);
                auto vec_sum2 = svd::rvv<T>::vfmv_v_f(0, vlmax);
                for( k = 0; k < m; k += vl )
                {
                    vl = svd::rvv<T>::vsetvl(m - k);
                    auto vec_src1 = svd::rvv<T>::vle(Ai + k, vl);
                    auto vec_src2 = svd::rvv<T>::vle(Aj + k, vl);
                    auto vec_t0 = __riscv_vfadd(__riscv_vfmul(vec_src1, c, vl), __riscv_vfmul(vec_src2, s, vl), vl);
                    auto vec_t1 = __riscv_vfsub(__riscv_vfmul(vec_src2, c, vl), __riscv_vfmul(vec_src1, s, vl), vl);
                    svd::rvv<T>::vse(Ai + k, vec_t0, vl);
                    svd::rvv<T>::vse(Aj + k, vec_t1, vl);
                    vec_sum1 = __riscv_vfmacc_tu(vec_sum1, vec_t0, vec_t0, vl);
                    vec_sum2 = __riscv_vfmacc_tu(vec_sum2, vec_t1, vec_t1, vl);
                }
                W[i] = __riscv_vfmv_f(__riscv_vfredosum(vec_sum1, svd::rvv<T>::vfmv_s_f(0, vlmax), vlmax));
                W[j] = __riscv_vfmv_f(__riscv_vfredosum(vec_sum2, svd::rvv<T>::vfmv_s_f(0, vlmax), vlmax));

                changed = true;

                if( vt )
                {
                    T *Vi = vt + i*vt_step, *Vj = vt + j*vt_step;
                    for( k = 0; k < n; k += vl )
                    {
                        vl = svd::rvv<T>::vsetvl(n - k);
                        auto vec_src1 = svd::rvv<T>::vle(Vi + k, vl);
                        auto vec_src2 = svd::rvv<T>::vle(Vj + k, vl);
                        auto vec_t0 = __riscv_vfadd(__riscv_vfmul(vec_src1, c, vl), __riscv_vfmul(vec_src2, s, vl), vl);
                        auto vec_t1 = __riscv_vfsub(__riscv_vfmul(vec_src2, c, vl), __riscv_vfmul(vec_src1, s, vl), vl);
                        svd::rvv<T>::vse(Vi + k, vec_t0, vl);
                        svd::rvv<T>::vse(Vj + k, vec_t1, vl);
                    }
                }
            }
        if( !changed )
            break;
    }

    for( i = 0; i < n; i++ )
    {
        auto vec_sum = svd::rvv<T>::vfmv_v_f(0, vlmax);
        for( k = 0; k < m; k += vl )
        {
            vl = svd::rvv<T>::vsetvl(m - k);
            auto vec_src = svd::rvv<T>::vle(src + i * src_step + k, vl);
            vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src, vec_src, vl);
        }
        W[i] = std::sqrt(__riscv_vfmv_f(__riscv_vfredosum(vec_sum, svd::rvv<T>::vfmv_s_f(0, vlmax), vlmax)));
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

    RNG rng(0x12345678);
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
                T val = (rng.next() & 256) != 0 ? val0 : -val0;
                src[i*src_step + k] = val;
            }
            for( iter = 0; iter < 2; iter++ )
            {
                for( j = 0; j < i; j++ )
                {
                    auto vec_sum = svd::rvv<T>::vfmv_v_f(0, vlmax);
                    for( k = 0; k < m; k += vl )
                    {
                        vl = svd::rvv<T>::vsetvl(m - k);
                        auto vec_src1 = svd::rvv<T>::vle(src + i * src_step + k, vl);
                        auto vec_src2 = svd::rvv<T>::vle(src + j * src_step + k, vl);
                        vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
                    }
                    sd = __riscv_vfmv_f(__riscv_vfredosum(vec_sum, svd::rvv<T>::vfmv_s_f(0, vlmax), vlmax));

                    vec_sum = svd::rvv<T>::vfmv_v_f(0, vlmax);
                    for( k = 0; k < m; k += vl )
                    {
                        vl = svd::rvv<T>::vsetvl(m - k);
                        auto vec_src1 = svd::rvv<T>::vle(src + i * src_step + k, vl);
                        auto vec_src2 = svd::rvv<T>::vle(src + j * src_step + k, vl);
                        vec_src1 = __riscv_vfnmsac(vec_src1, sd, vec_src2, vl);
                        svd::rvv<T>::vse(src + i * src_step + k, vec_src1, vl);
                        vec_sum = __riscv_vfadd_tu(vec_sum, vec_sum, __riscv_vfabs(vec_src1, vl), vl);
                    }
                    T asum = __riscv_vfmv_f(__riscv_vfredosum(vec_sum, svd::rvv<T>::vfmv_s_f(0, vlmax), vlmax));
                    asum = asum > eps*100 ? 1/asum : 0;
                    for( k = 0; k < m; k += vl )
                    {
                        vl = svd::rvv<T>::vsetvl(m - k);
                        auto vec_src = svd::rvv<T>::vle(src + i * src_step + k, vl);
                        vec_src = __riscv_vfmul(vec_src, asum, vl);
                        svd::rvv<T>::vse(src + i * src_step + k, vec_src, vl);
                    }
                }
            }
            
            auto vec_sum = svd::rvv<T>::vfmv_v_f(0, vlmax);
            for( k = 0; k < m; k += vl )
            {
                vl = svd::rvv<T>::vsetvl(m - k);
                auto vec_src1 = svd::rvv<T>::vle(src + i * src_step + k, vl);
                auto vec_src2 = svd::rvv<T>::vle(src + j * src_step + k, vl);
                vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
            }
            sd = std::sqrt(__riscv_vfmv_f(__riscv_vfredosum(vec_sum, svd::rvv<T>::vfmv_s_f(0, vlmax), vlmax)));
        }

        s = (T)(sd > minval ? 1/sd : 0.);
        for( k = 0; k < m; k += vl )
        {
            vl = svd::rvv<T>::vsetvl(m - k);
            auto vec_src = svd::rvv<T>::vle(src + i * src_step + k, vl);
            vec_src = __riscv_vfmul(vec_src, s, vl);
            svd::rvv<T>::vse(src + i * src_step + k, vec_src, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

}}

#endif
