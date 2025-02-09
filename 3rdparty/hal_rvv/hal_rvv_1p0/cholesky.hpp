// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_CHOLESKY_HPP_INCLUDED
#define OPENCV_HAL_RVV_CHOLESKY_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_Cholesky32f
#define cv_hal_Cholesky32f cv::cv_hal_rvv::Cholesky
#undef cv_hal_Cholesky64f
#define cv_hal_Cholesky64f cv::cv_hal_rvv::Cholesky

namespace cholesky
{
    template<typename T> struct rvv;

    template<> struct rvv<float>
    {
        static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m4(); }
        static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m4(a); }
        static inline vfloat32m4_t vfmv_v_f(float a, size_t b) { return __riscv_vfmv_v_f_f32m4(a, b); }
        static inline vfloat32m1_t vfmv_s_f(float a, size_t b) { return __riscv_vfmv_s_f_f32m1(a, b); }
        static inline vfloat32m4_t vle(const float* a, size_t b) { return __riscv_vle32_v_f32m4(a, b); }
        static inline vfloat32m4_t vlse(const float* a, ptrdiff_t b, size_t c) { return __riscv_vlse32_v_f32m4(a, b, c); }
        static inline void vsse(float* a, ptrdiff_t b, vfloat32m4_t c, size_t d) { __riscv_vsse32(a, b, c, d); }
    };

    template<> struct rvv<double>
    {
        static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e64m4(); }
        static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e64m4(a); }
        static inline vfloat64m4_t vfmv_v_f(double a, size_t b) { return __riscv_vfmv_v_f_f64m4(a, b); }
        static inline vfloat64m1_t vfmv_s_f(double a, size_t b) { return __riscv_vfmv_s_f_f64m1(a, b); }
        static inline vfloat64m4_t vle(const double* a, size_t b) { return __riscv_vle64_v_f64m4(a, b); }
        static inline vfloat64m4_t vlse(const double* a, ptrdiff_t b, size_t c) { return __riscv_vlse64_v_f64m4(a, b, c); }
        static inline void vsse(double* a, ptrdiff_t b, vfloat64m4_t c, size_t d) { __riscv_vsse64(a, b, c, d); }
    };
}

// the algorithm is copied from core/src/matrix_decomp.cpp,
// in the function template static int cv::CholImpl
template<typename T>
inline int Cholesky(T* src1, size_t src1_step, int m, T* src2, size_t src2_step, int n, bool* info)
{
    int i, j, k;
    double s;
    src1_step /= sizeof(src1[0]);
    src2_step /= sizeof(src2[0]);

    int vlmax = cholesky::rvv<T>::vsetvlmax(), vl;
    for( i = 0; i < m; i++ )
    {
        for( j = 0; j < i; j++ )
        {
            auto vec_sum = cholesky::rvv<T>::vfmv_v_f(0, vlmax);
            for( k = 0; k < j; k += vl )
            {
                vl = cholesky::rvv<T>::vsetvl(j - k);
                auto vec_src1 = cholesky::rvv<T>::vle(src1 + i * src1_step + k, vl);
                auto vec_src2 = cholesky::rvv<T>::vle(src1 + j * src1_step + k, vl);
                vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
            }
            s = src1[i*src1_step + j] - __riscv_vfmv_f(__riscv_vfredosum(vec_sum, cholesky::rvv<T>::vfmv_s_f(0, vlmax), vlmax));
            src1[i*src1_step + j] = (T)(s*src1[j*src1_step + j]);
        }
        auto vec_sum = cholesky::rvv<T>::vfmv_v_f(0, vlmax);
        for( k = 0; k < j; k += vl )
        {
            vl = cholesky::rvv<T>::vsetvl(j - k);
            auto vec_src = cholesky::rvv<T>::vle(src1 + i * src1_step + k, vl);
            vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src, vec_src, vl);
        }
        s = src1[i*src1_step + i] - __riscv_vfmv_f(__riscv_vfredosum(vec_sum, cholesky::rvv<T>::vfmv_s_f(0, vlmax), vlmax));
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
            vl = cholesky::rvv<T>::vsetvl(m - i);
            auto vec_src = cholesky::rvv<T>::vlse(src1 + i * src1_step + i, sizeof(T) * (src1_step + 1), vl);
            vec_src = __riscv_vfrdiv(vec_src, 1, vl);
            cholesky::rvv<T>::vsse(src1 + i * src1_step + i, sizeof(T) * (src1_step + 1), vec_src, vl);
        }
        *info = true;
        return CV_HAL_ERROR_OK;
    }

    for( i = 0; i < m; i++ )
    {
        for( j = 0; j < n; j++ )
        {
            auto vec_sum = cholesky::rvv<T>::vfmv_v_f(0, vlmax);
            for( k = 0; k < i; k += vl )
            {
                vl = cholesky::rvv<T>::vsetvl(i - k);
                auto vec_src1 = cholesky::rvv<T>::vle(src1 + i * src1_step + k, vl);
                auto vec_src2 = cholesky::rvv<T>::vlse(src2 + k * src2_step + j, sizeof(T) * src2_step, vl);
                vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
            }
            s = src2[i*src2_step + j] - __riscv_vfmv_f(__riscv_vfredosum(vec_sum, cholesky::rvv<T>::vfmv_s_f(0, vlmax), vlmax));
            src2[i*src2_step + j] = (T)(s*src1[i*src1_step + i]);
        }
    }

    for( i = m-1; i >= 0; i-- )
    {
        for( j = 0; j < n; j++ )
        {
            auto vec_sum = cholesky::rvv<T>::vfmv_v_f(0, vlmax);
            for( k = i + 1; k < m; k += vl )
            {
                vl = cholesky::rvv<T>::vsetvl(m - k);
                auto vec_src1 = cholesky::rvv<T>::vlse(src1 + k * src1_step + i, sizeof(T) * src1_step, vl);
                auto vec_src2 = cholesky::rvv<T>::vlse(src2 + k * src2_step + j, sizeof(T) * src2_step, vl);
                vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
            }
            s = src2[i*src2_step + j] - __riscv_vfmv_f(__riscv_vfredosum(vec_sum, cholesky::rvv<T>::vfmv_s_f(0, vlmax), vlmax));
            src2[i*src2_step + j] = (T)(s*src1[i*src1_step + i]);
        }
    }
    for( i = 0; i < m; i += vl )
    {
        vl = cholesky::rvv<T>::vsetvl(m - i);
        auto vec_src = cholesky::rvv<T>::vlse(src1 + i * src1_step + i, sizeof(T) * (src1_step + 1), vl);
        vec_src = __riscv_vfrdiv(vec_src, 1, vl);
        cholesky::rvv<T>::vsse(src1 + i * src1_step + i, sizeof(T) * (src1_step + 1), vec_src, vl);
    }

    *info = true;
    return CV_HAL_ERROR_OK;
}

}}

#endif
