// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_LU_HPP_INCLUDED
#define OPENCV_HAL_RVV_LU_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_LU32f
#define cv_hal_LU32f cv::cv_hal_rvv::LU
#undef cv_hal_LU64f
#define cv_hal_LU64f cv::cv_hal_rvv::LU

namespace lu {

template<typename T> struct rvv;

template<> struct rvv<float>
{
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m4(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m4(a); }
    static inline vfloat32m4_t vfmv_v_f(float a, size_t b) { return __riscv_vfmv_v_f_f32m4(a, b); }
    static inline vfloat32m1_t vfmv_s_f(float a, size_t b) { return __riscv_vfmv_s_f_f32m1(a, b); }
    static inline vfloat32m4_t vle(const float* a, size_t b) { return __riscv_vle32_v_f32m4(a, b); }
    static inline vfloat32m4_t vlse(const float* a, ptrdiff_t b, size_t c) { return __riscv_vlse32_v_f32m4(a, b, c); }
    static inline void vse(float* a, vfloat32m4_t b, size_t c) { __riscv_vse32(a, b, c); }
};

template<> struct rvv<double>
{
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e64m4(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e64m4(a); }
    static inline vfloat64m4_t vfmv_v_f(double a, size_t b) { return __riscv_vfmv_v_f_f64m4(a, b); }
    static inline vfloat64m1_t vfmv_s_f(double a, size_t b) { return __riscv_vfmv_s_f_f64m1(a, b); }
    static inline vfloat64m4_t vle(const double* a, size_t b) { return __riscv_vle64_v_f64m4(a, b); }
    static inline vfloat64m4_t vlse(const double* a, ptrdiff_t b, size_t c) { return __riscv_vlse64_v_f64m4(a, b, c); }
    static inline void vse(double* a, vfloat64m4_t b, size_t c) { __riscv_vse64(a, b, c); }
};

} // cv::cv_hal_rvv::lu

// the algorithm is copied from core/src/matrix_decomp.cpp,
// in the function template static int cv::LUImpl
template<typename T>
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

    int vlmax = lu::rvv<T>::vsetvlmax(), vl;
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
                    vl = lu::rvv<T>::vsetvl(m - j);
                    auto vec_src1 = lu::rvv<T>::vle(src1 + i * src1_step + j, vl);
                    auto vec_src2 = lu::rvv<T>::vle(src1 + k * src1_step + j, vl);
                    lu::rvv<T>::vse(src1 + k * src1_step + j, vec_src1, vl);
                    lu::rvv<T>::vse(src1 + i * src1_step + j, vec_src2, vl);
                }
                for( j = 0; j < n; j += vl )
                {
                    vl = lu::rvv<T>::vsetvl(n - j);
                    auto vec_src1 = lu::rvv<T>::vle(src2 + i * src2_step + j, vl);
                    auto vec_src2 = lu::rvv<T>::vle(src2 + k * src2_step + j, vl);
                    lu::rvv<T>::vse(src2 + k * src2_step + j, vec_src1, vl);
                    lu::rvv<T>::vse(src2 + i * src2_step + j, vec_src2, vl);
                }
                p = -p;
            }

            T d = -1/src1[i*src1_step + i];

            for( j = i+1; j < m; j++ )
            {
                T alpha = src1[j*src1_step + i]*d;

                for( k = i+1; k < m; k += vl )
                {
                    vl = lu::rvv<T>::vsetvl(m - k);
                    auto vec_src = lu::rvv<T>::vle(src1 + i * src1_step + k, vl);
                    auto vec_dst = lu::rvv<T>::vle(src1 + j * src1_step + k, vl);
                    vec_dst = __riscv_vfmacc(vec_dst, alpha, vec_src, vl);
                    lu::rvv<T>::vse(src1 + j * src1_step + k, vec_dst, vl);
                }
                for( k = 0; k < n; k += vl )
                {
                    vl = lu::rvv<T>::vsetvl(n - k);
                    auto vec_src = lu::rvv<T>::vle(src2 + i * src2_step + k, vl);
                    auto vec_dst = lu::rvv<T>::vle(src2 + j * src2_step + k, vl);
                    vec_dst = __riscv_vfmacc(vec_dst, alpha, vec_src, vl);
                    lu::rvv<T>::vse(src2 + j * src2_step + k, vec_dst, vl);
                }
            }
        }

        for( i = m-1; i >= 0; i-- )
            for( j = 0; j < n; j++ )
            {
                T s = src2[i*src2_step + j];
                auto vec_sum = lu::rvv<T>::vfmv_v_f(0, vlmax);
                for( k = i+1; k < m; k += vl )
                {
                    vl = lu::rvv<T>::vsetvl(m - k);
                    auto vec_src1 = lu::rvv<T>::vle(src1 + i * src1_step + k, vl);
                    auto vec_src2 = lu::rvv<T>::vlse(src2 + k * src2_step + j, sizeof(T) * src2_step, vl);
                    vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
                }
                s -= __riscv_vfmv_f(__riscv_vfredosum(vec_sum, lu::rvv<T>::vfmv_s_f(0, vlmax), vlmax));
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
                    vl = lu::rvv<T>::vsetvl(m - j);
                    auto vec_src1 = lu::rvv<T>::vle(src1 + i * src1_step + j, vl);
                    auto vec_src2 = lu::rvv<T>::vle(src1 + k * src1_step + j, vl);
                    lu::rvv<T>::vse(src1 + k * src1_step + j, vec_src1, vl);
                    lu::rvv<T>::vse(src1 + i * src1_step + j, vec_src2, vl);
                }
                p = -p;
            }

            T d = -1/src1[i*src1_step + i];

            for( j = i+1; j < m; j++ )
            {
                T alpha = src1[j*src1_step + i]*d;

                for( k = i+1; k < m; k += vl )
                {
                    vl = lu::rvv<T>::vsetvl(m - k);
                    auto vec_src = lu::rvv<T>::vle(src1 + i * src1_step + k, vl);
                    auto vec_dst = lu::rvv<T>::vle(src1 + j * src1_step + k, vl);
                    vec_dst = __riscv_vfmacc(vec_dst, alpha, vec_src, vl);
                    lu::rvv<T>::vse(src1 + j * src1_step + k, vec_dst, vl);
                }
            }
        }
    }

    *info = p;
    return CV_HAL_ERROR_OK;
}

}}

#endif
