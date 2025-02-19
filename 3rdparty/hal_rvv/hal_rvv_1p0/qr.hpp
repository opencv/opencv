// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_QR_HPP_INCLUDED
#define OPENCV_HAL_RVV_QR_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_QR32f
#define cv_hal_QR32f cv::cv_hal_rvv::QR
#undef cv_hal_QR64f
#define cv_hal_QR64f cv::cv_hal_rvv::QR

namespace qr {

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
    static inline void vse(double* a, vfloat64m4_t b, size_t c) { __riscv_vse64(a, b, c); }
    static inline void vsse(double* a, ptrdiff_t b, vfloat64m4_t c, size_t d) { __riscv_vsse64(a, b, c, d); }
};

} // cv::cv_hal_rvv::qr

// the algorithm is copied from core/src/matrix_decomp.cpp,
// in the function template static int cv::QRImpl
template<typename T>
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

    cv::AutoBuffer<T> buffer;
    size_t buf_size = m ? m + n : dst != NULL;
    buffer.allocate(buf_size);
    T* val = buffer.data();
    if (dst == NULL)
        dst = val + m;

    int vlmax = qr::rvv<T>::vsetvlmax(), vl;
    for (int l = 0; l < n; l++)
    {
        //generate val
        int vlSize = m - l;
        auto vec_sum = qr::rvv<T>::vfmv_v_f(0, vlmax);
        for (int i = 0; i < vlSize; i += vl)
        {
            vl = qr::rvv<T>::vsetvl(vlSize - i);
            auto vec_src = qr::rvv<T>::vlse(src1 + (l + i) * src1_step + l, sizeof(T) * src1_step, vl);
            qr::rvv<T>::vse(val + i, vec_src, vl);
            vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src, vec_src, vl);
        }
        T vlNorm = __riscv_vfmv_f(__riscv_vfredosum(vec_sum, qr::rvv<T>::vfmv_s_f(0, vlmax), vlmax));
        T tmpV = val[0];
        val[0] = val[0] + (val[0] >= 0 ? 1 : -1) * std::sqrt(vlNorm);
        vlNorm = std::sqrt(vlNorm + val[0] * val[0] - tmpV*tmpV);
        for (int i = 0; i < vlSize; i += vl)
        {
            vl = qr::rvv<T>::vsetvl(vlSize - i);
            auto vec_src = qr::rvv<T>::vle(val + i, vl);
            vec_src = __riscv_vfdiv(vec_src, vlNorm, vl);
            qr::rvv<T>::vse(val + i, vec_src, vl);
        }
        //multiply A_l*val
        for (int j = l; j < n; j++)
        {
            vec_sum = qr::rvv<T>::vfmv_v_f(0, vlmax);
            for (int i = l; i < m; i += vl)
            {
                vl = qr::rvv<T>::vsetvl(m - i);
                auto vec_src1 = qr::rvv<T>::vle(val + i - l, vl);
                auto vec_src2 = qr::rvv<T>::vlse(src1 + i * src1_step + j, sizeof(T) * src1_step, vl);
                vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
            }
            T v_lA = 2 * __riscv_vfmv_f(__riscv_vfredosum(vec_sum, qr::rvv<T>::vfmv_s_f(0, vlmax), vlmax));

            for (int i = l; i < m; i += vl)
            {
                vl = qr::rvv<T>::vsetvl(m - i);
                auto vec_src1 = qr::rvv<T>::vle(val + i - l, vl);
                auto vec_src2 = qr::rvv<T>::vlse(src1 + i * src1_step + j, sizeof(T) * src1_step, vl);
                vec_src2 = __riscv_vfnmsac(vec_src2, v_lA, vec_src1, vl);
                qr::rvv<T>::vsse(src1 + i * src1_step + j, sizeof(T) * src1_step, vec_src2, vl);
            }
        }

        //save val and factors
        dst[l] = val[0] * val[0];
        for (int i = 1; i < vlSize; i += vl)
        {
            vl = qr::rvv<T>::vsetvl(vlSize - i);
            auto vec_src = qr::rvv<T>::vle(val + i, vl);
            vec_src = __riscv_vfdiv(vec_src, val[0], vl);
            qr::rvv<T>::vsse(src1 + (l + i) * src1_step + l, sizeof(T) * src1_step, vec_src, vl);
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
                vl = qr::rvv<T>::vsetvl(m - l - j);
                auto vec_src = qr::rvv<T>::vlse(src1 + (j + l) * src1_step + l, sizeof(T) * src1_step, vl);
                qr::rvv<T>::vse(val + j, vec_src, vl);
            }

            //h_l*x
            for (int j = 0; j < k; j++)
            {
                auto vec_sum = qr::rvv<T>::vfmv_v_f(0, vlmax);
                for (int i = l; i < m; i += vl)
                {
                    vl = qr::rvv<T>::vsetvl(m - i);
                    auto vec_src1 = qr::rvv<T>::vle(val + i - l, vl);
                    auto vec_src2 = qr::rvv<T>::vlse(src2 + i * src2_step + j, sizeof(T) * src2_step, vl);
                    vec_sum = __riscv_vfmacc_tu(vec_sum, vec_src1, vec_src2, vl);
                }
                T v_lB = 2 * dst[l] * __riscv_vfmv_f(__riscv_vfredosum(vec_sum, qr::rvv<T>::vfmv_s_f(0, vlmax), vlmax));

                for (int i = l; i < m; i += vl)
                {
                    vl = qr::rvv<T>::vsetvl(m - i);
                    auto vec_src1 = qr::rvv<T>::vle(val + i - l, vl);
                    auto vec_src2 = qr::rvv<T>::vlse(src2 + i * src2_step + j, sizeof(T) * src2_step, vl);
                    vec_src2 = __riscv_vfnmsac(vec_src2, v_lB, vec_src1, vl);
                    qr::rvv<T>::vsse(src2 + i * src2_step + j, sizeof(T) * src2_step, vec_src2, vl);
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
                    vl = qr::rvv<T>::vsetvl(k - p);
                    auto vec_src1 = qr::rvv<T>::vle(src2 + i * src2_step + p, vl);
                    auto vec_src2 = qr::rvv<T>::vle(src2 + j * src2_step + p, vl);
                    vec_src1 = __riscv_vfnmsac(vec_src1, src1[i*src1_step + j], vec_src2, vl);
                    qr::rvv<T>::vse(src2 + i * src2_step + p, vec_src1, vl);
                }
            }
            if (std::abs(src1[i*src1_step + i]) < eps)
            {
                *info = 0;
                return CV_HAL_ERROR_OK;
            }
            for (int p = 0; p < k; p += vl)
            {
                vl = qr::rvv<T>::vsetvl(k - p);
                auto vec_src = qr::rvv<T>::vle(src2 + i * src2_step + p, vl);
                vec_src = __riscv_vfdiv(vec_src, src1[i*src1_step + i], vl);
                qr::rvv<T>::vse(src2 + i * src2_step + p, vec_src, vl);
            }
        }
    }

    *info = 1;
    return CV_HAL_ERROR_OK;
}

}}

#endif
