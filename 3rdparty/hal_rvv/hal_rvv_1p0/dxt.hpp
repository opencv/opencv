// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_DXT_HPP_INCLUDED
#define OPENCV_HAL_RVV_DXT_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_dft
#define cv_hal_dft cv::cv_hal_rvv::dft

namespace dxt {

template<typename T> struct rvv;

template<> struct rvv<float>
{
    static inline size_t vsetvl_itab(size_t a) { return __riscv_vsetvl_e32m8(a); }
    static inline vuint32m8_t vlse_itab(const uint* a, ptrdiff_t b, size_t c) { return __riscv_vlse32_v_u32m8(a, b, c); }
    static inline vfloat32m8_t vlse_itab_f(const float* a, ptrdiff_t b, size_t c) { return __riscv_vlse32_v_f32m8(a, b, c); }
    static inline void vsse_itab(float* a, ptrdiff_t b, vfloat32m8_t c, size_t d) { return __riscv_vsse32(a, b, c, d); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32mf2(a); }
    static inline vfloat32m1_t vfmv_s(float a, size_t b) { return __riscv_vfmv_s_f_f32m1(a, b); }
    static inline vfloat32mf2_t vlse(const float* a, ptrdiff_t b, size_t c) { return __riscv_vlse32_v_f32mf2(a, b, c); }
    static inline void vsse(float* a, ptrdiff_t b, vfloat32mf2_t c, size_t d) { return __riscv_vsse32(a, b, c, d); }
};

template<> struct rvv<double>
{
    static inline size_t vsetvl_itab(size_t a) { return __riscv_vsetvl_e32m4(a); }
    static inline vuint32m4_t vlse_itab(const uint* a, ptrdiff_t b, size_t c) { return __riscv_vlse32_v_u32m4(a, b, c); }
    static inline vfloat64m8_t vlse_itab_f(const double* a, ptrdiff_t b, size_t c) { return __riscv_vlse64_v_f64m8(a, b, c); }
    static inline void vsse_itab(double* a, ptrdiff_t b, vfloat64m8_t c, size_t d) { return __riscv_vsse64(a, b, c, d); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e64m1(a); }
    static inline vfloat64m1_t vfmv_s(double a, size_t b) { return __riscv_vfmv_s_f_f64m1(a, b); }
    static inline vfloat64m1_t vlse(const double* a, ptrdiff_t b, size_t c) { return __riscv_vlse64_v_f64m1(a, b, c); }
    static inline void vsse(double* a, ptrdiff_t b, vfloat64m1_t c, size_t d) { return __riscv_vsse64(a, b, c, d); }
};

} // cv::cv_hal_rvv::dxt

// the algorithm is copied from core/src/dxt.cpp,
// in the function template static void cv::DFT and cv::DFT_R2, cv::DFT_R3, cv::DFT_R5
template<typename T>
inline int dft(const Complex<T>* src, Complex<T>* dst, int nf, int *factors, T scale, int* itab,
                  const Complex<T>* wave, int tab_size, int len, bool isInverse, bool noPermute)
{
    int n = len;
    int f_idx, nx;
    int dw0 = tab_size, dw;
    int i, j, k;
    Complex<T> t;

    int tab_step = tab_size == n ? 1 : tab_size == n*2 ? 2 : tab_size/n;
    int vl;

    // 0. shuffle data
    if( dst != src )
    {
        if( !isInverse )
        {
            for( i = 0; i < n; i += vl )
            {
                vl = dxt::rvv<T>::vsetvl_itab(n - i);
                auto vec_itab = dxt::rvv<T>::vlse_itab(reinterpret_cast<const uint*>(itab + i * tab_step), sizeof(int) * tab_step, vl);
                vec_itab = __riscv_vmul(vec_itab, sizeof(T) * 2, vl);
                auto vec_src_re = __riscv_vloxei32(reinterpret_cast<const T*>(src), vec_itab, vl);
                vec_itab = __riscv_vadd(vec_itab, sizeof(T), vl);
                auto vec_src_im = __riscv_vloxei32(reinterpret_cast<const T*>(src), vec_itab, vl);
                dxt::rvv<T>::vsse_itab(reinterpret_cast<T*>(dst + i), sizeof(T) * 2, vec_src_re, vl);
                dxt::rvv<T>::vsse_itab(reinterpret_cast<T*>(dst + i) + 1, sizeof(T) * 2, vec_src_im, vl);
            }
        }
        else
        {
            for( i = 0; i < n; i += vl )
            {
                vl = dxt::rvv<T>::vsetvl_itab(n - i);
                auto vec_itab = dxt::rvv<T>::vlse_itab(reinterpret_cast<const uint*>(itab + i * tab_step), sizeof(int) * tab_step, vl);
                vec_itab = __riscv_vmul(vec_itab, sizeof(T) * 2, vl);
                auto vec_src_re = __riscv_vloxei32(reinterpret_cast<const T*>(src), vec_itab, vl);
                vec_itab = __riscv_vadd(vec_itab, sizeof(T), vl);
                auto vec_src_im = __riscv_vloxei32(reinterpret_cast<const T*>(src), vec_itab, vl);
                vec_src_im = __riscv_vfneg(vec_src_im, vl);
                dxt::rvv<T>::vsse_itab(reinterpret_cast<T*>(dst + i), sizeof(T) * 2, vec_src_re, vl);
                dxt::rvv<T>::vsse_itab(reinterpret_cast<T*>(dst + i) + 1, sizeof(T) * 2, vec_src_im, vl);
            }
        }
    }
    else
    {
        // copied from core/src/dxt.cpp, it is slow to swap elements by intrinsics
        if( !noPermute )
        {
            if( nf == 1 )
            {
                if( (n & 3) == 0 )
                {
                    int n2 = n/2;
                    Complex<T>* dsth = dst + n2;

                    for( i = 0; i < n2; i += 2, itab += tab_step*2 )
                    {
                        j = itab[0];

                        t = dst[i+1], dst[i+1] = dsth[j], dsth[j] = t;
                        if( j > i )
                        {
                            t = dst[i], dst[i] = dst[j], dst[j] = t;
                            t = dsth[i+1], dsth[i+1] = dsth[j+1], dsth[j+1] = t;
                        }
                    }
                }
                // else do nothing
            }
            else
            {
                for( i = 0; i < n; i++, itab += tab_step )
                {
                    j = itab[0];
                    if( j > i )
                        t = dst[i], dst[i] = dst[j], dst[j] = t;
                }
            }
        }

        if( isInverse )
        {
            for( i = 0; i < n; i += vl )
            {
                vl = dxt::rvv<T>::vsetvl_itab(n - i);
                auto vec_src_im = dxt::rvv<T>::vlse_itab_f(reinterpret_cast<const T*>(dst + i) + 1, sizeof(T) * 2, vl);
                vec_src_im = __riscv_vfneg(vec_src_im, vl);
                dxt::rvv<T>::vsse_itab(reinterpret_cast<T*>(dst + i) + 1, sizeof(T) * 2, vec_src_im, vl);
            }
        }
    }

    n = 1;
    // 1. power-2 transforms
    if( (factors[0] & 1) == 0 )
    {
        // radix-4 transform
        for( ; n*4 <= factors[0]; )
        {
            nx = n;
            n *= 4;
            dw0 /= 4;

            for( i = 0; i < len; i += n )
            {
                Complex<T> *v0, *v1;
                T r0, i0, r1, i1, r2, i2, r3, i3, r4, i4;

                v0 = dst + i;
                v1 = v0 + nx*2;

                r0 = v1[0].re; i0 = v1[0].im;
                r4 = v1[nx].re; i4 = v1[nx].im;

                r1 = r0 + r4; i1 = i0 + i4;
                r3 = i0 - i4; i3 = r4 - r0;

                r2 = v0[0].re; i2 = v0[0].im;
                r4 = v0[nx].re; i4 = v0[nx].im;

                r0 = r2 + r4; i0 = i2 + i4;
                r2 -= r4; i2 -= i4;

                v0[0].re = r0 + r1; v0[0].im = i0 + i1;
                v1[0].re = r0 - r1; v1[0].im = i0 - i1;
                v0[nx].re = r2 + r3; v0[nx].im = i2 + i3;
                v1[nx].re = r2 - r3; v1[nx].im = i2 - i3;

                for( j = 1; j < nx; j += vl )
                {
                    vl = dxt::rvv<T>::vsetvl(nx - j);
                    v0 = dst + i + j;
                    v1 = v0 + nx*2;

                    auto vec_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v1), sizeof(T) * 2, vl);
                    auto vec_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v1) + 1, sizeof(T) * 2, vl);
                    auto vec_w_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0), sizeof(T) * dw0 * 2, vl);
                    auto vec_w_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0) + 1, sizeof(T) * dw0 * 2, vl);
                    auto vec_r0 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);
                    auto vec_i0 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);

                    vec_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v1 + nx), sizeof(T) * 2, vl);
                    vec_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v1 + nx) + 1, sizeof(T) * 2, vl);
                    vec_w_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0 * 3), sizeof(T) * dw0 * 6, vl);
                    vec_w_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0 * 3) + 1, sizeof(T) * dw0 * 6, vl);
                    auto vec_r3 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);
                    auto vec_i3 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);

                    auto vec_r1 = __riscv_vfadd(vec_i0, vec_i3, vl);
                    auto vec_i1 = __riscv_vfadd(vec_r0, vec_r3, vl);
                    vec_r3 = __riscv_vfsub(vec_r0, vec_r3, vl);
                    vec_i3 = __riscv_vfsub(vec_i3, vec_i0, vl);
                    auto vec_r4 = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v0), sizeof(T) * 2, vl);
                    auto vec_i4 = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v0) + 1, sizeof(T) * 2, vl);

                    vec_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v0 + nx), sizeof(T) * 2, vl);
                    vec_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v0 + nx) + 1, sizeof(T) * 2, vl);
                    vec_w_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0 * 2), sizeof(T) * dw0 * 4, vl);
                    vec_w_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0 * 2) + 1, sizeof(T) * dw0 * 4, vl);
                    auto vec_r2 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i2 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_r0 = __riscv_vfadd(vec_r4, vec_r2, vl);
                    vec_i0 = __riscv_vfadd(vec_i4, vec_i2, vl);

                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v0), sizeof(T) * 2, __riscv_vfadd(vec_r0, vec_r1, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v0) + 1, sizeof(T) * 2, __riscv_vfadd(vec_i0, vec_i1, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v1), sizeof(T) * 2, __riscv_vfsub(vec_r0, vec_r1, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v1) + 1, sizeof(T) * 2, __riscv_vfsub(vec_i0, vec_i1, vl), vl);

                    vec_r2 = __riscv_vfsub(vec_r4, vec_r2, vl);
                    vec_i2 = __riscv_vfsub(vec_i4, vec_i2, vl);

                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v0 + nx), sizeof(T) * 2, __riscv_vfadd(vec_r2, vec_r3, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v0 + nx) + 1, sizeof(T) * 2, __riscv_vfadd(vec_i2, vec_i3, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v1 + nx), sizeof(T) * 2, __riscv_vfsub(vec_r2, vec_r3, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v1 + nx) + 1, sizeof(T) * 2, __riscv_vfsub(vec_i2, vec_i3, vl), vl);
                }
            }
        }

        for( ; n < factors[0]; )
        {
            // do the remaining radix-2 transform
            nx = n;
            n *= 2;
            dw0 /= 2;

            for( i = 0; i < len; i += n )
            {
                Complex<T>* v = dst + i;
                T r0 = v[0].re + v[nx].re;
                T i0 = v[0].im + v[nx].im;
                T r1 = v[0].re - v[nx].re;
                T i1 = v[0].im - v[nx].im;
                v[0].re = r0; v[0].im = i0;
                v[nx].re = r1; v[nx].im = i1;

                for( j = 1; j < nx; j += vl )
                {
                    vl = dxt::rvv<T>::vsetvl(nx - j);
                    v = dst + i + j;

                    auto vec_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + nx), sizeof(T) * 2, vl);
                    auto vec_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + nx) + 1, sizeof(T) * 2, vl);
                    auto vec_w_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0), sizeof(T) * dw0 * 2, vl);
                    auto vec_w_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0) + 1, sizeof(T) * dw0 * 2, vl);

                    auto vec_r1 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i1 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);
                    auto vec_r0 = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v), sizeof(T) * 2, vl);
                    auto vec_i0 = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v) + 1, sizeof(T) * 2, vl);

                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v), sizeof(T) * 2, __riscv_vfadd(vec_r0, vec_r1, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v) + 1, sizeof(T) * 2, __riscv_vfadd(vec_i0, vec_i1, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v + nx), sizeof(T) * 2, __riscv_vfsub(vec_r0, vec_r1, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v + nx) + 1, sizeof(T) * 2, __riscv_vfsub(vec_i0, vec_i1, vl), vl);
                }
            }
        }
    }

    // 2. all the other transforms
    for( f_idx = (factors[0]&1) ? 0 : 1; f_idx < nf; f_idx++ )
    {
        int factor = factors[f_idx];
        nx = n;
        n *= factor;
        dw0 /= factor;

        if( factor == 3 )
        {
            const T sin_120 = 0.86602540378443864676372317075294;
            for( i = 0; i < len; i += n )
            {
                Complex<T>* v = dst + i;
                T r1 = v[nx].re + v[nx*2].re;
                T i1 = v[nx].im + v[nx*2].im;
                T r0 = v[0].re;
                T i0 = v[0].im;
                T r2 = sin_120*(v[nx].im - v[nx*2].im);
                T i2 = sin_120*(v[nx*2].re - v[nx].re);
                v[0].re = r0 + r1; v[0].im = i0 + i1;
                r0 -= (T)0.5*r1; i0 -= (T)0.5*i1;
                v[nx].re = r0 + r2; v[nx].im = i0 + i2;
                v[nx*2].re = r0 - r2; v[nx*2].im = i0 - i2;

                for( j = 1; j < nx; j += vl )
                {
                    vl = dxt::rvv<T>::vsetvl(nx - j);
                    v = dst + i + j;

                    auto vec_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + nx), sizeof(T) * 2, vl);
                    auto vec_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + nx) + 1, sizeof(T) * 2, vl);
                    auto vec_w_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0), sizeof(T) * dw0 * 2, vl);
                    auto vec_w_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0) + 1, sizeof(T) * dw0 * 2, vl);
                    auto vec_r0 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i0 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + nx * 2), sizeof(T) * 2, vl);
                    vec_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + nx * 2) + 1, sizeof(T) * 2, vl);
                    vec_w_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0 * 2), sizeof(T) * dw0 * 4, vl);
                    vec_w_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0 * 2) + 1, sizeof(T) * dw0 * 4, vl);
                    auto vec_r2 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);
                    auto vec_i2 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);

                    auto vec_r1 = __riscv_vfadd(vec_r0, vec_i2, vl);
                    auto vec_i1 = __riscv_vfadd(vec_i0, vec_r2, vl);

                    vec_r2 = __riscv_vfmul(__riscv_vfsub(vec_i0, vec_r2, vl), sin_120, vl);
                    vec_i2 = __riscv_vfmul(__riscv_vfsub(vec_i2, vec_r0, vl), sin_120, vl);
                    vec_r0 = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v), sizeof(T) * 2, vl);
                    vec_i0 = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v) + 1, sizeof(T) * 2, vl);

                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v), sizeof(T) * 2, __riscv_vfadd(vec_r0, vec_r1, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v) + 1, sizeof(T) * 2, __riscv_vfadd(vec_i0, vec_i1, vl), vl);
                    vec_r0 = __riscv_vfsub(vec_r0, __riscv_vfmul(vec_r1, 0.5, vl), vl);
                    vec_i0 = __riscv_vfsub(vec_i0, __riscv_vfmul(vec_i1, 0.5, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v + nx), sizeof(T) * 2, __riscv_vfadd(vec_r0, vec_r2, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v + nx) + 1, sizeof(T) * 2, __riscv_vfadd(vec_i0, vec_i2, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v + nx * 2), sizeof(T) * 2, __riscv_vfsub(vec_r0, vec_r2, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v + nx * 2) + 1, sizeof(T) * 2, __riscv_vfsub(vec_i0, vec_i2, vl), vl);
                }
            }
        }
        else if( factor == 5 )
        {
            const T fft5_2 = 0.559016994374947424102293417182819;
            const T fft5_3 = -0.951056516295153572116439333379382;
            const T fft5_4 = -1.538841768587626701285145288018455;
            const T fft5_5 = 0.363271264002680442947733378740309;
            for( i = 0; i < len; i += n )
            {
                for( j = 0; j < nx; j += vl )
                {
                    vl = dxt::rvv<T>::vsetvl(nx - j);
                    Complex<T>* v0 = dst + i + j;
                    Complex<T>* v1 = v0 + nx*2;
                    Complex<T>* v2 = v1 + nx*2;

                    auto vec_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v0 + nx), sizeof(T) * 2, vl);
                    auto vec_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v0 + nx) + 1, sizeof(T) * 2, vl);
                    auto vec_w_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0), sizeof(T) * dw0 * 2, vl);
                    auto vec_w_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0) + 1, sizeof(T) * dw0 * 2, vl);
                    auto vec_r3 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i3 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v2), sizeof(T) * 2, vl);
                    vec_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v2) + 1, sizeof(T) * 2, vl);
                    vec_w_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0 * 4), sizeof(T) * dw0 * 8, vl);
                    vec_w_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0 * 4) + 1, sizeof(T) * dw0 * 8, vl);
                    auto vec_r2 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i2 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    auto vec_r1 = __riscv_vfadd(vec_r3, vec_r2, vl);
                    auto vec_i1 = __riscv_vfadd(vec_i3, vec_i2, vl);
                    vec_r3 = __riscv_vfsub(vec_r3, vec_r2, vl);
                    vec_i3 = __riscv_vfsub(vec_i3, vec_i2, vl);

                    vec_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v1 + nx), sizeof(T) * 2, vl);
                    vec_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v1 + nx) + 1, sizeof(T) * 2, vl);
                    vec_w_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0 * 3), sizeof(T) * dw0 * 6, vl);
                    vec_w_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0 * 3) + 1, sizeof(T) * dw0 * 6, vl);
                    auto vec_r4 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i4 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v1), sizeof(T) * 2, vl);
                    vec_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v1) + 1, sizeof(T) * 2, vl);
                    vec_w_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0 * 2), sizeof(T) * dw0 * 4, vl);
                    vec_w_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + j * dw0 * 2) + 1, sizeof(T) * dw0 * 4, vl);
                    auto vec_r0 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i0 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_r2 = __riscv_vfadd(vec_r4, vec_r0, vl);
                    vec_i2 = __riscv_vfadd(vec_i4, vec_i0, vl);
                    vec_r4 = __riscv_vfsub(vec_r4, vec_r0, vl);
                    vec_i4 = __riscv_vfsub(vec_i4, vec_i0, vl);

                    vec_r0 = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v0), sizeof(T) * 2, vl);
                    vec_i0 = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v0) + 1, sizeof(T) * 2, vl);
                    auto vec_r5 = __riscv_vfadd(vec_r1, vec_r2, vl);
                    auto vec_i5 = __riscv_vfadd(vec_i1, vec_i2, vl);

                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v0), sizeof(T) * 2, __riscv_vfadd(vec_r0, vec_r5, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v0) + 1, sizeof(T) * 2, __riscv_vfadd(vec_i0, vec_i5, vl), vl);

                    vec_r0 = __riscv_vfsub(vec_r0, __riscv_vfmul(vec_r5, 0.25, vl), vl);
                    vec_i0 = __riscv_vfsub(vec_i0, __riscv_vfmul(vec_i5, 0.25, vl), vl);
                    vec_r1 = __riscv_vfmul(__riscv_vfsub(vec_r1, vec_r2, vl), fft5_2, vl);
                    vec_i1 = __riscv_vfmul(__riscv_vfsub(vec_i1, vec_i2, vl), fft5_2, vl);
                    vec_r2 = __riscv_vfmul(__riscv_vfadd(vec_i3, vec_i4, vl), -fft5_3, vl);
                    vec_i2 = __riscv_vfmul(__riscv_vfadd(vec_r3, vec_r4, vl), fft5_3, vl);

                    vec_i3 = __riscv_vfmul(vec_i3, -fft5_5, vl);
                    vec_r3 = __riscv_vfmul(vec_r3, fft5_5, vl);
                    vec_i4 = __riscv_vfmul(vec_i4, -fft5_4, vl);
                    vec_r4 = __riscv_vfmul(vec_r4, fft5_4, vl);

                    vec_r5 = __riscv_vfadd(vec_r2, vec_i3, vl);
                    vec_i5 = __riscv_vfadd(vec_i2, vec_r3, vl);
                    vec_r2 = __riscv_vfsub(vec_r2, vec_i4, vl);
                    vec_i2 = __riscv_vfsub(vec_i2, vec_r4, vl);

                    vec_r3 = __riscv_vfadd(vec_r0, vec_r1, vl);
                    vec_i3 = __riscv_vfadd(vec_i0, vec_i1, vl);

                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v0 + nx), sizeof(T) * 2, __riscv_vfadd(vec_r3, vec_r2, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v0 + nx) + 1, sizeof(T) * 2, __riscv_vfadd(vec_i3, vec_i2, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v2), sizeof(T) * 2, __riscv_vfsub(vec_r3, vec_r2, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v2) + 1, sizeof(T) * 2, __riscv_vfsub(vec_i3, vec_i2, vl), vl);

                    vec_r0 = __riscv_vfsub(vec_r0, vec_r1, vl);
                    vec_i0 = __riscv_vfsub(vec_i0, vec_i1, vl);

                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v1), sizeof(T) * 2, __riscv_vfadd(vec_r0, vec_r5, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v1) + 1, sizeof(T) * 2, __riscv_vfadd(vec_i0, vec_i5, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v1 + nx), sizeof(T) * 2, __riscv_vfsub(vec_r0, vec_r5, vl), vl);
                    dxt::rvv<T>::vsse(reinterpret_cast<T*>(v1 + nx) + 1, sizeof(T) * 2, __riscv_vfsub(vec_i0, vec_i5, vl), vl);
                }
            }
        }
        else
        {
            // radix-"factor" - an odd number
            int p, q, factor2 = (factor - 1)/2;
            int dd, dw_f = tab_size/factor;
            AutoBuffer<Complex<T> > buf(factor2 * 2);
            Complex<T>* a = buf.data();
            Complex<T>* b = a + factor2;

            for( i = 0; i < len; i += n )
            {
                for( j = 0, dw = 0; j < nx; j++, dw += dw0 )
                {
                    Complex<T>* v = dst + i + j;
                    Complex<T> v_0 = v[0];
                    Complex<T> vn_0 = v_0;

                    if( j == 0 )
                    {
                        for( p = 1; p <= factor2; p += vl )
                        {
                            vl = dxt::rvv<T>::vsetvl(factor2 + 1 - p);

                            auto vec_a = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + nx * p), sizeof(T) * nx * 2, vl);
                            auto vec_b = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + n - nx * p), (long)sizeof(T) * nx * -2, vl);
                            auto vec_r0 = __riscv_vfadd(vec_a, vec_b, vl);
                            auto vec_r1 = __riscv_vfsub(vec_a, vec_b, vl);

                            vec_a = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + nx * p) + 1, sizeof(T) * nx * 2, vl);
                            vec_b = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + n - nx * p) + 1, (long)sizeof(T) * nx * -2, vl);
                            auto vec_i0 = __riscv_vfsub(vec_a, vec_b, vl);
                            auto vec_i1 = __riscv_vfadd(vec_a, vec_b, vl);

                            vn_0.re += __riscv_vfmv_f(__riscv_vfredosum(vec_r0, dxt::rvv<T>::vfmv_s(0, vl), vl));
                            vn_0.im += __riscv_vfmv_f(__riscv_vfredosum(vec_i1, dxt::rvv<T>::vfmv_s(0, vl), vl));

                            dxt::rvv<T>::vsse(reinterpret_cast<T*>(a + p - 1), sizeof(T) * 2, vec_r0, vl);
                            dxt::rvv<T>::vsse(reinterpret_cast<T*>(a + p - 1) + 1, sizeof(T) * 2, vec_i0, vl);
                            dxt::rvv<T>::vsse(reinterpret_cast<T*>(b + p - 1), sizeof(T) * 2, vec_r1, vl);
                            dxt::rvv<T>::vsse(reinterpret_cast<T*>(b + p - 1) + 1, sizeof(T) * 2, vec_i1, vl);
                        }
                    }
                    else
                    {
                        const Complex<T>* wave_ = wave + dw*factor;

                        for( p = 1; p <= factor2; p += vl )
                        {
                            vl = dxt::rvv<T>::vsetvl(factor2 + 1 - p);

                            auto vec_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + nx * p), sizeof(T) * nx * 2, vl);
                            auto vec_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + nx * p) + 1, sizeof(T) * nx * 2, vl);
                            auto vec_w_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + p * dw), sizeof(T) * dw * 2, vl);
                            auto vec_w_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave + p * dw) + 1, sizeof(T) * dw * 2, vl);
                            auto vec_r2 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                            auto vec_i2 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                            vec_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + n - nx * p), (long)sizeof(T) * nx * -2, vl);
                            vec_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(v + n - nx * p) + 1, (long)sizeof(T) * nx * -2, vl);
                            vec_w_re = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave_ - p * dw), (long)sizeof(T) * dw * -2, vl);
                            vec_w_im = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(wave_ - p * dw) + 1, (long)sizeof(T) * dw * -2, vl);
                            auto vec_r1 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                            auto vec_i1 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                            auto vec_r0 = __riscv_vfadd(vec_r2, vec_r1, vl);
                            auto vec_i0 = __riscv_vfsub(vec_i2, vec_i1, vl);
                            vec_r1 = __riscv_vfsub(vec_r2, vec_r1, vl);
                            vec_i1 = __riscv_vfadd(vec_i2, vec_i1, vl);

                            vn_0.re += __riscv_vfmv_f(__riscv_vfredosum(vec_r0, dxt::rvv<T>::vfmv_s(0, vl), vl));
                            vn_0.im += __riscv_vfmv_f(__riscv_vfredosum(vec_i1, dxt::rvv<T>::vfmv_s(0, vl), vl));

                            dxt::rvv<T>::vsse(reinterpret_cast<T*>(a + p - 1), sizeof(T) * 2, vec_r0, vl);
                            dxt::rvv<T>::vsse(reinterpret_cast<T*>(a + p - 1) + 1, sizeof(T) * 2, vec_i0, vl);
                            dxt::rvv<T>::vsse(reinterpret_cast<T*>(b + p - 1), sizeof(T) * 2, vec_r1, vl);
                            dxt::rvv<T>::vsse(reinterpret_cast<T*>(b + p - 1) + 1, sizeof(T) * 2, vec_i1, vl);
                        }
                    }

                    v[0] = vn_0;

                    for( p = 1, k = nx; p <= factor2; p++, k += nx )
                    {
                        Complex<T> s0 = v_0, s1 = v_0;
                        dd = dw_f*p;

                        vl = __riscv_vsetvlmax_e32mf2();
                        auto vec_dd = __riscv_vid_v_u32mf2(vl);
                        vec_dd = __riscv_vmul(vec_dd, dd, vl);
                        vec_dd = __riscv_vremu(vec_dd, tab_size, vl);

                        for( q = 0; q < factor2; q += vl )
                        {
                            vl = dxt::rvv<T>::vsetvl(factor2 - q);

                            auto vec_d = __riscv_vadd(vec_dd, (q + 1) * dd % tab_size, vl);
                            auto vmask = __riscv_vmsgeu(vec_d, tab_size, vl);
                            vec_d = __riscv_vsub_mu(vmask, vec_d, vec_d, tab_size, vl);
                            vec_d = __riscv_vmul(vec_d, sizeof(T) * 2, vl);

                            auto vec_w = __riscv_vloxei32(reinterpret_cast<const T*>(wave), vec_d, vl);
                            auto vec_v = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(a + q), sizeof(T) * 2, vl);
                            auto vec_r0 = __riscv_vfmul(vec_w, vec_v, vl);
                            vec_v = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(b + q) + 1, sizeof(T) * 2, vl);
                            auto vec_r1 = __riscv_vfmul(vec_w, vec_v, vl);

                            vec_w = __riscv_vloxei32(reinterpret_cast<const T*>(wave) + 1, vec_d, vl);
                            vec_v = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(a + q) + 1, sizeof(T) * 2, vl);
                            auto vec_i0 = __riscv_vfmul(vec_w, vec_v, vl);
                            vec_v = dxt::rvv<T>::vlse(reinterpret_cast<const T*>(b + q), sizeof(T) * 2, vl);
                            auto vec_i1 = __riscv_vfmul(vec_w, vec_v, vl);

                            T r0 = __riscv_vfmv_f(__riscv_vfredosum(vec_r0, dxt::rvv<T>::vfmv_s(0, vl), vl));
                            T i0 = __riscv_vfmv_f(__riscv_vfredosum(vec_i0, dxt::rvv<T>::vfmv_s(0, vl), vl));
                            T r1 = __riscv_vfmv_f(__riscv_vfredosum(vec_r1, dxt::rvv<T>::vfmv_s(0, vl), vl));
                            T i1 = __riscv_vfmv_f(__riscv_vfredosum(vec_i1, dxt::rvv<T>::vfmv_s(0, vl), vl));

                            s1.re += r0 + i0; s0.re += r0 - i0;
                            s1.im += r1 - i1; s0.im += r1 + i1;
                        }

                        v[k] = s0;
                        v[n-k] = s1;
                    }
                }
            }
        }
    }

    if( scale != 1 )
    {
        T re_scale = scale, im_scale = scale;
        if( isInverse )
            im_scale = -im_scale;

        for( i = 0; i < len; i += vl )
        {
            vl = dxt::rvv<T>::vsetvl_itab(len - i);
            auto vec_src_re = dxt::rvv<T>::vlse_itab_f(reinterpret_cast<const T*>(dst + i), sizeof(T) * 2, vl);
            auto vec_src_im = dxt::rvv<T>::vlse_itab_f(reinterpret_cast<const T*>(dst + i) + 1, sizeof(T) * 2, vl);
            vec_src_re = __riscv_vfmul(vec_src_re, re_scale, vl);
            vec_src_im = __riscv_vfmul(vec_src_im, im_scale, vl);
            dxt::rvv<T>::vsse_itab(reinterpret_cast<T*>(dst + i), sizeof(T) * 2, vec_src_re, vl);
            dxt::rvv<T>::vsse_itab(reinterpret_cast<T*>(dst + i) + 1, sizeof(T) * 2, vec_src_im, vl);
        }
    }
    else if( isInverse )
    {
        for( i = 0; i < len; i += vl )
        {
            vl = dxt::rvv<T>::vsetvl_itab(len - i);
            auto vec_src_im = dxt::rvv<T>::vlse_itab_f(reinterpret_cast<const T*>(dst + i) + 1, sizeof(T) * 2, vl);
            vec_src_im = __riscv_vfneg(vec_src_im, vl);
            dxt::rvv<T>::vsse_itab(reinterpret_cast<T*>(dst + i) + 1, sizeof(T) * 2, vec_src_im, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int dft(const uchar* src, uchar* dst, int depth, int nf, int *factors, double scale, int* itab, void* wave,
                  int tab_size, int n, bool isInverse, bool noPermute)
{
    if( n == 0 )
        return CV_HAL_ERROR_OK;

    switch( depth )
    {
    case CV_32F:
        return dft(reinterpret_cast<const Complex<float>*>(src), reinterpret_cast<Complex<float>*>(dst), nf, factors, (float)scale,
                      itab, reinterpret_cast<const Complex<float>*>(wave), tab_size, n, isInverse, noPermute);
    case CV_64F:
        return dft(reinterpret_cast<const Complex<double>*>(src), reinterpret_cast<Complex<double>*>(dst), nf, factors, (double)scale,
                      itab, reinterpret_cast<const Complex<double>*>(wave), tab_size, n, isInverse, noPermute);
    }
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

}}

#endif
