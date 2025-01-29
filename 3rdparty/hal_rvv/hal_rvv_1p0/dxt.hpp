// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_DXT_HPP_INCLUDED
#define OPENCV_HAL_RVV_DXT_HPP_INCLUDED

#include "opencv2/core/types.hpp"
#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_dftOcv
#define cv_hal_dftOcv cv::cv_hal_rvv::dftOcv

inline int dftOcv32F(const Complex<float>* src, Complex<float>* dst, int nf, int *factors, float scale, int* itab,
                     const Complex<float>* wave, int tab_size, int len, bool isInverse, bool noPermute)
{
    int n = len;
    int f_idx, nx;
    int dw0 = tab_size, dw;
    int i, j, k;
    Complex<float> t;

    int tab_step = tab_size == n ? 1 : tab_size == n*2 ? 2 : tab_size/n;
    int vl;

    // 0. shuffle data
    if( dst != src )
    {
        if( !isInverse )
        {
            for( i = 0; i < n; i += vl )
            {
                vl = __riscv_vsetvl_e32m8(n - i);
                auto vec_itab = __riscv_vlse32_v_u32m8(reinterpret_cast<const uint*>(itab + i * tab_step), sizeof(int) * tab_step, vl);
                vec_itab = __riscv_vmul(vec_itab, sizeof(float) * 2, vl);
                auto vec_src_re = __riscv_vloxei32(reinterpret_cast<const float*>(src), vec_itab, vl);
                vec_itab = __riscv_vadd(vec_itab, sizeof(float), vl);
                auto vec_src_im = __riscv_vloxei32(reinterpret_cast<const float*>(src), vec_itab, vl);
                __riscv_vsse32(reinterpret_cast<float*>(dst + i), sizeof(float) * 2, vec_src_re, vl);
                __riscv_vsse32(reinterpret_cast<float*>(dst + i) + 1, sizeof(float) * 2, vec_src_im, vl);
            }
        }
        else
        {
            for( i = 0; i < n; i += vl )
            {
                vl = __riscv_vsetvl_e32m8(n - i);
                auto vec_itab = __riscv_vlse32_v_u32m8(reinterpret_cast<const uint*>(itab + i * tab_step), sizeof(int) * tab_step, vl);
                vec_itab = __riscv_vmul(vec_itab, sizeof(float) * 2, vl);
                auto vec_src_re = __riscv_vloxei32(reinterpret_cast<const float*>(src), vec_itab, vl);
                vec_itab = __riscv_vadd(vec_itab, sizeof(float), vl);
                auto vec_src_im = __riscv_vloxei32(reinterpret_cast<const float*>(src), vec_itab, vl);
                vec_src_im = __riscv_vfneg(vec_src_im, vl);
                __riscv_vsse32(reinterpret_cast<float*>(dst + i), sizeof(float) * 2, vec_src_re, vl);
                __riscv_vsse32(reinterpret_cast<float*>(dst + i) + 1, sizeof(float) * 2, vec_src_im, vl);
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
                    Complex<float>* dsth = dst + n2;

                    for( i = 0; i < n2; i += 2, itab += tab_step*2 )
                    {
                        j = itab[0];

                        CV_SWAP(dst[i+1], dsth[j], t);
                        if( j > i )
                        {
                            CV_SWAP(dst[i], dst[j], t);
                            CV_SWAP(dsth[i+1], dsth[j+1], t);
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
                        CV_SWAP(dst[i], dst[j], t);
                }
            }
        }

        if( isInverse )
        {
            for( i = 0; i < n; i += vl )
            {
                vl = __riscv_vsetvl_e32m8(n - i);
                auto vec_src_im = __riscv_vlse32_v_f32m8(reinterpret_cast<const float*>(dst + i) + 1, sizeof(float) * 2, vl);
                vec_src_im = __riscv_vfneg(vec_src_im, vl);
                __riscv_vsse32(reinterpret_cast<float*>(dst + i) + 1, sizeof(float) * 2, vec_src_im, vl);
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
                Complex<float> *v0, *v1;
                float r0, i0, r1, i1, r2, i2, r3, i3, r4, i4;

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
                    vl = __riscv_vsetvl_e32mf2(nx - j);
                    v0 = dst + i + j;
                    v1 = v0 + nx*2;

                    auto vec_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v1), sizeof(float) * 2, vl);
                    auto vec_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v1) + 1, sizeof(float) * 2, vl);
                    auto vec_w_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0), sizeof(float) * dw0 * 2, vl);
                    auto vec_w_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0) + 1, sizeof(float) * dw0 * 2, vl);
                    auto vec_r0 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);
                    auto vec_i0 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);

                    vec_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v1 + nx), sizeof(float) * 2, vl);
                    vec_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v1 + nx) + 1, sizeof(float) * 2, vl);
                    vec_w_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0 * 3), sizeof(float) * dw0 * 6, vl);
                    vec_w_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0 * 3) + 1, sizeof(float) * dw0 * 6, vl);
                    auto vec_r3 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);
                    auto vec_i3 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);

                    auto vec_r1 = __riscv_vfadd(vec_i0, vec_i3, vl);
                    auto vec_i1 = __riscv_vfadd(vec_r0, vec_r3, vl);
                    vec_r3 = __riscv_vfsub(vec_r0, vec_r3, vl);
                    vec_i3 = __riscv_vfsub(vec_i3, vec_i0, vl);
                    auto vec_r4 = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v0), sizeof(float) * 2, vl);
                    auto vec_i4 = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v0) + 1, sizeof(float) * 2, vl);

                    vec_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v0 + nx), sizeof(float) * 2, vl);
                    vec_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v0 + nx) + 1, sizeof(float) * 2, vl);
                    vec_w_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0 * 2), sizeof(float) * dw0 * 4, vl);
                    vec_w_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0 * 2) + 1, sizeof(float) * dw0 * 4, vl);
                    auto vec_r2 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i2 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_r0 = __riscv_vfadd(vec_r4, vec_r2, vl);
                    vec_i0 = __riscv_vfadd(vec_i4, vec_i2, vl);

                    __riscv_vsse32(reinterpret_cast<float*>(v0), sizeof(float) * 2, __riscv_vfadd(vec_r0, vec_r1, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v0) + 1, sizeof(float) * 2, __riscv_vfadd(vec_i0, vec_i1, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v1), sizeof(float) * 2, __riscv_vfsub(vec_r0, vec_r1, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v1) + 1, sizeof(float) * 2, __riscv_vfsub(vec_i0, vec_i1, vl), vl);

                    vec_r2 = __riscv_vfsub(vec_r4, vec_r2, vl);
                    vec_i2 = __riscv_vfsub(vec_i4, vec_i2, vl);

                    __riscv_vsse32(reinterpret_cast<float*>(v0 + nx), sizeof(float) * 2, __riscv_vfadd(vec_r2, vec_r3, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v0 + nx) + 1, sizeof(float) * 2, __riscv_vfadd(vec_i2, vec_i3, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v1 + nx), sizeof(float) * 2, __riscv_vfsub(vec_r2, vec_r3, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v1 + nx) + 1, sizeof(float) * 2, __riscv_vfsub(vec_i2, vec_i3, vl), vl);
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
                Complex<float>* v = dst + i;
                float r0 = v[0].re + v[nx].re;
                float i0 = v[0].im + v[nx].im;
                float r1 = v[0].re - v[nx].re;
                float i1 = v[0].im - v[nx].im;
                v[0].re = r0; v[0].im = i0;
                v[nx].re = r1; v[nx].im = i1;

                for( j = 1; j < nx; j += vl )
                {
                    vl = __riscv_vsetvl_e32mf2(nx - j);
                    v = dst + i + j;

                    auto vec_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + nx), sizeof(float) * 2, vl);
                    auto vec_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + nx) + 1, sizeof(float) * 2, vl);
                    auto vec_w_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0), sizeof(float) * dw0 * 2, vl);
                    auto vec_w_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0) + 1, sizeof(float) * dw0 * 2, vl);

                    auto vec_r1 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i1 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);
                    auto vec_r0 = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v), sizeof(float) * 2, vl);
                    auto vec_i0 = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v) + 1, sizeof(float) * 2, vl);

                    __riscv_vsse32(reinterpret_cast<float*>(v), sizeof(float) * 2, __riscv_vfadd(vec_r0, vec_r1, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v) + 1, sizeof(float) * 2, __riscv_vfadd(vec_i0, vec_i1, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v + nx), sizeof(float) * 2, __riscv_vfsub(vec_r0, vec_r1, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v + nx) + 1, sizeof(float) * 2, __riscv_vfsub(vec_i0, vec_i1, vl), vl);
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
            const float sin_120 = 0.86602540378443864676372317075294;
            for( i = 0; i < len; i += n )
            {
                Complex<float>* v = dst + i;
                float r1 = v[nx].re + v[nx*2].re;
                float i1 = v[nx].im + v[nx*2].im;
                float r0 = v[0].re;
                float i0 = v[0].im;
                float r2 = sin_120*(v[nx].im - v[nx*2].im);
                float i2 = sin_120*(v[nx*2].re - v[nx].re);
                v[0].re = r0 + r1; v[0].im = i0 + i1;
                r0 -= (float)0.5*r1; i0 -= (float)0.5*i1;
                v[nx].re = r0 + r2; v[nx].im = i0 + i2;
                v[nx*2].re = r0 - r2; v[nx*2].im = i0 - i2;

                for( j = 1; j < nx; j += vl )
                {
                    vl = __riscv_vsetvl_e32mf2(nx - j);
                    v = dst + i + j;

                    auto vec_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + nx), sizeof(float) * 2, vl);
                    auto vec_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + nx) + 1, sizeof(float) * 2, vl);
                    auto vec_w_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0), sizeof(float) * dw0 * 2, vl);
                    auto vec_w_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0) + 1, sizeof(float) * dw0 * 2, vl);
                    auto vec_r0 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i0 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + nx * 2), sizeof(float) * 2, vl);
                    vec_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + nx * 2) + 1, sizeof(float) * 2, vl);
                    vec_w_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0 * 2), sizeof(float) * dw0 * 4, vl);
                    vec_w_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0 * 2) + 1, sizeof(float) * dw0 * 4, vl);
                    auto vec_r2 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);
                    auto vec_i2 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);

                    auto vec_r1 = __riscv_vfadd(vec_r0, vec_i2, vl);
                    auto vec_i1 = __riscv_vfadd(vec_i0, vec_r2, vl);

                    vec_r2 = __riscv_vfmul(__riscv_vfsub(vec_i0, vec_r2, vl), sin_120, vl);
                    vec_i2 = __riscv_vfmul(__riscv_vfsub(vec_i2, vec_r0, vl), sin_120, vl);
                    vec_r0 = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v), sizeof(float) * 2, vl);
                    vec_i0 = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v) + 1, sizeof(float) * 2, vl);

                    __riscv_vsse32(reinterpret_cast<float*>(v), sizeof(float) * 2, __riscv_vfadd(vec_r0, vec_r1, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v) + 1, sizeof(float) * 2, __riscv_vfadd(vec_i0, vec_i1, vl), vl);
                    vec_r0 = __riscv_vfsub(vec_r0, __riscv_vfmul(vec_r1, 0.5, vl), vl);
                    vec_i0 = __riscv_vfsub(vec_i0, __riscv_vfmul(vec_i1, 0.5, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v + nx), sizeof(float) * 2, __riscv_vfadd(vec_r0, vec_r2, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v + nx) + 1, sizeof(float) * 2, __riscv_vfadd(vec_i0, vec_i2, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v + nx * 2), sizeof(float) * 2, __riscv_vfsub(vec_r0, vec_r2, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v + nx * 2) + 1, sizeof(float) * 2, __riscv_vfsub(vec_i0, vec_i2, vl), vl);
                }
            }
        }
        else if( factor == 5 )
        {
            const float fft5_2 = 0.559016994374947424102293417182819;
            const float fft5_3 = -0.951056516295153572116439333379382;
            const float fft5_4 = -1.538841768587626701285145288018455;
            const float fft5_5 = 0.363271264002680442947733378740309;
            for( i = 0; i < len; i += n )
            {
                for( j = 0; j < nx; j += vl )
                {
                    vl = __riscv_vsetvl_e32mf2(nx - j);
                    Complex<float>* v0 = dst + i + j;
                    Complex<float>* v1 = v0 + nx*2;
                    Complex<float>* v2 = v1 + nx*2;

                    auto vec_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v0 + nx), sizeof(float) * 2, vl);
                    auto vec_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v0 + nx) + 1, sizeof(float) * 2, vl);
                    auto vec_w_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0), sizeof(float) * dw0 * 2, vl);
                    auto vec_w_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0) + 1, sizeof(float) * dw0 * 2, vl);
                    auto vec_r3 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i3 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v2), sizeof(float) * 2, vl);
                    vec_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v2) + 1, sizeof(float) * 2, vl);
                    vec_w_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0 * 4), sizeof(float) * dw0 * 8, vl);
                    vec_w_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0 * 4) + 1, sizeof(float) * dw0 * 8, vl);
                    auto vec_r2 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i2 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    auto vec_r1 = __riscv_vfadd(vec_r3, vec_r2, vl);
                    auto vec_i1 = __riscv_vfadd(vec_i3, vec_i2, vl);
                    vec_r3 = __riscv_vfsub(vec_r3, vec_r2, vl);
                    vec_i3 = __riscv_vfsub(vec_i3, vec_i2, vl);

                    vec_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v1 + nx), sizeof(float) * 2, vl);
                    vec_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v1 + nx) + 1, sizeof(float) * 2, vl);
                    vec_w_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0 * 3), sizeof(float) * dw0 * 6, vl);
                    vec_w_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0 * 3) + 1, sizeof(float) * dw0 * 6, vl);
                    auto vec_r4 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i4 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v1), sizeof(float) * 2, vl);
                    vec_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v1) + 1, sizeof(float) * 2, vl);
                    vec_w_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0 * 2), sizeof(float) * dw0 * 4, vl);
                    vec_w_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + j * dw0 * 2) + 1, sizeof(float) * dw0 * 4, vl);
                    auto vec_r0 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i0 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_r2 = __riscv_vfadd(vec_r4, vec_r0, vl);
                    vec_i2 = __riscv_vfadd(vec_i4, vec_i0, vl);
                    vec_r4 = __riscv_vfsub(vec_r4, vec_r0, vl);
                    vec_i4 = __riscv_vfsub(vec_i4, vec_i0, vl);

                    vec_r0 = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v0), sizeof(float) * 2, vl);
                    vec_i0 = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v0) + 1, sizeof(float) * 2, vl);
                    auto vec_r5 = __riscv_vfadd(vec_r1, vec_r2, vl);
                    auto vec_i5 = __riscv_vfadd(vec_i1, vec_i2, vl);

                    __riscv_vsse32(reinterpret_cast<float*>(v0), sizeof(float) * 2, __riscv_vfadd(vec_r0, vec_r5, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v0) + 1, sizeof(float) * 2, __riscv_vfadd(vec_i0, vec_i5, vl), vl);

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

                    __riscv_vsse32(reinterpret_cast<float*>(v0 + nx), sizeof(float) * 2, __riscv_vfadd(vec_r3, vec_r2, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v0 + nx) + 1, sizeof(float) * 2, __riscv_vfadd(vec_i3, vec_i2, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v2), sizeof(float) * 2, __riscv_vfsub(vec_r3, vec_r2, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v2) + 1, sizeof(float) * 2, __riscv_vfsub(vec_i3, vec_i2, vl), vl);

                    vec_r0 = __riscv_vfsub(vec_r0, vec_r1, vl);
                    vec_i0 = __riscv_vfsub(vec_i0, vec_i1, vl);

                    __riscv_vsse32(reinterpret_cast<float*>(v1), sizeof(float) * 2, __riscv_vfadd(vec_r0, vec_r5, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v1) + 1, sizeof(float) * 2, __riscv_vfadd(vec_i0, vec_i5, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v1 + nx), sizeof(float) * 2, __riscv_vfsub(vec_r0, vec_r5, vl), vl);
                    __riscv_vsse32(reinterpret_cast<float*>(v1 + nx) + 1, sizeof(float) * 2, __riscv_vfsub(vec_i0, vec_i5, vl), vl);
                }
            }
        }
        else
        {
            // radix-"factor" - an odd number
            int p, q, factor2 = (factor - 1)/2;
            int dd, dw_f = tab_size/factor;
            AutoBuffer<Complex<float> > buf(factor2 * 2);
            Complex<float>* a = buf.data();
            Complex<float>* b = a + factor2;

            for( i = 0; i < len; i += n )
            {
                for( j = 0, dw = 0; j < nx; j++, dw += dw0 )
                {
                    Complex<float>* v = dst + i + j;
                    Complex<float> v_0 = v[0];
                    Complex<float> vn_0 = v_0;

                    if( j == 0 )
                    {
                        for( p = 1; p <= factor2; p += vl )
                        {
                            vl = __riscv_vsetvl_e32mf2(factor2 + 1 - p);

                            auto vec_a = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + nx * p), sizeof(float) * nx * 2, vl);
                            auto vec_b = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + n - nx * p), (long)sizeof(float) * nx * -2, vl);
                            auto vec_r0 = __riscv_vfadd(vec_a, vec_b, vl);
                            auto vec_r1 = __riscv_vfsub(vec_a, vec_b, vl);

                            vec_a = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + nx * p) + 1, sizeof(float) * nx * 2, vl);
                            vec_b = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + n - nx * p) + 1, (long)sizeof(float) * nx * -2, vl);
                            auto vec_i0 = __riscv_vfsub(vec_a, vec_b, vl);
                            auto vec_i1 = __riscv_vfadd(vec_a, vec_b, vl);

                            vn_0.re += __riscv_vfmv_f(__riscv_vfredosum(vec_r0, __riscv_vfmv_s_f_f32m1(0, vl), vl));
                            vn_0.im += __riscv_vfmv_f(__riscv_vfredosum(vec_i1, __riscv_vfmv_s_f_f32m1(0, vl), vl));

                            __riscv_vsse32(reinterpret_cast<float*>(a + p - 1), sizeof(float) * 2, vec_r0, vl);
                            __riscv_vsse32(reinterpret_cast<float*>(a + p - 1) + 1, sizeof(float) * 2, vec_i0, vl);
                            __riscv_vsse32(reinterpret_cast<float*>(b + p - 1), sizeof(float) * 2, vec_r1, vl);
                            __riscv_vsse32(reinterpret_cast<float*>(b + p - 1) + 1, sizeof(float) * 2, vec_i1, vl);
                        }
                    }
                    else
                    {
                        const Complex<float>* wave_ = wave + dw*factor;

                        for( p = 1; p <= factor2; p += vl )
                        {
                            vl = __riscv_vsetvl_e32mf2(factor2 + 1 - p);

                            auto vec_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + nx * p), sizeof(float) * nx * 2, vl);
                            auto vec_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + nx * p) + 1, sizeof(float) * nx * 2, vl);
                            auto vec_w_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + p * dw), sizeof(float) * dw * 2, vl);
                            auto vec_w_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave + p * dw) + 1, sizeof(float) * dw * 2, vl);
                            auto vec_r2 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                            auto vec_i2 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                            vec_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + n - nx * p), (long)sizeof(float) * nx * -2, vl);
                            vec_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(v + n - nx * p) + 1, (long)sizeof(float) * nx * -2, vl);
                            vec_w_re = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave_ - p * dw), (long)sizeof(float) * dw * -2, vl);
                            vec_w_im = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(wave_ - p * dw) + 1, (long)sizeof(float) * dw * -2, vl);
                            auto vec_r1 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                            auto vec_i1 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                            auto vec_r0 = __riscv_vfadd(vec_r2, vec_r1, vl);
                            auto vec_i0 = __riscv_vfsub(vec_i2, vec_i1, vl);
                            vec_r1 = __riscv_vfsub(vec_r2, vec_r1, vl);
                            vec_i1 = __riscv_vfadd(vec_i2, vec_i1, vl);

                            vn_0.re += __riscv_vfmv_f(__riscv_vfredosum(vec_r0, __riscv_vfmv_s_f_f32m1(0, vl), vl));
                            vn_0.im += __riscv_vfmv_f(__riscv_vfredosum(vec_i1, __riscv_vfmv_s_f_f32m1(0, vl), vl));

                            __riscv_vsse32(reinterpret_cast<float*>(a + p - 1), sizeof(float) * 2, vec_r0, vl);
                            __riscv_vsse32(reinterpret_cast<float*>(a + p - 1) + 1, sizeof(float) * 2, vec_i0, vl);
                            __riscv_vsse32(reinterpret_cast<float*>(b + p - 1), sizeof(float) * 2, vec_r1, vl);
                            __riscv_vsse32(reinterpret_cast<float*>(b + p - 1) + 1, sizeof(float) * 2, vec_i1, vl);
                        }
                    }

                    v[0] = vn_0;

                    for( p = 1, k = nx; p <= factor2; p++, k += nx )
                    {
                        Complex<float> s0 = v_0, s1 = v_0;
                        dd = dw_f*p;

                        vl = __riscv_vsetvlmax_e32mf2();
                        auto vec_dd = __riscv_vid_v_u32mf2(vl);
                        vec_dd = __riscv_vmul(vec_dd, dd, vl);
                        vec_dd = __riscv_vremu(vec_dd, tab_size, vl);

                        for( q = 0; q < factor2; q += vl )
                        {
                            vl = __riscv_vsetvl_e32mf2(factor2 - q);

                            auto vec_d = __riscv_vadd(vec_dd, (q + 1) * dd % tab_size, vl);
                            auto vmask = __riscv_vmsgeu(vec_d, tab_size, vl);
                            vec_d = __riscv_vsub_mu(vmask, vec_d, vec_d, tab_size, vl);
                            vec_d = __riscv_vmul(vec_d, sizeof(float) * 2, vl);

                            auto vec_w = __riscv_vloxei32(reinterpret_cast<const float*>(wave), vec_d, vl);
                            auto vec_v = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(a + q), sizeof(float) * 2, vl);
                            auto vec_r0 = __riscv_vfmul(vec_w, vec_v, vl);
                            vec_v = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(b + q) + 1, sizeof(float) * 2, vl);
                            auto vec_r1 = __riscv_vfmul(vec_w, vec_v, vl);

                            vec_w = __riscv_vloxei32(reinterpret_cast<const float*>(wave) + 1, vec_d, vl);
                            vec_v = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(a + q) + 1, sizeof(float) * 2, vl);
                            auto vec_i0 = __riscv_vfmul(vec_w, vec_v, vl);
                            vec_v = __riscv_vlse32_v_f32mf2(reinterpret_cast<const float*>(b + q), sizeof(float) * 2, vl);
                            auto vec_i1 = __riscv_vfmul(vec_w, vec_v, vl);

                            float r0 = __riscv_vfmv_f(__riscv_vfredosum(vec_r0, __riscv_vfmv_s_f_f32m1(0, vl), vl));
                            float i0 = __riscv_vfmv_f(__riscv_vfredosum(vec_i0, __riscv_vfmv_s_f_f32m1(0, vl), vl));
                            float r1 = __riscv_vfmv_f(__riscv_vfredosum(vec_r1, __riscv_vfmv_s_f_f32m1(0, vl), vl));
                            float i1 = __riscv_vfmv_f(__riscv_vfredosum(vec_i1, __riscv_vfmv_s_f_f32m1(0, vl), vl));

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
        float re_scale = scale, im_scale = scale;
        if( isInverse )
            im_scale = -im_scale;

        for( i = 0; i < len; i += vl )
        {
            vl = __riscv_vsetvl_e32m8(len - i);
            auto vec_src_re = __riscv_vlse32_v_f32m8(reinterpret_cast<const float*>(dst + i), sizeof(float) * 2, vl);
            auto vec_src_im = __riscv_vlse32_v_f32m8(reinterpret_cast<const float*>(dst + i) + 1, sizeof(float) * 2, vl);
            vec_src_re = __riscv_vfmul(vec_src_re, re_scale, vl);
            vec_src_im = __riscv_vfmul(vec_src_im, im_scale, vl);
            __riscv_vsse32(reinterpret_cast<float*>(dst + i), sizeof(float) * 2, vec_src_re, vl);
            __riscv_vsse32(reinterpret_cast<float*>(dst + i) + 1, sizeof(float) * 2, vec_src_im, vl);
        }
    }
    else if( isInverse )
    {
        for( i = 0; i < len; i += vl )
        {
            vl = __riscv_vsetvl_e32m8(len - i);
            auto vec_src_im = __riscv_vlse32_v_f32m8(reinterpret_cast<const float*>(dst + i) + 1, sizeof(float) * 2, vl);
            vec_src_im = __riscv_vfneg(vec_src_im, vl);
            __riscv_vsse32(reinterpret_cast<float*>(dst + i) + 1, sizeof(float) * 2, vec_src_im, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int dftOcv64F(const Complex<double>* src, Complex<double>* dst, int nf, int *factors, double scale, int* itab,
                     const Complex<double>* wave, int tab_size, int len, bool isInverse, bool noPermute)
{
    int n = len;
    int f_idx, nx;
    int dw0 = tab_size, dw;
    int i, j, k;
    Complex<double> t;

    int tab_step = tab_size == n ? 1 : tab_size == n*2 ? 2 : tab_size/n;
    int vl;

    // 0. shuffle data
    if( dst != src )
    {
        if( !isInverse )
        {
            for( i = 0; i < n; i += vl )
            {
                vl = __riscv_vsetvl_e32m4(n - i);
                auto vec_itab = __riscv_vlse32_v_u32m4(reinterpret_cast<const uint*>(itab + i * tab_step), sizeof(int) * tab_step, vl);
                vec_itab = __riscv_vmul(vec_itab, sizeof(double) * 2, vl);
                auto vec_src_re = __riscv_vloxei32(reinterpret_cast<const double*>(src), vec_itab, vl);
                vec_itab = __riscv_vadd(vec_itab, sizeof(double), vl);
                auto vec_src_im = __riscv_vloxei32(reinterpret_cast<const double*>(src), vec_itab, vl);
                __riscv_vsse64(reinterpret_cast<double*>(dst + i), sizeof(double) * 2, vec_src_re, vl);
                __riscv_vsse64(reinterpret_cast<double*>(dst + i) + 1, sizeof(double) * 2, vec_src_im, vl);
            }
        }
        else
        {
            for( i = 0; i < n; i += vl )
            {
                vl = __riscv_vsetvl_e32m4(n - i);
                auto vec_itab = __riscv_vlse32_v_u32m4(reinterpret_cast<const uint*>(itab + i * tab_step), sizeof(int) * tab_step, vl);
                vec_itab = __riscv_vmul(vec_itab, sizeof(double) * 2, vl);
                auto vec_src_re = __riscv_vloxei32(reinterpret_cast<const double*>(src), vec_itab, vl);
                vec_itab = __riscv_vadd(vec_itab, sizeof(double), vl);
                auto vec_src_im = __riscv_vloxei32(reinterpret_cast<const double*>(src), vec_itab, vl);
                vec_src_im = __riscv_vfneg(vec_src_im, vl);
                __riscv_vsse64(reinterpret_cast<double*>(dst + i), sizeof(double) * 2, vec_src_re, vl);
                __riscv_vsse64(reinterpret_cast<double*>(dst + i) + 1, sizeof(double) * 2, vec_src_im, vl);
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
                    Complex<double>* dsth = dst + n2;

                    for( i = 0; i < n2; i += 2, itab += tab_step*2 )
                    {
                        j = itab[0];

                        CV_SWAP(dst[i+1], dsth[j], t);
                        if( j > i )
                        {
                            CV_SWAP(dst[i], dst[j], t);
                            CV_SWAP(dsth[i+1], dsth[j+1], t);
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
                        CV_SWAP(dst[i], dst[j], t);
                }
            }
        }

        if( isInverse )
        {
            for( i = 0; i < n; i += vl )
            {
                vl = __riscv_vsetvl_e64m8(n - i);
                auto vec_src_im = __riscv_vlse64_v_f64m8(reinterpret_cast<const double*>(dst + i) + 1, sizeof(double) * 2, vl);
                vec_src_im = __riscv_vfneg(vec_src_im, vl);
                __riscv_vsse64(reinterpret_cast<double*>(dst + i) + 1, sizeof(double) * 2, vec_src_im, vl);
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
                Complex<double> *v0, *v1;
                double r0, i0, r1, i1, r2, i2, r3, i3, r4, i4;

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
                    vl = __riscv_vsetvl_e64m1(nx - j);
                    v0 = dst + i + j;
                    v1 = v0 + nx*2;

                    auto vec_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v1), sizeof(double) * 2, vl);
                    auto vec_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v1) + 1, sizeof(double) * 2, vl);
                    auto vec_w_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0), sizeof(double) * dw0 * 2, vl);
                    auto vec_w_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0) + 1, sizeof(double) * dw0 * 2, vl);
                    auto vec_r0 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);
                    auto vec_i0 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);

                    vec_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v1 + nx), sizeof(double) * 2, vl);
                    vec_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v1 + nx) + 1, sizeof(double) * 2, vl);
                    vec_w_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0 * 3), sizeof(double) * dw0 * 6, vl);
                    vec_w_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0 * 3) + 1, sizeof(double) * dw0 * 6, vl);
                    auto vec_r3 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);
                    auto vec_i3 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);

                    auto vec_r1 = __riscv_vfadd(vec_i0, vec_i3, vl);
                    auto vec_i1 = __riscv_vfadd(vec_r0, vec_r3, vl);
                    vec_r3 = __riscv_vfsub(vec_r0, vec_r3, vl);
                    vec_i3 = __riscv_vfsub(vec_i3, vec_i0, vl);
                    auto vec_r4 = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v0), sizeof(double) * 2, vl);
                    auto vec_i4 = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v0) + 1, sizeof(double) * 2, vl);

                    vec_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v0 + nx), sizeof(double) * 2, vl);
                    vec_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v0 + nx) + 1, sizeof(double) * 2, vl);
                    vec_w_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0 * 2), sizeof(double) * dw0 * 4, vl);
                    vec_w_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0 * 2) + 1, sizeof(double) * dw0 * 4, vl);
                    auto vec_r2 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i2 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_r0 = __riscv_vfadd(vec_r4, vec_r2, vl);
                    vec_i0 = __riscv_vfadd(vec_i4, vec_i2, vl);

                    __riscv_vsse64(reinterpret_cast<double*>(v0), sizeof(double) * 2, __riscv_vfadd(vec_r0, vec_r1, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v0) + 1, sizeof(double) * 2, __riscv_vfadd(vec_i0, vec_i1, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v1), sizeof(double) * 2, __riscv_vfsub(vec_r0, vec_r1, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v1) + 1, sizeof(double) * 2, __riscv_vfsub(vec_i0, vec_i1, vl), vl);

                    vec_r2 = __riscv_vfsub(vec_r4, vec_r2, vl);
                    vec_i2 = __riscv_vfsub(vec_i4, vec_i2, vl);

                    __riscv_vsse64(reinterpret_cast<double*>(v0 + nx), sizeof(double) * 2, __riscv_vfadd(vec_r2, vec_r3, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v0 + nx) + 1, sizeof(double) * 2, __riscv_vfadd(vec_i2, vec_i3, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v1 + nx), sizeof(double) * 2, __riscv_vfsub(vec_r2, vec_r3, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v1 + nx) + 1, sizeof(double) * 2, __riscv_vfsub(vec_i2, vec_i3, vl), vl);
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
                Complex<double>* v = dst + i;
                double r0 = v[0].re + v[nx].re;
                double i0 = v[0].im + v[nx].im;
                double r1 = v[0].re - v[nx].re;
                double i1 = v[0].im - v[nx].im;
                v[0].re = r0; v[0].im = i0;
                v[nx].re = r1; v[nx].im = i1;

                for( j = 1; j < nx; j += vl )
                {
                    vl = __riscv_vsetvl_e64m1(nx - j);
                    v = dst + i + j;

                    auto vec_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + nx), sizeof(double) * 2, vl);
                    auto vec_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + nx) + 1, sizeof(double) * 2, vl);
                    auto vec_w_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0), sizeof(double) * dw0 * 2, vl);
                    auto vec_w_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0) + 1, sizeof(double) * dw0 * 2, vl);

                    auto vec_r1 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i1 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);
                    auto vec_r0 = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v), sizeof(double) * 2, vl);
                    auto vec_i0 = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v) + 1, sizeof(double) * 2, vl);

                    __riscv_vsse64(reinterpret_cast<double*>(v), sizeof(double) * 2, __riscv_vfadd(vec_r0, vec_r1, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v) + 1, sizeof(double) * 2, __riscv_vfadd(vec_i0, vec_i1, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v + nx), sizeof(double) * 2, __riscv_vfsub(vec_r0, vec_r1, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v + nx) + 1, sizeof(double) * 2, __riscv_vfsub(vec_i0, vec_i1, vl), vl);
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
            const double sin_120 = 0.86602540378443864676372317075294;
            for( i = 0; i < len; i += n )
            {
                Complex<double>* v = dst + i;
                double r1 = v[nx].re + v[nx*2].re;
                double i1 = v[nx].im + v[nx*2].im;
                double r0 = v[0].re;
                double i0 = v[0].im;
                double r2 = sin_120*(v[nx].im - v[nx*2].im);
                double i2 = sin_120*(v[nx*2].re - v[nx].re);
                v[0].re = r0 + r1; v[0].im = i0 + i1;
                r0 -= (double)0.5*r1; i0 -= (double)0.5*i1;
                v[nx].re = r0 + r2; v[nx].im = i0 + i2;
                v[nx*2].re = r0 - r2; v[nx*2].im = i0 - i2;

                for( j = 1; j < nx; j += vl )
                {
                    vl = __riscv_vsetvl_e64m1(nx - j);
                    v = dst + i + j;

                    auto vec_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + nx), sizeof(double) * 2, vl);
                    auto vec_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + nx) + 1, sizeof(double) * 2, vl);
                    auto vec_w_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0), sizeof(double) * dw0 * 2, vl);
                    auto vec_w_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0) + 1, sizeof(double) * dw0 * 2, vl);
                    auto vec_r0 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i0 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + nx * 2), sizeof(double) * 2, vl);
                    vec_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + nx * 2) + 1, sizeof(double) * 2, vl);
                    vec_w_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0 * 2), sizeof(double) * dw0 * 4, vl);
                    vec_w_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0 * 2) + 1, sizeof(double) * dw0 * 4, vl);
                    auto vec_r2 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);
                    auto vec_i2 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);

                    auto vec_r1 = __riscv_vfadd(vec_r0, vec_i2, vl);
                    auto vec_i1 = __riscv_vfadd(vec_i0, vec_r2, vl);

                    vec_r2 = __riscv_vfmul(__riscv_vfsub(vec_i0, vec_r2, vl), sin_120, vl);
                    vec_i2 = __riscv_vfmul(__riscv_vfsub(vec_i2, vec_r0, vl), sin_120, vl);
                    vec_r0 = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v), sizeof(double) * 2, vl);
                    vec_i0 = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v) + 1, sizeof(double) * 2, vl);

                    __riscv_vsse64(reinterpret_cast<double*>(v), sizeof(double) * 2, __riscv_vfadd(vec_r0, vec_r1, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v) + 1, sizeof(double) * 2, __riscv_vfadd(vec_i0, vec_i1, vl), vl);
                    vec_r0 = __riscv_vfsub(vec_r0, __riscv_vfmul(vec_r1, 0.5, vl), vl);
                    vec_i0 = __riscv_vfsub(vec_i0, __riscv_vfmul(vec_i1, 0.5, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v + nx), sizeof(double) * 2, __riscv_vfadd(vec_r0, vec_r2, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v + nx) + 1, sizeof(double) * 2, __riscv_vfadd(vec_i0, vec_i2, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v + nx * 2), sizeof(double) * 2, __riscv_vfsub(vec_r0, vec_r2, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v + nx * 2) + 1, sizeof(double) * 2, __riscv_vfsub(vec_i0, vec_i2, vl), vl);
                }
            }
        }
        else if( factor == 5 )
        {
            const double fft5_2 = 0.559016994374947424102293417182819;
            const double fft5_3 = -0.951056516295153572116439333379382;
            const double fft5_4 = -1.538841768587626701285145288018455;
            const double fft5_5 = 0.363271264002680442947733378740309;
            for( i = 0; i < len; i += n )
            {
                for( j = 0; j < nx; j += vl )
                {
                    vl = __riscv_vsetvl_e64m1(nx - j);
                    Complex<double>* v0 = dst + i + j;
                    Complex<double>* v1 = v0 + nx*2;
                    Complex<double>* v2 = v1 + nx*2;

                    auto vec_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v0 + nx), sizeof(double) * 2, vl);
                    auto vec_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v0 + nx) + 1, sizeof(double) * 2, vl);
                    auto vec_w_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0), sizeof(double) * dw0 * 2, vl);
                    auto vec_w_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0) + 1, sizeof(double) * dw0 * 2, vl);
                    auto vec_r3 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i3 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v2), sizeof(double) * 2, vl);
                    vec_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v2) + 1, sizeof(double) * 2, vl);
                    vec_w_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0 * 4), sizeof(double) * dw0 * 8, vl);
                    vec_w_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0 * 4) + 1, sizeof(double) * dw0 * 8, vl);
                    auto vec_r2 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i2 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    auto vec_r1 = __riscv_vfadd(vec_r3, vec_r2, vl);
                    auto vec_i1 = __riscv_vfadd(vec_i3, vec_i2, vl);
                    vec_r3 = __riscv_vfsub(vec_r3, vec_r2, vl);
                    vec_i3 = __riscv_vfsub(vec_i3, vec_i2, vl);

                    vec_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v1 + nx), sizeof(double) * 2, vl);
                    vec_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v1 + nx) + 1, sizeof(double) * 2, vl);
                    vec_w_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0 * 3), sizeof(double) * dw0 * 6, vl);
                    vec_w_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0 * 3) + 1, sizeof(double) * dw0 * 6, vl);
                    auto vec_r4 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i4 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v1), sizeof(double) * 2, vl);
                    vec_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v1) + 1, sizeof(double) * 2, vl);
                    vec_w_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0 * 2), sizeof(double) * dw0 * 4, vl);
                    vec_w_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + j * dw0 * 2) + 1, sizeof(double) * dw0 * 4, vl);
                    auto vec_r0 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                    auto vec_i0 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                    vec_r2 = __riscv_vfadd(vec_r4, vec_r0, vl);
                    vec_i2 = __riscv_vfadd(vec_i4, vec_i0, vl);
                    vec_r4 = __riscv_vfsub(vec_r4, vec_r0, vl);
                    vec_i4 = __riscv_vfsub(vec_i4, vec_i0, vl);

                    vec_r0 = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v0), sizeof(double) * 2, vl);
                    vec_i0 = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v0) + 1, sizeof(double) * 2, vl);
                    auto vec_r5 = __riscv_vfadd(vec_r1, vec_r2, vl);
                    auto vec_i5 = __riscv_vfadd(vec_i1, vec_i2, vl);

                    __riscv_vsse64(reinterpret_cast<double*>(v0), sizeof(double) * 2, __riscv_vfadd(vec_r0, vec_r5, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v0) + 1, sizeof(double) * 2, __riscv_vfadd(vec_i0, vec_i5, vl), vl);

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

                    __riscv_vsse64(reinterpret_cast<double*>(v0 + nx), sizeof(double) * 2, __riscv_vfadd(vec_r3, vec_r2, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v0 + nx) + 1, sizeof(double) * 2, __riscv_vfadd(vec_i3, vec_i2, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v2), sizeof(double) * 2, __riscv_vfsub(vec_r3, vec_r2, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v2) + 1, sizeof(double) * 2, __riscv_vfsub(vec_i3, vec_i2, vl), vl);

                    vec_r0 = __riscv_vfsub(vec_r0, vec_r1, vl);
                    vec_i0 = __riscv_vfsub(vec_i0, vec_i1, vl);

                    __riscv_vsse64(reinterpret_cast<double*>(v1), sizeof(double) * 2, __riscv_vfadd(vec_r0, vec_r5, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v1) + 1, sizeof(double) * 2, __riscv_vfadd(vec_i0, vec_i5, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v1 + nx), sizeof(double) * 2, __riscv_vfsub(vec_r0, vec_r5, vl), vl);
                    __riscv_vsse64(reinterpret_cast<double*>(v1 + nx) + 1, sizeof(double) * 2, __riscv_vfsub(vec_i0, vec_i5, vl), vl);
                }
            }
        }
        else
        {
            // radix-"factor" - an odd number
            int p, q, factor2 = (factor - 1)/2;
            int dd, dw_f = tab_size/factor;
            AutoBuffer<Complex<double> > buf(factor2 * 2);
            Complex<double>* a = buf.data();
            Complex<double>* b = a + factor2;

            for( i = 0; i < len; i += n )
            {
                for( j = 0, dw = 0; j < nx; j++, dw += dw0 )
                {
                    Complex<double>* v = dst + i + j;
                    Complex<double> v_0 = v[0];
                    Complex<double> vn_0 = v_0;

                    if( j == 0 )
                    {
                        for( p = 1; p <= factor2; p += vl )
                        {
                            vl = __riscv_vsetvl_e64m1(factor2 + 1 - p);

                            auto vec_a = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + nx * p), sizeof(double) * nx * 2, vl);
                            auto vec_b = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + n - nx * p), (long)sizeof(double) * nx * -2, vl);
                            auto vec_r0 = __riscv_vfadd(vec_a, vec_b, vl);
                            auto vec_r1 = __riscv_vfsub(vec_a, vec_b, vl);

                            vec_a = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + nx * p) + 1, sizeof(double) * nx * 2, vl);
                            vec_b = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + n - nx * p) + 1, (long)sizeof(double) * nx * -2, vl);
                            auto vec_i0 = __riscv_vfsub(vec_a, vec_b, vl);
                            auto vec_i1 = __riscv_vfadd(vec_a, vec_b, vl);

                            vn_0.re += __riscv_vfmv_f(__riscv_vfredosum(vec_r0, __riscv_vfmv_s_f_f64m1(0, vl), vl));
                            vn_0.im += __riscv_vfmv_f(__riscv_vfredosum(vec_i1, __riscv_vfmv_s_f_f64m1(0, vl), vl));

                            __riscv_vsse64(reinterpret_cast<double*>(a + p - 1), sizeof(double) * 2, vec_r0, vl);
                            __riscv_vsse64(reinterpret_cast<double*>(a + p - 1) + 1, sizeof(double) * 2, vec_i0, vl);
                            __riscv_vsse64(reinterpret_cast<double*>(b + p - 1), sizeof(double) * 2, vec_r1, vl);
                            __riscv_vsse64(reinterpret_cast<double*>(b + p - 1) + 1, sizeof(double) * 2, vec_i1, vl);
                        }
                    }
                    else
                    {
                        const Complex<double>* wave_ = wave + dw*factor;

                        for( p = 1; p <= factor2; p += vl )
                        {
                            vl = __riscv_vsetvl_e64m1(factor2 + 1 - p);

                            auto vec_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + nx * p), sizeof(double) * nx * 2, vl);
                            auto vec_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + nx * p) + 1, sizeof(double) * nx * 2, vl);
                            auto vec_w_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + p * dw), sizeof(double) * dw * 2, vl);
                            auto vec_w_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave + p * dw) + 1, sizeof(double) * dw * 2, vl);
                            auto vec_r2 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                            auto vec_i2 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                            vec_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + n - nx * p), (long)sizeof(double) * nx * -2, vl);
                            vec_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(v + n - nx * p) + 1, (long)sizeof(double) * nx * -2, vl);
                            vec_w_re = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave_ - p * dw), (long)sizeof(double) * dw * -2, vl);
                            vec_w_im = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(wave_ - p * dw) + 1, (long)sizeof(double) * dw * -2, vl);
                            auto vec_r1 = __riscv_vfsub(__riscv_vfmul(vec_re, vec_w_re, vl), __riscv_vfmul(vec_im, vec_w_im, vl), vl);
                            auto vec_i1 = __riscv_vfadd(__riscv_vfmul(vec_re, vec_w_im, vl), __riscv_vfmul(vec_im, vec_w_re, vl), vl);

                            auto vec_r0 = __riscv_vfadd(vec_r2, vec_r1, vl);
                            auto vec_i0 = __riscv_vfsub(vec_i2, vec_i1, vl);
                            vec_r1 = __riscv_vfsub(vec_r2, vec_r1, vl);
                            vec_i1 = __riscv_vfadd(vec_i2, vec_i1, vl);

                            vn_0.re += __riscv_vfmv_f(__riscv_vfredosum(vec_r0, __riscv_vfmv_s_f_f64m1(0, vl), vl));
                            vn_0.im += __riscv_vfmv_f(__riscv_vfredosum(vec_i1, __riscv_vfmv_s_f_f64m1(0, vl), vl));

                            __riscv_vsse64(reinterpret_cast<double*>(a + p - 1), sizeof(double) * 2, vec_r0, vl);
                            __riscv_vsse64(reinterpret_cast<double*>(a + p - 1) + 1, sizeof(double) * 2, vec_i0, vl);
                            __riscv_vsse64(reinterpret_cast<double*>(b + p - 1), sizeof(double) * 2, vec_r1, vl);
                            __riscv_vsse64(reinterpret_cast<double*>(b + p - 1) + 1, sizeof(double) * 2, vec_i1, vl);
                        }
                    }

                    v[0] = vn_0;

                    for( p = 1, k = nx; p <= factor2; p++, k += nx )
                    {
                        Complex<double> s0 = v_0, s1 = v_0;
                        dd = dw_f*p;

                        vl = __riscv_vsetvlmax_e64m1();
                        auto vec_dd = __riscv_vid_v_u32mf2(vl);
                        vec_dd = __riscv_vmul(vec_dd, dd, vl);
                        vec_dd = __riscv_vremu(vec_dd, tab_size, vl);

                        for( q = 0; q < factor2; q += vl )
                        {
                            vl = __riscv_vsetvl_e64m1(factor2 - q);

                            auto vec_d = __riscv_vadd(vec_dd, (q + 1) * dd % tab_size, vl);
                            auto vmask = __riscv_vmsgeu(vec_d, tab_size, vl);
                            vec_d = __riscv_vsub_mu(vmask, vec_d, vec_d, tab_size, vl);
                            vec_d = __riscv_vmul(vec_d, sizeof(double) * 2, vl);

                            auto vec_w = __riscv_vloxei32(reinterpret_cast<const double*>(wave), vec_d, vl);
                            auto vec_v = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(a + q), sizeof(double) * 2, vl);
                            auto vec_r0 = __riscv_vfmul(vec_w, vec_v, vl);
                            vec_v = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(b + q) + 1, sizeof(double) * 2, vl);
                            auto vec_r1 = __riscv_vfmul(vec_w, vec_v, vl);

                            vec_w = __riscv_vloxei32(reinterpret_cast<const double*>(wave) + 1, vec_d, vl);
                            vec_v = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(a + q) + 1, sizeof(double) * 2, vl);
                            auto vec_i0 = __riscv_vfmul(vec_w, vec_v, vl);
                            vec_v = __riscv_vlse64_v_f64m1(reinterpret_cast<const double*>(b + q), sizeof(double) * 2, vl);
                            auto vec_i1 = __riscv_vfmul(vec_w, vec_v, vl);

                            double r0 = __riscv_vfmv_f(__riscv_vfredosum(vec_r0, __riscv_vfmv_s_f_f64m1(0, vl), vl));
                            double i0 = __riscv_vfmv_f(__riscv_vfredosum(vec_i0, __riscv_vfmv_s_f_f64m1(0, vl), vl));
                            double r1 = __riscv_vfmv_f(__riscv_vfredosum(vec_r1, __riscv_vfmv_s_f_f64m1(0, vl), vl));
                            double i1 = __riscv_vfmv_f(__riscv_vfredosum(vec_i1, __riscv_vfmv_s_f_f64m1(0, vl), vl));

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
        double re_scale = scale, im_scale = scale;
        if( isInverse )
            im_scale = -im_scale;

        for( i = 0; i < len; i += vl )
        {
            vl = __riscv_vsetvl_e64m8(len - i);
            auto vec_src_re = __riscv_vlse64_v_f64m8(reinterpret_cast<const double*>(dst + i), sizeof(double) * 2, vl);
            auto vec_src_im = __riscv_vlse64_v_f64m8(reinterpret_cast<const double*>(dst + i) + 1, sizeof(double) * 2, vl);
            vec_src_re = __riscv_vfmul(vec_src_re, re_scale, vl);
            vec_src_im = __riscv_vfmul(vec_src_im, im_scale, vl);
            __riscv_vsse64(reinterpret_cast<double*>(dst + i), sizeof(double) * 2, vec_src_re, vl);
            __riscv_vsse64(reinterpret_cast<double*>(dst + i) + 1, sizeof(double) * 2, vec_src_im, vl);
        }
    }
    else if( isInverse )
    {
        for( i = 0; i < len; i += vl )
        {
            vl = __riscv_vsetvl_e64m8(len - i);
            auto vec_src_im = __riscv_vlse64_v_f64m8(reinterpret_cast<const double*>(dst + i) + 1, sizeof(double) * 2, vl);
            vec_src_im = __riscv_vfneg(vec_src_im, vl);
            __riscv_vsse64(reinterpret_cast<double*>(dst + i) + 1, sizeof(double) * 2, vec_src_im, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int dftOcv(const uchar* src, uchar* dst, int depth, int nf, int *factors, double scale, int* itab, void* wave,
                  int tab_size, int n, bool isInverse, bool noPermute)
{
    switch (depth)
    {
    case CV_32F:
        return dftOcv32F(reinterpret_cast<const Complex<float>*>(src), reinterpret_cast<Complex<float>*>(dst), nf, factors, scale,
                         itab, reinterpret_cast<const Complex<float>*>(wave), tab_size, n, isInverse, noPermute);
    case CV_64F:
        return dftOcv64F(reinterpret_cast<const Complex<double>*>(src), reinterpret_cast<Complex<double>*>(dst), nf, factors, scale,
                         itab, reinterpret_cast<const Complex<double>*>(wave), tab_size, n, isInverse, noPermute);
    }
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

}}

#endif
