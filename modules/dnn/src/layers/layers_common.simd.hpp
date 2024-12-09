/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void fastGEMM1T( const float* vec, const float* weights,
                 size_t wstep, const float* bias,
                 float* dst, int nvecs, int vecsize );
void fastGEMM( const float* aptr, size_t astep, const float* bptr,
               size_t bstep, float* cptr, size_t cstep,
               int ma, int na, int nb );

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_AVX

#if !CV_FMA3 // AVX workaround
#undef _mm256_fmadd_ps
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))
#endif

// Used to generate the mask used when calculating tails
static const uint32_t tailMaskArray[15] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    0xffffffffUL, 0xffffffffUL, 0xffffffffUL, 0xffffffffUL, 0xffffffffUL, 0xffffffffUL, 0xffffffffUL
};

// dst = vec * weights^t + bias
// Requires that vecsize is at least 8 or equal to 0 to avoid memory access problems. Does not require alignment.
void fastGEMM1T( const float* vec, const float* weights,
                 size_t wstep, const float* bias,
                 float* dst, int nvecs, int vecsize )
{
    int i = 0;

    CV_Assert(vecsize >= 8 || vecsize == 0);

    __m256 tailMask = _mm256_loadu_ps(reinterpret_cast<const float*>(tailMaskArray) + (vecsize % 8));

    for( ; i <= nvecs - 8; i += 8 )
    {
        const float* wptr = weights + i*wstep;
        __m256 vs0 = _mm256_setzero_ps(), vs1 = _mm256_setzero_ps(),
               vs2 = _mm256_setzero_ps(), vs3 = _mm256_setzero_ps(),
               vs4 = _mm256_setzero_ps(), vs5 = _mm256_setzero_ps(),
               vs6 = _mm256_setzero_ps(), vs7 = _mm256_setzero_ps();

        int k = 0;
        for( ; k <= vecsize-8; k += 8, wptr += 8 )
        {
            __m256 v = _mm256_loadu_ps(vec + k);

            vs0 = _mm256_fmadd_ps(_mm256_loadu_ps(wptr), v, vs0);
            vs1 = _mm256_fmadd_ps(_mm256_loadu_ps(wptr + wstep), v, vs1);
            vs2 = _mm256_fmadd_ps(_mm256_loadu_ps(wptr + wstep*2), v, vs2);
            vs3 = _mm256_fmadd_ps(_mm256_loadu_ps(wptr + wstep*3), v, vs3);
            vs4 = _mm256_fmadd_ps(_mm256_loadu_ps(wptr + wstep*4), v, vs4);
            vs5 = _mm256_fmadd_ps(_mm256_loadu_ps(wptr + wstep*5), v, vs5);
            vs6 = _mm256_fmadd_ps(_mm256_loadu_ps(wptr + wstep*6), v, vs6);
            vs7 = _mm256_fmadd_ps(_mm256_loadu_ps(wptr + wstep*7), v, vs7);
        }

        if (k != vecsize) {
            // Tail
            k = vecsize - 8;
            wptr = weights + i * wstep + k;
            __m256 v = _mm256_loadu_ps(vec + k);
            v = _mm256_and_ps(v, tailMask);

            vs0 = _mm256_fmadd_ps(_mm256_and_ps(_mm256_loadu_ps(wptr), tailMask), v, vs0);
            vs1 = _mm256_fmadd_ps(_mm256_and_ps(_mm256_loadu_ps(wptr + wstep), tailMask), v, vs1);
            vs2 = _mm256_fmadd_ps(_mm256_and_ps(_mm256_loadu_ps(wptr + wstep * 2), tailMask), v, vs2);
            vs3 = _mm256_fmadd_ps(_mm256_and_ps(_mm256_loadu_ps(wptr + wstep * 3), tailMask), v, vs3);
            vs4 = _mm256_fmadd_ps(_mm256_and_ps(_mm256_loadu_ps(wptr + wstep * 4), tailMask), v, vs4);
            vs5 = _mm256_fmadd_ps(_mm256_and_ps(_mm256_loadu_ps(wptr + wstep * 5), tailMask), v, vs5);
            vs6 = _mm256_fmadd_ps(_mm256_and_ps(_mm256_loadu_ps(wptr + wstep * 6), tailMask), v, vs6);
            vs7 = _mm256_fmadd_ps(_mm256_and_ps(_mm256_loadu_ps(wptr + wstep * 7), tailMask), v, vs7);
        }

        __m256 s0 = _mm256_hadd_ps(_mm256_hadd_ps(vs0, vs1), _mm256_hadd_ps(vs2, vs3));
        __m256 s1 = _mm256_hadd_ps(_mm256_hadd_ps(vs4, vs5), _mm256_hadd_ps(vs6, vs7));

        s0 = _mm256_add_ps(s0, _mm256_permute2f128_ps(s0, s0, 1));
        s1 = _mm256_add_ps(s1, _mm256_permute2f128_ps(s1, s1, 1));

        s0 = _mm256_add_ps(s0, _mm256_castps128_ps256(_mm_loadu_ps(bias + i)));
        s1 = _mm256_add_ps(s1, _mm256_castps128_ps256(_mm_loadu_ps(bias + i + 4)));

        _mm_storeu_ps(dst + i, _mm256_castps256_ps128(s0));
        _mm_storeu_ps(dst + i + 4, _mm256_castps256_ps128(s1));
    }

    float temp = 0.f;
    for( ; i < nvecs; i++ )
    {
        const float* wptr = weights + i*wstep;
        __m256 vs0 = _mm256_setzero_ps();

        int k = 0;
        for( ; k <= vecsize-8; k += 8, wptr += 8 )
        {
            __m256 v = _mm256_loadu_ps(vec + k);
            vs0 = _mm256_fmadd_ps(_mm256_loadu_ps(wptr), v, vs0);
        }

        if (k != vecsize) {
            // Tail
            k = vecsize - 8;
            wptr = weights + i * wstep + k;
            __m256 v = _mm256_loadu_ps(vec + k);
            v = _mm256_and_ps(v, tailMask);
            vs0 = _mm256_fmadd_ps(_mm256_and_ps(_mm256_loadu_ps(wptr), tailMask), v, vs0);
        }

        __m256 s0 = _mm256_hadd_ps(_mm256_hadd_ps(vs0, vs0), vs0);
        s0 = _mm256_add_ps(s0, _mm256_permute2f128_ps(s0, s0, 1));
        _mm_store_ss(&temp, _mm256_castps256_ps128(s0));
        dst[i] = temp + bias[i];
    }

    _mm256_zeroupper();
}


void fastGEMM( const float* aptr, size_t astep, const float* bptr,
               size_t bstep, float* cptr, size_t cstep,
               int ma, int na, int nb )
{
    int n = 0;

#if CV_AVX512_SKX // AVX512VL is necessary to avoid register spilling
    for( ; n <= nb - 32; n += 32 )
    {
        for( int m = 0; m < ma; m += 4 )
        {
            const float* aptr0 = aptr + astep*m;
            const float* aptr1 = aptr + astep*std::min(m+1, ma-1);
            const float* aptr2 = aptr + astep*std::min(m+2, ma-1);
            const float* aptr3 = aptr + astep*std::min(m+3, ma-1);

            float* cptr0 = cptr + cstep*m;
            float* cptr1 = cptr + cstep*std::min(m+1, ma-1);
            float* cptr2 = cptr + cstep*std::min(m+2, ma-1);
            float* cptr3 = cptr + cstep*std::min(m+3, ma-1);

            __m512 d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
            __m512 d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
            __m512 d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
            __m512 d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();

            for( int k = 0; k < na; k++ )
            {
                __m512 a0 = _mm512_set1_ps(aptr0[k]);
                __m512 a1 = _mm512_set1_ps(aptr1[k]);
                __m512 a2 = _mm512_set1_ps(aptr2[k]);
                __m512 a3 = _mm512_set1_ps(aptr3[k]);
                __m512 b0 = _mm512_loadu_ps(bptr + k*bstep + n);
                __m512 b1 = _mm512_loadu_ps(bptr + k*bstep + n + 16);
                d00 = _mm512_fmadd_ps(a0, b0, d00);
                d01 = _mm512_fmadd_ps(a0, b1, d01);
                d10 = _mm512_fmadd_ps(a1, b0, d10);
                d11 = _mm512_fmadd_ps(a1, b1, d11);
                d20 = _mm512_fmadd_ps(a2, b0, d20);
                d21 = _mm512_fmadd_ps(a2, b1, d21);
                d30 = _mm512_fmadd_ps(a3, b0, d30);
                d31 = _mm512_fmadd_ps(a3, b1, d31);
            }

            _mm512_storeu_ps(cptr0 + n, d00);
            _mm512_storeu_ps(cptr0 + n + 16, d01);
            _mm512_storeu_ps(cptr1 + n, d10);
            _mm512_storeu_ps(cptr1 + n + 16, d11);
            _mm512_storeu_ps(cptr2 + n, d20);
            _mm512_storeu_ps(cptr2 + n + 16, d21);
            _mm512_storeu_ps(cptr3 + n, d30);
            _mm512_storeu_ps(cptr3 + n + 16, d31);
        }
    }
#endif

    for( ; n <= nb - 16; n += 16 )
    {
        for( int m = 0; m < ma; m += 4 )
        {
            const float* aptr0 = aptr + astep*m;
            const float* aptr1 = aptr + astep*std::min(m+1, ma-1);
            const float* aptr2 = aptr + astep*std::min(m+2, ma-1);
            const float* aptr3 = aptr + astep*std::min(m+3, ma-1);

            float* cptr0 = cptr + cstep*m;
            float* cptr1 = cptr + cstep*std::min(m+1, ma-1);
            float* cptr2 = cptr + cstep*std::min(m+2, ma-1);
            float* cptr3 = cptr + cstep*std::min(m+3, ma-1);

            __m256 d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
            __m256 d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
            __m256 d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
            __m256 d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();

            for( int k = 0; k < na; k++ )
            {
                __m256 a0 = _mm256_set1_ps(aptr0[k]);
                __m256 a1 = _mm256_set1_ps(aptr1[k]);
                __m256 a2 = _mm256_set1_ps(aptr2[k]);
                __m256 a3 = _mm256_set1_ps(aptr3[k]);
                __m256 b0 = _mm256_loadu_ps(bptr + k*bstep + n);
                __m256 b1 = _mm256_loadu_ps(bptr + k*bstep + n + 8);
                d00 = _mm256_fmadd_ps(a0, b0, d00);
                d01 = _mm256_fmadd_ps(a0, b1, d01);
                d10 = _mm256_fmadd_ps(a1, b0, d10);
                d11 = _mm256_fmadd_ps(a1, b1, d11);
                d20 = _mm256_fmadd_ps(a2, b0, d20);
                d21 = _mm256_fmadd_ps(a2, b1, d21);
                d30 = _mm256_fmadd_ps(a3, b0, d30);
                d31 = _mm256_fmadd_ps(a3, b1, d31);
            }

            _mm256_storeu_ps(cptr0 + n, d00);
            _mm256_storeu_ps(cptr0 + n + 8, d01);
            _mm256_storeu_ps(cptr1 + n, d10);
            _mm256_storeu_ps(cptr1 + n + 8, d11);
            _mm256_storeu_ps(cptr2 + n, d20);
            _mm256_storeu_ps(cptr2 + n + 8, d21);
            _mm256_storeu_ps(cptr3 + n, d30);
            _mm256_storeu_ps(cptr3 + n + 8, d31);
        }
    }

    for( ; n < nb; n++ )
    {
        for( int m = 0; m < ma; m++ )
        {
            const float* aptr0 = aptr + astep*m;
            float* cptr0 = cptr + cstep*m;
            float d0 = 0.f;

            for( int k = 0; k < na; k++ )
                d0 += aptr0[k]*bptr[k*bstep + n];

            cptr0[n] = d0;
        }
    }
    _mm256_zeroupper();
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_RVV

void fastGEMM( const float* aptr, size_t astep, const float* bptr,
               size_t bstep, float* cptr, size_t cstep,
               int ma, int na, int nb )
{
    int avl = nb, vl;
    for(int n = 0; n < nb; n += vl, avl -= vl)
    {
        vl = __riscv_vsetvl_e32m4(avl);
        for( int m = 0; m < ma; m += 7 )
        {
            const float* aptr0 = aptr + astep*m;
            const float* aptr1 = aptr + astep*std::min(m+1, ma-1);
            const float* aptr2 = aptr + astep*std::min(m+2, ma-1);
            const float* aptr3 = aptr + astep*std::min(m+3, ma-1);
            const float* aptr4 = aptr + astep*std::min(m+4, ma-1);
            const float* aptr5 = aptr + astep*std::min(m+5, ma-1);
            const float* aptr6 = aptr + astep*std::min(m+6, ma-1);

            float* cptr0 = cptr + cstep*m;
            float* cptr1 = cptr + cstep*std::min(m+1, ma-1);
            float* cptr2 = cptr + cstep*std::min(m+2, ma-1);
            float* cptr3 = cptr + cstep*std::min(m+3, ma-1);
            float* cptr4 = cptr + cstep*std::min(m+4, ma-1);
            float* cptr5 = cptr + cstep*std::min(m+5, ma-1);
            float* cptr6 = cptr + cstep*std::min(m+6, ma-1);

            vfloat32m4_t d0 = __riscv_vfmv_v_f_f32m4(0, vl);
            vfloat32m4_t d1 = __riscv_vfmv_v_f_f32m4(0, vl);
            vfloat32m4_t d2 = __riscv_vfmv_v_f_f32m4(0, vl);
            vfloat32m4_t d3 = __riscv_vfmv_v_f_f32m4(0, vl);
            vfloat32m4_t d4 = __riscv_vfmv_v_f_f32m4(0, vl);
            vfloat32m4_t d5 = __riscv_vfmv_v_f_f32m4(0, vl);
            vfloat32m4_t d6 = __riscv_vfmv_v_f_f32m4(0, vl);

            for( int k = 0; k < na; k++ )
            {
                float a0 = aptr0[k];
                float a1 = aptr1[k];
                float a2 = aptr2[k];
                float a3 = aptr3[k];
                float a4 = aptr4[k];
                float a5 = aptr5[k];
                float a6 = aptr6[k];

                vfloat32m4_t b = __riscv_vle32_v_f32m4(bptr + k*bstep + n, vl);
                d0 = __riscv_vfmacc_vf_f32m4(d0, a0, b, vl);
                d1 = __riscv_vfmacc_vf_f32m4(d1, a1, b, vl);
                d2 = __riscv_vfmacc_vf_f32m4(d2, a2, b, vl);
                d3 = __riscv_vfmacc_vf_f32m4(d3, a3, b, vl);
                d4 = __riscv_vfmacc_vf_f32m4(d4, a4, b, vl);
                d5 = __riscv_vfmacc_vf_f32m4(d5, a5, b, vl);
                d6 = __riscv_vfmacc_vf_f32m4(d6, a6, b, vl);
            }
            __riscv_vse32_v_f32m4(cptr0 + n, d0, vl);
            __riscv_vse32_v_f32m4(cptr1 + n, d1, vl);
            __riscv_vse32_v_f32m4(cptr2 + n, d2, vl);
            __riscv_vse32_v_f32m4(cptr3 + n, d3, vl);
            __riscv_vse32_v_f32m4(cptr4 + n, d4, vl);
            __riscv_vse32_v_f32m4(cptr5 + n, d5, vl);
            __riscv_vse32_v_f32m4(cptr6 + n, d6, vl);
        }
    }
}

void fastGEMM1T( const float* vec, const float* weights,
                 size_t wstep, const float* bias,
                 float* dst, int nvecs, int vecsize )
{
    const int vlm2 = __riscv_vsetvlmax_e32m2();
    int i = 0;
    for( ; i <= nvecs - 15; i += 15 )
    {
        const float* wptr = weights + i*wstep;
        vfloat32m2_t
               vs0 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs1 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs2 = __riscv_vfmv_v_f_f32m2(0, vlm2),
               vs3 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs4 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs5 = __riscv_vfmv_v_f_f32m2(0, vlm2),
               vs6 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs7 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs8 = __riscv_vfmv_v_f_f32m2(0, vlm2),
               vs9 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs10 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs11 = __riscv_vfmv_v_f_f32m2(0, vlm2),
               vs12 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs13 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs14 = __riscv_vfmv_v_f_f32m2(0, vlm2);
        int avl = vecsize, vl;
        for(int k = 0 ; k < vecsize; k += vl, wptr += vl, avl -= vl)
        {
            vl = __riscv_vsetvl_e32m2(avl);
            vfloat32m2_t v = __riscv_vle32_v_f32m2(vec + k, vl);
            vs0 = __riscv_vfmacc_vv_f32m2(vs0, __riscv_vle32_v_f32m2(wptr, vl), v, vl);
            vs1 = __riscv_vfmacc_vv_f32m2(vs1, __riscv_vle32_v_f32m2(wptr + wstep, vl), v, vl);
            vs2 = __riscv_vfmacc_vv_f32m2(vs2, __riscv_vle32_v_f32m2(wptr + wstep*2, vl), v, vl);
            vs3 = __riscv_vfmacc_vv_f32m2(vs3, __riscv_vle32_v_f32m2(wptr + wstep*3, vl), v, vl);
            vs4 = __riscv_vfmacc_vv_f32m2(vs4, __riscv_vle32_v_f32m2(wptr + wstep*4, vl), v, vl);
            vs5 = __riscv_vfmacc_vv_f32m2(vs5, __riscv_vle32_v_f32m2(wptr + wstep*5, vl), v, vl);
            vs6 = __riscv_vfmacc_vv_f32m2(vs6, __riscv_vle32_v_f32m2(wptr + wstep*6, vl), v, vl);
            vs7 = __riscv_vfmacc_vv_f32m2(vs7, __riscv_vle32_v_f32m2(wptr + wstep*7, vl), v, vl);
            vs8 = __riscv_vfmacc_vv_f32m2(vs8, __riscv_vle32_v_f32m2(wptr + wstep*8, vl), v, vl);
            vs9 = __riscv_vfmacc_vv_f32m2(vs9, __riscv_vle32_v_f32m2(wptr + wstep*9, vl), v, vl);
            vs10 = __riscv_vfmacc_vv_f32m2(vs10, __riscv_vle32_v_f32m2(wptr + wstep*10, vl), v, vl);
            vs11 = __riscv_vfmacc_vv_f32m2(vs11, __riscv_vle32_v_f32m2(wptr + wstep*11, vl), v, vl);
            vs12 = __riscv_vfmacc_vv_f32m2(vs12, __riscv_vle32_v_f32m2(wptr + wstep*12, vl), v, vl);
            vs13 = __riscv_vfmacc_vv_f32m2(vs13, __riscv_vle32_v_f32m2(wptr + wstep*13, vl), v, vl);
            vs14 = __riscv_vfmacc_vv_f32m2(vs14, __riscv_vle32_v_f32m2(wptr + wstep*14, vl), v, vl);
        }

        // Calculate the sum of each vector
        float sum[15];
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0, vlm2);
        sum[0] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs0, zero, vlm2));
        sum[1] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs1, zero, vlm2));
        sum[2] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs2, zero, vlm2));
        sum[3] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs3, zero, vlm2));
        sum[4] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs4, zero, vlm2));
        sum[5] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs5, zero, vlm2));
        sum[6] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs6, zero, vlm2));
        sum[7] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs7, zero, vlm2));
        sum[8] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs8, zero, vlm2));
        sum[9] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs9, zero, vlm2));
        sum[10] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs10, zero, vlm2));
        sum[11] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs11, zero, vlm2));
        sum[12] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs12, zero, vlm2));
        sum[13] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs13, zero, vlm2));
        sum[14] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs14, zero, vlm2));

        vfloat32m4_t s0 = __riscv_vfadd_vv_f32m4(__riscv_vle32_v_f32m4(sum, 15), __riscv_vle32_v_f32m4(bias + i, 15), 15);
        __riscv_vse32_v_f32m4(dst + i, s0, 15);
    }
    int unroll_tail = nvecs - i;
    if (unroll_tail > 0)
    {
        const float* wptr = weights + i*wstep;
        vfloat32m2_t
               vs0 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs1 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs2 = __riscv_vfmv_v_f_f32m2(0, vlm2),
               vs3 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs4 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs5 = __riscv_vfmv_v_f_f32m2(0, vlm2),
               vs6 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs7 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs8 = __riscv_vfmv_v_f_f32m2(0, vlm2),
               vs9 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs10 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs11 = __riscv_vfmv_v_f_f32m2(0, vlm2),
               vs12 = __riscv_vfmv_v_f_f32m2(0, vlm2), vs13 = __riscv_vfmv_v_f_f32m2(0, vlm2);
        int avl = vecsize, vl;
        for(int k = 0; k < vecsize; k += vl, wptr += vl, avl -= vl)
        {
            vl = __riscv_vsetvl_e32m2(avl);
            vfloat32m2_t v = __riscv_vle32_v_f32m2(vec + k, vl);
            vs0 = __riscv_vfmacc_vv_f32m2(vs0, __riscv_vle32_v_f32m2(wptr, vl), v, vl);
            vs1 = __riscv_vfmacc_vv_f32m2(vs1, __riscv_vle32_v_f32m2(wptr + wstep*std::min(1, unroll_tail-1), vl), v, vl);
            vs2 = __riscv_vfmacc_vv_f32m2(vs2, __riscv_vle32_v_f32m2(wptr + wstep*std::min(2, unroll_tail-1), vl), v, vl);
            vs3 = __riscv_vfmacc_vv_f32m2(vs3, __riscv_vle32_v_f32m2(wptr + wstep*std::min(3, unroll_tail-1), vl), v, vl);
            vs4 = __riscv_vfmacc_vv_f32m2(vs4, __riscv_vle32_v_f32m2(wptr + wstep*std::min(4, unroll_tail-1), vl), v, vl);
            vs5 = __riscv_vfmacc_vv_f32m2(vs5, __riscv_vle32_v_f32m2(wptr + wstep*std::min(5, unroll_tail-1), vl), v, vl);
            vs6 = __riscv_vfmacc_vv_f32m2(vs6, __riscv_vle32_v_f32m2(wptr + wstep*std::min(6, unroll_tail-1), vl), v, vl);
            vs7 = __riscv_vfmacc_vv_f32m2(vs7, __riscv_vle32_v_f32m2(wptr + wstep*std::min(7, unroll_tail-1), vl), v, vl);
            vs8 = __riscv_vfmacc_vv_f32m2(vs8, __riscv_vle32_v_f32m2(wptr + wstep*std::min(8, unroll_tail-1), vl), v, vl);
            vs9 = __riscv_vfmacc_vv_f32m2(vs9, __riscv_vle32_v_f32m2(wptr + wstep*std::min(9, unroll_tail-1), vl), v, vl);
            vs10 = __riscv_vfmacc_vv_f32m2(vs10, __riscv_vle32_v_f32m2(wptr + wstep*std::min(10, unroll_tail-1), vl), v, vl);
            vs11 = __riscv_vfmacc_vv_f32m2(vs11, __riscv_vle32_v_f32m2(wptr + wstep*std::min(11, unroll_tail-1), vl), v, vl);
            vs12 = __riscv_vfmacc_vv_f32m2(vs12, __riscv_vle32_v_f32m2(wptr + wstep*std::min(12, unroll_tail-1), vl), v, vl);
            vs13 = __riscv_vfmacc_vv_f32m2(vs13, __riscv_vle32_v_f32m2(wptr + wstep*std::min(13, unroll_tail-1), vl), v, vl);
        }

        // Calculate the sum of each vector
        float sum[14];
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0, vlm2);
        sum[0] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs0, zero, vlm2));
        sum[1] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs1, zero, vlm2));
        sum[2] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs2, zero, vlm2));
        sum[3] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs3, zero, vlm2));
        sum[4] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs4, zero, vlm2));
        sum[5] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs5, zero, vlm2));
        sum[6] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs6, zero, vlm2));
        sum[7] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs7, zero, vlm2));
        sum[8] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs8, zero, vlm2));
        sum[9] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs9, zero, vlm2));
        sum[10] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs10, zero, vlm2));
        sum[11] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs11, zero, vlm2));
        sum[12] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs12, zero, vlm2));
        sum[13] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vs13, zero, vlm2));

        vfloat32m4_t s0 = __riscv_vfadd_vv_f32m4(__riscv_vle32_v_f32m4(sum, unroll_tail), __riscv_vle32_v_f32m4(bias + i, unroll_tail), unroll_tail);
        __riscv_vse32_v_f32m4(dst + i, s0, unroll_tail);
    }
}

#endif // CV_RVV

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_LASX

// dst = vec * weights^t + bias
void fastGEMM1T( const float* vec, const float* weights,
                 size_t wstep, const float* bias,
                 float* dst, int nvecs, int vecsize )
{
    int i = 0;
    __m256i v256_tmp;

    for( ; i <= nvecs - 8; i += 8 )
    {
        const float* wptr = weights + i*wstep;
        __m256 vs0 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp), vs1 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp),
               vs2 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp), vs3 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp),
               vs4 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp), vs5 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp),
               vs6 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp), vs7 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp);

        for( int k = 0; k < vecsize; k += 8, wptr += 8 )
        {
            __m256 v = (__m256)__lasx_xvld(vec + k, 0);

            vs0 = __lasx_xvfmadd_s((__m256)__lasx_xvld(wptr, 0), v, vs0);
            vs1 = __lasx_xvfmadd_s((__m256)__lasx_xvld(wptr + wstep, 0), v, vs1);
            vs2 = __lasx_xvfmadd_s((__m256)__lasx_xvld(wptr + wstep*2, 0), v, vs2);
            vs3 = __lasx_xvfmadd_s((__m256)__lasx_xvld(wptr + wstep*3, 0), v, vs3);
            vs4 = __lasx_xvfmadd_s((__m256)__lasx_xvld(wptr + wstep*4, 0), v, vs4);
            vs5 = __lasx_xvfmadd_s((__m256)__lasx_xvld(wptr + wstep*5, 0), v, vs5);
            vs6 = __lasx_xvfmadd_s((__m256)__lasx_xvld(wptr + wstep*6, 0), v, vs6);
            vs7 = __lasx_xvfmadd_s((__m256)__lasx_xvld(wptr + wstep*7, 0), v, vs7);
        }

        /*s0*/
        __m256  vs00_perm   = (__m256)__lasx_xvpermi_d(vs0, (2<<6) + (3<<4) + (0<<2) + 1);
        __m256  vs00_add_2w = __lasx_xvfadd_s(vs0, vs00_perm);
        __m256  tmp00_srl   = (__m256)__lasx_xvsrli_d(vs00_add_2w, 32);
        __m256  vs00_add_4w = __lasx_xvfadd_s(vs00_add_2w, tmp00_srl);

        __m256  vs01_perm   = (__m256)__lasx_xvpermi_d(vs1, (2<<6) + (3<<4) + (0<<2) + 1);
        __m256  vs01_add_2w = __lasx_xvfadd_s(vs1, vs01_perm);
        __m256  tmp01_srl   = (__m256)__lasx_xvsrli_d(vs01_add_2w, 32);
        __m256  vs01_add_4w = __lasx_xvfadd_s(vs01_add_2w, tmp01_srl);

        __m256  vs02_perm   = (__m256)__lasx_xvpermi_d(vs2, (2<<6) + (3<<4) + (0<<2) + 1);
        __m256  vs02_add_2w = __lasx_xvfadd_s(vs2, vs02_perm);
        __m256  tmp02_srl   = (__m256)__lasx_xvsrli_d(vs02_add_2w, 32);
        __m256  vs02_add_4w = __lasx_xvfadd_s(vs02_add_2w, tmp02_srl);

        __m256  vs03_perm   = (__m256)__lasx_xvpermi_d(vs3, (2<<6) + (3<<4) + (0<<2) + 1);
        __m256  vs03_add_2w = __lasx_xvfadd_s(vs3, vs03_perm);
        __m256  tmp03_srl   = (__m256)__lasx_xvsrli_d(vs03_add_2w, 32);
        __m256  vs03_add_4w = __lasx_xvfadd_s(vs03_add_2w, tmp03_srl);

        __m256i  vs01_vs00 = __lasx_xvpackev_w((__m256i)vs01_add_4w, (__m256i)vs00_add_4w);
        __m256i  vs03_vs02 = __lasx_xvpackev_w((__m256i)vs03_add_4w, (__m256i)vs02_add_4w);
        __m256          s0 = (__m256)__lasx_xvpackev_d(vs03_vs02, vs01_vs00);

        /*s1*/
        __m256  vs10_perm   = (__m256)__lasx_xvpermi_d(vs4, (2<<6) + (3<<4) + (0<<2) + 1);
        __m256  vs10_add_2w = __lasx_xvfadd_s(vs4, vs10_perm);
        __m256  tmp10_srl   = (__m256)__lasx_xvsrli_d(vs10_add_2w, 32);
        __m256  vs10_add_4w = __lasx_xvfadd_s(vs10_add_2w, tmp10_srl);

        __m256  vs11_perm   = (__m256)__lasx_xvpermi_d(vs5, (2<<6) + (3<<4) + (0<<2) + 1);
        __m256  vs11_add_2w = __lasx_xvfadd_s(vs5, vs11_perm);
        __m256  tmp11_srl   = (__m256)__lasx_xvsrli_d(vs11_add_2w, 32);
        __m256  vs11_add_4w = __lasx_xvfadd_s(vs11_add_2w, tmp11_srl);

        __m256  vs12_perm   = (__m256)__lasx_xvpermi_d(vs6, (2<<6) + (3<<4) + (0<<2) + 1);
        __m256  vs12_add_2w = __lasx_xvfadd_s(vs6, vs12_perm);
        __m256  tmp12_srl   = (__m256)__lasx_xvsrli_d(vs12_add_2w, 32);
        __m256  vs12_add_4w = __lasx_xvfadd_s(vs12_add_2w, tmp12_srl);

        __m256  vs13_perm   = (__m256)__lasx_xvpermi_d(vs7, (2<<6) + (3<<4) + (0<<2) + 1);
        __m256  vs13_add_2w = __lasx_xvfadd_s(vs7, vs13_perm);
        __m256  tmp13_srl   = (__m256)__lasx_xvsrli_d(vs13_add_2w, 32);
        __m256  vs13_add_4w = __lasx_xvfadd_s(vs13_add_2w, tmp13_srl);

        __m256i vs11_vs10 = __lasx_xvpackev_w((__m256i)vs11_add_4w, (__m256i)vs10_add_4w);
        __m256i vs13_vs12 = __lasx_xvpackev_w((__m256i)vs13_add_4w, (__m256i)vs12_add_4w);
        __m256         s1 = (__m256)__lasx_xvpackev_d(vs13_vs12, vs11_vs10);

        s0 = __lasx_xvfadd_s(s0, (__m256)__lasx_xvpermi_q(s0, s0, 1));
        s1 = __lasx_xvfadd_s(s1, (__m256)__lasx_xvpermi_q(s1, s1, 1));

        s0 = __lasx_xvfadd_s(s0, (__m256)__lasx_xvld(bias + i, 0));
        s1 = __lasx_xvfadd_s(s1, (__m256)__lasx_xvld(bias + i, 4*4));

        __lsx_vst(*(__m128*)&s0, dst + i, 0);
        __lsx_vst(*(__m128*)&s1, dst + i, 4*4);
    }

    float temp = 0.f;
    for( ; i < nvecs; i++ )
    {
        const float* wptr = weights + i*wstep;
        __m256 vs0 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp);

        for( int k = 0; k < vecsize; k += 8, wptr += 8 )
        {
            __m256 v = (__m256)__lasx_xvld(vec + k, 0);
            vs0 = __lasx_xvfmadd_s((__m256)__lasx_xvld(wptr, 0), v, vs0);
        }

        __m256i vs0_perm   = __lasx_xvpermi_d(vs0, (2<<6) + (3<<4) + (0<<2) + 1);
        __m256  vs0_add_2w = __lasx_xvfadd_s(vs0, (__m256)vs0_perm);
        __m256i tmp_srl    = __lasx_xvsrli_d(vs0_add_2w, 32);
        __m256  vs0_add_4w = __lasx_xvfadd_s(vs0_add_2w, (__m256)tmp_srl);
        temp = ((v8f32)vs0_add_4w)[0] + ((v8f32)vs0_add_4w)[4];
        dst[i] = temp + bias[i];
    }
}


void fastGEMM( const float* aptr, size_t astep, const float* bptr,
               size_t bstep, float* cptr, size_t cstep,
               int ma, int na, int nb )
{
    int n = 0;

    for( ; n <= nb - 16; n += 16 )
    {
        for( int m = 0; m < ma; m += 4 )
        {
            const float* aptr0 = aptr + astep*m;
            const float* aptr1 = aptr + astep*std::min(m+1, ma-1);
            const float* aptr2 = aptr + astep*std::min(m+2, ma-1);
            const float* aptr3 = aptr + astep*std::min(m+3, ma-1);

            float* cptr0 = cptr + cstep*m;
            float* cptr1 = cptr + cstep*std::min(m+1, ma-1);
            float* cptr2 = cptr + cstep*std::min(m+2, ma-1);
            float* cptr3 = cptr + cstep*std::min(m+3, ma-1);

            __m256i v256_tmp;
            __m256 d00 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp), d01 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp);
            __m256 d10 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp), d11 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp);
            __m256 d20 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp), d21 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp);
            __m256 d30 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp), d31 = (__m256)__lasx_xvxor_v(v256_tmp, v256_tmp);

            for( int k = 0; k < na; k++ )
            {
                __m256 a0 = _v256_setall_ps(aptr0[k]);
                __m256 a1 = _v256_setall_ps(aptr1[k]);
                __m256 a2 = _v256_setall_ps(aptr2[k]);
                __m256 a3 = _v256_setall_ps(aptr3[k]);

                __m256 b0 = (__m256)__lasx_xvld(bptr + k*bstep + n, 0);
                __m256 b1 = (__m256)__lasx_xvld(bptr + k*bstep + n + 8, 0);
                d00 = __lasx_xvfmadd_s(a0, b0, d00);
                d01 = __lasx_xvfmadd_s(a0, b1, d01);
                d10 = __lasx_xvfmadd_s(a1, b0, d10);
                d11 = __lasx_xvfmadd_s(a1, b1, d11);
                d20 = __lasx_xvfmadd_s(a2, b0, d20);
                d21 = __lasx_xvfmadd_s(a2, b1, d21);
                d30 = __lasx_xvfmadd_s(a3, b0, d30);
                d31 = __lasx_xvfmadd_s(a3, b1, d31);
            }

            __lasx_xvst(d00, cptr0 + n, 0);
            __lasx_xvst(d01, cptr0 + n, 8*4);
            __lasx_xvst(d10, cptr1 + n, 0);
            __lasx_xvst(d11, cptr1 + n, 8*4);
            __lasx_xvst(d20, cptr2 + n, 0);
            __lasx_xvst(d21, cptr2 + n, 8*4);
            __lasx_xvst(d30, cptr3 + n, 0);
            __lasx_xvst(d31, cptr3 + n, 8*4);
        }
    }

    for( ; n < nb; n++ )
    {
        for( int m = 0; m < ma; m++ )
        {
            const float* aptr0 = aptr + astep*m;
            float* cptr0 = cptr + cstep*m;
            float d0 = 0.f;

            for( int k = 0; k < na; k++ )
                d0 += aptr0[k]*bptr[k*bstep + n];

            cptr0[n] = d0;
        }
    }
}

#endif // CV_LASX

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace
