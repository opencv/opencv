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

void fastConv( const float* weights, size_t wstep, const float* bias,
               const float* rowbuf, float* output, const int* outShape,
               int blockSize, int vecsize, int vecsize_aligned,
               const float* relu, bool initOutput );
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

void fastConv( const float* weights, size_t wstep, const float* bias,
               const float* rowbuf, float* output, const int* outShape,
               int blockSize, int vecsize, int vecsize_aligned,
               const float* relu, bool initOutput )
{
    int outCn = outShape[1];
    size_t outPlaneSize = outShape[2]*outShape[3];
    float r0 = 1.f, r1 = 1.f, r2 = 1.f;
    __m128 vr0 = _mm_set1_ps(1.f), vr1 = vr0, vr2 = vr0, z = _mm_setzero_ps();

    // now compute dot product of the weights
    // and im2row-transformed part of the tensor
    for( int i = 0; i < outCn; i += 3 )
    {
        const float* wptr0 = weights + i*wstep;
        const float* wptr1 = wptr0 + wstep;
        const float* wptr2 = wptr1 + wstep;
        float* outptr0 = output + i*outPlaneSize;
        float* outptr1 = outptr0 + outPlaneSize;
        float* outptr2 = outptr1 + outPlaneSize;
        float bias0 = bias[i], bias1 = bias[i+1], bias2 = bias[i+2];

        if( i+2 >= outCn )
        {
            wptr2 = wptr1;
            outptr2 = outptr1;
            bias2 = bias1;
            if( i+1 >= outCn )
            {
                wptr2 = wptr1 = wptr0;
                outptr2 = outptr1 = outptr0;
                bias2 = bias1 = bias0;
            }
        }

        if( relu )
        {
            r0 = relu[i]; r1 = relu[i+1]; r2 = relu[i+2];
            if( i+2 >= outCn )
            {
                r2 = r1;
                if( i+1 >= outCn )
                    r2 = r1 = r0;
            }
            vr0 = _mm_set1_ps(r0);
            vr1 = _mm_set1_ps(r1);
            vr2 = _mm_set1_ps(r2);
        }

        int j = 0;
        for( ; j <= blockSize - 4; j += 4 )
        {
            int k = 0;
            const float* rptr = rowbuf + j*vecsize_aligned;

            __m256 vs00 = _mm256_setzero_ps(), vs01 = _mm256_setzero_ps(),
                   vs02 = _mm256_setzero_ps(), vs03 = _mm256_setzero_ps(),
                   vs10 = _mm256_setzero_ps(), vs11 = _mm256_setzero_ps(),
                   vs12 = _mm256_setzero_ps(), vs13 = _mm256_setzero_ps(),
                   vs20 = _mm256_setzero_ps(), vs21 = _mm256_setzero_ps(),
                   vs22 = _mm256_setzero_ps(), vs23 = _mm256_setzero_ps();

#if CV_AVX512_SKX // AVX512VL is necessary to avoid register spilling
            if (vecsize >= 32)
            {
                __m512 vs00_5 = _mm512_setzero_ps(), vs01_5 = _mm512_setzero_ps(),
                       vs02_5 = _mm512_setzero_ps(), vs03_5 = _mm512_setzero_ps(),
                       vs10_5 = _mm512_setzero_ps(), vs11_5 = _mm512_setzero_ps(),
                       vs12_5 = _mm512_setzero_ps(), vs13_5 = _mm512_setzero_ps(),
                       vs20_5 = _mm512_setzero_ps(), vs21_5 = _mm512_setzero_ps(),
                       vs22_5 = _mm512_setzero_ps(), vs23_5 = _mm512_setzero_ps();

                for (; k <= vecsize - 16; k += 16, rptr += 16)
                {
                    __m512 w0 = _mm512_loadu_ps(wptr0 + k);
                    __m512 w1 = _mm512_loadu_ps(wptr1 + k);
                    __m512 w2 = _mm512_loadu_ps(wptr2 + k);
                    __m512 r0 = _mm512_loadu_ps(rptr);

                    vs00_5 = _mm512_fmadd_ps(w0, r0, vs00_5);
                    vs10_5 = _mm512_fmadd_ps(w1, r0, vs10_5);
                    vs20_5 = _mm512_fmadd_ps(w2, r0, vs20_5);

                    r0 = _mm512_loadu_ps(rptr + vecsize_aligned);
                    vs01_5 = _mm512_fmadd_ps(w0, r0, vs01_5);
                    vs11_5 = _mm512_fmadd_ps(w1, r0, vs11_5);
                    vs21_5 = _mm512_fmadd_ps(w2, r0, vs21_5);

                    r0 = _mm512_loadu_ps(rptr + vecsize_aligned*2);
                    vs02_5 = _mm512_fmadd_ps(w0, r0, vs02_5);
                    vs12_5 = _mm512_fmadd_ps(w1, r0, vs12_5);
                    vs22_5 = _mm512_fmadd_ps(w2, r0, vs22_5);

                    r0 = _mm512_loadu_ps(rptr + vecsize_aligned*3);
                    vs03_5 = _mm512_fmadd_ps(w0, r0, vs03_5);
                    vs13_5 = _mm512_fmadd_ps(w1, r0, vs13_5);
                    vs23_5 = _mm512_fmadd_ps(w2, r0, vs23_5);
                }
                /*
                 * now fold the 512 bit accumulator vectors into 256 bit vectors so that the AVX2 code can finish
                 * the tail of the vector
                 */
                vs00 = _mm256_add_ps( _mm512_extractf32x8_ps(vs00_5, 0), _mm512_extractf32x8_ps(vs00_5, 1));
                vs10 = _mm256_add_ps( _mm512_extractf32x8_ps(vs10_5, 0), _mm512_extractf32x8_ps(vs10_5, 1));
                vs20 = _mm256_add_ps( _mm512_extractf32x8_ps(vs20_5, 0), _mm512_extractf32x8_ps(vs20_5, 1));

                vs01 = _mm256_add_ps( _mm512_extractf32x8_ps(vs01_5, 0), _mm512_extractf32x8_ps(vs01_5, 1));
                vs11 = _mm256_add_ps( _mm512_extractf32x8_ps(vs11_5, 0), _mm512_extractf32x8_ps(vs11_5, 1));
                vs21 = _mm256_add_ps( _mm512_extractf32x8_ps(vs21_5, 0), _mm512_extractf32x8_ps(vs21_5, 1));

                vs02 = _mm256_add_ps( _mm512_extractf32x8_ps(vs02_5, 0), _mm512_extractf32x8_ps(vs02_5, 1));
                vs12 = _mm256_add_ps( _mm512_extractf32x8_ps(vs12_5, 0), _mm512_extractf32x8_ps(vs12_5, 1));
                vs22 = _mm256_add_ps( _mm512_extractf32x8_ps(vs22_5, 0), _mm512_extractf32x8_ps(vs22_5, 1));

                vs03 = _mm256_add_ps( _mm512_extractf32x8_ps(vs03_5, 0), _mm512_extractf32x8_ps(vs03_5, 1));
                vs13 = _mm256_add_ps( _mm512_extractf32x8_ps(vs13_5, 0), _mm512_extractf32x8_ps(vs13_5, 1));
                vs23 = _mm256_add_ps( _mm512_extractf32x8_ps(vs23_5, 0), _mm512_extractf32x8_ps(vs23_5, 1));
            }
#endif

            for (; k < vecsize; k += 8, rptr += 8 )
            {
                __m256 w0 = _mm256_load_ps(wptr0 + k);
                __m256 w1 = _mm256_load_ps(wptr1 + k);
                __m256 w2 = _mm256_load_ps(wptr2 + k);
                __m256 r0 = _mm256_load_ps(rptr);

                vs00 = _mm256_fmadd_ps(w0, r0, vs00);
                vs10 = _mm256_fmadd_ps(w1, r0, vs10);
                vs20 = _mm256_fmadd_ps(w2, r0, vs20);

                r0 = _mm256_load_ps(rptr + vecsize_aligned);
                vs01 = _mm256_fmadd_ps(w0, r0, vs01);
                vs11 = _mm256_fmadd_ps(w1, r0, vs11);
                vs21 = _mm256_fmadd_ps(w2, r0, vs21);

                r0 = _mm256_load_ps(rptr + vecsize_aligned*2);
                vs02 = _mm256_fmadd_ps(w0, r0, vs02);
                vs12 = _mm256_fmadd_ps(w1, r0, vs12);
                vs22 = _mm256_fmadd_ps(w2, r0, vs22);

                r0 = _mm256_load_ps(rptr + vecsize_aligned*3);
                vs03 = _mm256_fmadd_ps(w0, r0, vs03);
                vs13 = _mm256_fmadd_ps(w1, r0, vs13);
                vs23 = _mm256_fmadd_ps(w2, r0, vs23);
            }

            __m256 t0 = _mm256_hadd_ps(_mm256_hadd_ps(vs00, vs01), _mm256_hadd_ps(vs02, vs03));
            __m256 t1 = _mm256_hadd_ps(_mm256_hadd_ps(vs10, vs11), _mm256_hadd_ps(vs12, vs13));
            __m256 t2 = _mm256_hadd_ps(_mm256_hadd_ps(vs20, vs21), _mm256_hadd_ps(vs22, vs23));

            t0 = _mm256_add_ps(t0, _mm256_permute2f128_ps(t0, t0, 1));
            t1 = _mm256_add_ps(t1, _mm256_permute2f128_ps(t1, t1, 1));
            t2 = _mm256_add_ps(t2, _mm256_permute2f128_ps(t2, t2, 1));

            __m128 s0, s1, s2;

            if( initOutput )
            {
                s0 = _mm_set1_ps(bias0);
                s1 = _mm_set1_ps(bias1);
                s2 = _mm_set1_ps(bias2);
            }
            else
            {
                s0 = _mm_loadu_ps(outptr0 + j);
                s1 = _mm_loadu_ps(outptr1 + j);
                s2 = _mm_loadu_ps(outptr2 + j);
            }

            s0 = _mm_add_ps(s0, _mm256_castps256_ps128(t0));
            s1 = _mm_add_ps(s1, _mm256_castps256_ps128(t1));
            s2 = _mm_add_ps(s2, _mm256_castps256_ps128(t2));

            if( relu )
            {
                __m128 m0 = _mm_cmp_ps(s0, z, _CMP_GT_OS);
                __m128 m1 = _mm_cmp_ps(s1, z, _CMP_GT_OS);
                __m128 m2 = _mm_cmp_ps(s2, z, _CMP_GT_OS);
                s0 = _mm_xor_ps(s0, _mm_andnot_ps(m0, _mm_xor_ps(_mm_mul_ps(s0, vr0), s0)));
                s1 = _mm_xor_ps(s1, _mm_andnot_ps(m1, _mm_xor_ps(_mm_mul_ps(s1, vr1), s1)));
                s2 = _mm_xor_ps(s2, _mm_andnot_ps(m2, _mm_xor_ps(_mm_mul_ps(s2, vr2), s2)));
            }

            _mm_storeu_ps(outptr0 + j, s0);
            _mm_storeu_ps(outptr1 + j, s1);
            _mm_storeu_ps(outptr2 + j, s2);
        }

        for( ; j < blockSize; j++ )
        {
            const float* rptr = rowbuf + j*vecsize_aligned;
            float s00, s10, s20;

            if( initOutput )
            {
                s00 = bias0;
                s10 = bias1;
                s20 = bias2;
            }
            else
            {
                s00 = outptr0[j];
                s10 = outptr1[j];
                s20 = outptr2[j];
            }

            for( int k = 0; k < vecsize; k++ )
            {
                float r0 = rptr[k];
                s00 += wptr0[k]*r0;
                s10 += wptr1[k]*r0;
                s20 += wptr2[k]*r0;
            }

            if( relu )
            {
                s00 = s00 > 0.f ? s00 : s00*r0;
                s10 = s10 > 0.f ? s10 : s10*r1;
                s20 = s20 > 0.f ? s20 : s20*r2;
            }

            outptr0[j] = s00;
            outptr1[j] = s10;
            outptr2[j] = s20;
        }
    }
    _mm256_zeroupper();
}

// dst = vec * weights^t + bias
void fastGEMM1T( const float* vec, const float* weights,
                 size_t wstep, const float* bias,
                 float* dst, int nvecs, int vecsize )
{
    int i = 0;

    for( ; i <= nvecs - 8; i += 8 )
    {
        const float* wptr = weights + i*wstep;
        __m256 vs0 = _mm256_setzero_ps(), vs1 = _mm256_setzero_ps(),
               vs2 = _mm256_setzero_ps(), vs3 = _mm256_setzero_ps(),
               vs4 = _mm256_setzero_ps(), vs5 = _mm256_setzero_ps(),
               vs6 = _mm256_setzero_ps(), vs7 = _mm256_setzero_ps();

        for( int k = 0; k < vecsize; k += 8, wptr += 8 )
        {
            __m256 v = _mm256_load_ps(vec + k);

            vs0 = _mm256_fmadd_ps(_mm256_load_ps(wptr), v, vs0);
            vs1 = _mm256_fmadd_ps(_mm256_load_ps(wptr + wstep), v, vs1);
            vs2 = _mm256_fmadd_ps(_mm256_load_ps(wptr + wstep*2), v, vs2);
            vs3 = _mm256_fmadd_ps(_mm256_load_ps(wptr + wstep*3), v, vs3);
            vs4 = _mm256_fmadd_ps(_mm256_load_ps(wptr + wstep*4), v, vs4);
            vs5 = _mm256_fmadd_ps(_mm256_load_ps(wptr + wstep*5), v, vs5);
            vs6 = _mm256_fmadd_ps(_mm256_load_ps(wptr + wstep*6), v, vs6);
            vs7 = _mm256_fmadd_ps(_mm256_load_ps(wptr + wstep*7), v, vs7);
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

        for( int k = 0; k < vecsize; k += 8, wptr += 8 )
        {
            __m256 v = _mm256_load_ps(vec + k);
            vs0 = _mm256_fmadd_ps(_mm256_load_ps(wptr), v, vs0);
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

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace
