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
void fastDepthwiseConv( const float* weights,
                      int kernel_h, int kernel_w,
                      int stride_h, int stride_w,
                      int dilation_h, int dilation_w,
                      int pad_t, int pad_l,
                      const float* bias, const float* relu,
                      const float* inptr,
                      int height, int width,
                      float* outptr,
                      int out_d, int outH, int outW );
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

enum { FASCONV_BASE_VECSZ = 4 };

void fastConv( const float* weights, size_t wstep, const float* bias,
               const float* rowbuf, float* output, const int* outShape,
               int blockSize, int vecsize, int vecsize_aligned,
               const float* relu, bool initOutput )
{
    int outCn = outShape[1];
    size_t outPlaneSize = outShape[2]*outShape[3];
    float r0 = 1.f, r1 = 1.f, r2 = 1.f;
    __m128 vr0 = _mm_set1_ps(1.f), vr1 = vr0, vr2 = vr0, z = _mm_setzero_ps();
    int CV_DECL_ALIGNED(16) maskbuf[FASCONV_BASE_VECSZ] = {0};
    int rsz = blockSize % FASCONV_BASE_VECSZ;
    for( int i = 0; i < rsz; i++ )
        maskbuf[FASCONV_BASE_VECSZ - i - 1] = -1;
    __m128 mask = _mm_loadu_ps((const float*)maskbuf);

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
        for( ; j < blockSize; j += FASCONV_BASE_VECSZ )
        {
            bool tail = false;
            if (j + FASCONV_BASE_VECSZ > blockSize)
            {
                if (j == 0)
                    break;
                j = blockSize - FASCONV_BASE_VECSZ;
                tail = true;
            }
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
                s0 = _mm_blendv_ps(_mm_mul_ps(s0, vr0), s0, m0);
                s1 = _mm_blendv_ps(_mm_mul_ps(s1, vr1), s1, m1);
                s2 = _mm_blendv_ps(_mm_mul_ps(s2, vr2), s2, m2);
            }

            if( tail )
            {
                s0 = _mm_blendv_ps(_mm_loadu_ps(outptr0 + j), s0, mask);
                s1 = _mm_blendv_ps(_mm_loadu_ps(outptr1 + j), s1, mask);
                s2 = _mm_blendv_ps(_mm_loadu_ps(outptr2 + j), s2, mask);
            }

            _mm_storeu_ps(outptr0 + j, s0);
            _mm_storeu_ps(outptr1 + j, s1);
            _mm_storeu_ps(outptr2 + j, s2);
        }

        for( ; j <= blockSize - 2; j += 2 )
        {
            const float* rptr0 = rowbuf + j*vecsize_aligned;
            const float* rptr1 = rowbuf + (j+1)*vecsize_aligned;
            float s00, s01, s10, s11, s20, s21;

            if( initOutput )
            {
                s00 = s01 = bias0;
                s10 = s11 = bias1;
                s20 = s21 = bias2;
            }
            else
            {
                s00 = outptr0[j]; s01 = outptr0[j+1];
                s10 = outptr1[j]; s11 = outptr1[j+1];
                s20 = outptr2[j]; s21 = outptr2[j+1];
            }

            for( int k = 0; k < vecsize; k++ )
            {
                float w0 = wptr0[k], w1 = wptr1[k], w2 = wptr2[k];
                float r = rptr0[k];
                s00 += w0*r; s10 += w1*r; s20 += w2*r;
                r = rptr1[k];
                s01 += w0*r; s11 += w1*r; s21 += w2*r;
            }

            if( relu )
            {
                s00 = s00 > 0.f ? s00 : s00*r0;
                s01 = s01 > 0.f ? s01 : s01*r0;
                s10 = s10 > 0.f ? s10 : s10*r1;
                s11 = s11 > 0.f ? s11 : s11*r1;
                s20 = s20 > 0.f ? s20 : s20*r2;
                s21 = s21 > 0.f ? s21 : s21*r2;
            }

            outptr0[j] = s00;
            outptr0[j+1] = s01;
            outptr1[j] = s10;
            outptr1[j+1] = s11;
            outptr2[j] = s20;
            outptr2[j+1] = s21;
        }

        for( ; j < blockSize; j++ )
        {
            const float* rptr0 = rowbuf + j*vecsize_aligned;
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
                float w0 = wptr0[k], w1 = wptr1[k], w2 = wptr2[k];
                float r = rptr0[k];
                s00 += w0*r; s10 += w1*r; s20 += w2*r;
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

static inline void _mm256_load_deinterleave(const float* ptr, __m256& a, __m256& b)
{
    __m256 t0 = _mm256_loadu_ps(ptr);
    __m256 t1 = _mm256_loadu_ps(ptr + 8);

    __m256 lo = _mm256_permute2f128_ps(t0, t1, 0+2*16);
    __m256 hi = _mm256_permute2f128_ps(t0, t1, 1+3*16);
    a = _mm256_shuffle_ps(lo, hi, 0x88);
    b = _mm256_shuffle_ps(lo, hi, 0xdd);
}

void fastDepthwiseConv( const float* wptr,
                     int kernel_h, int kernel_w,
                     int stride_h, int stride_w,
                     int dilation_h, int dilation_w,
                     int pad_t, int pad_l,
                     const float* biasptr, const float* relu,
                     const float* inptr_,
                     int height, int width,
                     float* outptr_,
                     int out_d, int outH, int outW )
{
    const float w00_ = wptr[0], w01_ = wptr[1], w02_ = wptr[2],
                w10 = wptr[3], w11 = wptr[4], w12 = wptr[5],
                w20_ = wptr[6], w21_ = wptr[7], w22_ = wptr[8];
    int outW1 = min(outW, (width - dilation_w*(kernel_w - 1) + pad_l)/stride_w);
    float relu_coeff = relu ? relu[out_d] : 1.f, bias = biasptr[out_d];

    for (int out_i = 0; out_i < outH; out_i++)
    {
        int in_i = out_i * stride_h - pad_t, out_j = 0;
        const float* imgptr0 = inptr_ + in_i*width;
        const float* imgptr1 = imgptr0 + dilation_h*width;
        const float* imgptr2 = imgptr0 + (dilation_h*2)*width;
        float out, w00 = w00_, w01 = w01_, w02 = w02_;
        float w20 = w20_, w21 = w21_, w22 = w22_;
        if (in_i < 0)
        {
            w00 = w01 = w02 = 0.f;
            imgptr0 = imgptr1;
        }
        else if (in_i + dilation_h*(kernel_h-1) >= height)
        {
            w20 = w21 = w22 = 0.f;
            imgptr2 = imgptr1;
        }
        float* outptr = outptr_ + out_i*outW;
        if (pad_l > 0)
        {
            out = imgptr0[0]*w01 + imgptr0[dilation_w]*w02 +
                  imgptr1[0]*w11 + imgptr1[dilation_w]*w12 +
                  imgptr2[0]*w21 + imgptr2[dilation_w]*w22 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[0] = out;
            out_j = 1;
        }

        if (stride_w == 1 || (stride_w == 2 && dilation_w == 1))
        {
            const int VECSZ = 8;
            __m256 vw00 = _mm256_set1_ps(w00), vw01 = _mm256_set1_ps(w01), vw02 = _mm256_set1_ps(w02),
                      vw10 = _mm256_set1_ps(w10), vw11 = _mm256_set1_ps(w11), vw12 = _mm256_set1_ps(w12),
                      vw20 = _mm256_set1_ps(w20), vw21 = _mm256_set1_ps(w21), vw22 = _mm256_set1_ps(w22);
            __m256 z = _mm256_setzero_ps(), vbias = _mm256_set1_ps(bias), vrc = _mm256_set1_ps(relu_coeff);

            if( stride_w == 1 )
                for( ; out_j < outW1; out_j += VECSZ )
                {
                    if (out_j + VECSZ > outW1 && out_j > pad_l)
                        out_j = outW1 - VECSZ;
                    int in_j = out_j * stride_w - pad_l;
                    __m256 v00 = _mm256_loadu_ps(imgptr0 + in_j),
                           v01 = _mm256_loadu_ps(imgptr0 + in_j + dilation_w),
                           v02 = _mm256_loadu_ps(imgptr0 + in_j + dilation_w*2),
                           v10 = _mm256_loadu_ps(imgptr1 + in_j),
                           v11 = _mm256_loadu_ps(imgptr1 + in_j + dilation_w),
                           v12 = _mm256_loadu_ps(imgptr1 + in_j + dilation_w*2),
                           v20 = _mm256_loadu_ps(imgptr2 + in_j),
                           v21 = _mm256_loadu_ps(imgptr2 + in_j + dilation_w),
                           v22 = _mm256_loadu_ps(imgptr2 + in_j + dilation_w*2);

                    __m256 vout0 = _mm256_fmadd_ps(v00, vw00, vbias);
                    __m256 vout1 = _mm256_mul_ps(v01, vw01);
                    __m256 vout2 = _mm256_mul_ps(v02, vw02);

                    vout0 = _mm256_fmadd_ps(v10, vw10, vout0);
                    vout1 = _mm256_fmadd_ps(v11, vw11, vout1);
                    vout2 = _mm256_fmadd_ps(v12, vw12, vout2);

                    vout0 = _mm256_fmadd_ps(v20, vw20, vout0);
                    vout1 = _mm256_fmadd_ps(v21, vw21, vout1);
                    vout2 = _mm256_fmadd_ps(v22, vw22, vout2);

                    vout0 = _mm256_add_ps(_mm256_add_ps(vout0, vout1), vout2);
                    if (relu)
                    {
                        __m256 m = _mm256_cmp_ps(vout0, z, _CMP_GT_OQ);
                        vout0 = _mm256_blendv_ps(_mm256_mul_ps(vout0, vrc), vout0, m);
                    }
                    _mm256_storeu_ps(outptr + out_j, vout0);
                }
            else
                for( ; out_j < outW1; out_j += VECSZ )
                {
                    if (out_j + VECSZ > outW1 && out_j > pad_l)
                        out_j = outW1 - VECSZ;
                    int in_j = out_j * stride_w - pad_l;
                    __m256 v00, v01, v02, v10, v11, v12, v20, v21, v22, unused;
                    _mm256_load_deinterleave(imgptr0 + in_j, v00, v01);
                    _mm256_load_deinterleave(imgptr0 + in_j + 2, v02, unused);
                    _mm256_load_deinterleave(imgptr1 + in_j, v10, v11);
                    _mm256_load_deinterleave(imgptr1 + in_j + 2, v12, unused);
                    _mm256_load_deinterleave(imgptr2 + in_j, v20, v21);
                    _mm256_load_deinterleave(imgptr2 + in_j + 2, v22, unused);

                    __m256 vout0 = _mm256_fmadd_ps(v00, vw00, vbias);
                    __m256 vout1 = _mm256_mul_ps(v01, vw01);
                    __m256 vout2 = _mm256_mul_ps(v02, vw02);

                    vout0 = _mm256_fmadd_ps(v10, vw10, vout0);
                    vout1 = _mm256_fmadd_ps(v11, vw11, vout1);
                    vout2 = _mm256_fmadd_ps(v12, vw12, vout2);

                    vout0 = _mm256_fmadd_ps(v20, vw20, vout0);
                    vout1 = _mm256_fmadd_ps(v21, vw21, vout1);
                    vout2 = _mm256_fmadd_ps(v22, vw22, vout2);

                    vout0 = _mm256_add_ps(_mm256_add_ps(vout0, vout1), vout2);
                    if (relu)
                    {
                        __m256 m = _mm256_cmp_ps(vout0, z, _CMP_GT_OQ);
                        vout0 = _mm256_blendv_ps(_mm256_mul_ps(vout0, vrc), vout0, m);
                    }
                    _mm256_storeu_ps(outptr + out_j, vout0);
                }
        }

        for (; out_j < outW1; out_j++)
        {
            int in_j = out_j * stride_w - pad_l;
            out = imgptr0[in_j]*w00 + imgptr0[in_j + dilation_w]*w01 + imgptr0[in_j + dilation_w*2]*w02 +
                  imgptr1[in_j]*w10 + imgptr1[in_j + dilation_w]*w11 + imgptr1[in_j + dilation_w*2]*w12 +
                  imgptr2[in_j]*w20 + imgptr2[in_j + dilation_w]*w21 + imgptr2[in_j + dilation_w*2]*w22 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[out_j] = out;
        }

        for (; out_j < outW; out_j++ )
        {
            int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w*2;
            float s0 = 1.f, s1 = 1.f, s2 = 1.f;
            if (in_j0 >= width)
            {
                in_j0 = 0;
                s0 = 0.f;
            }
            if (in_j1 >= width)
            {
                in_j1 = 0;
                s1 = 0.f;
            }
            if (in_j2 >= width)
            {
                in_j2 = 0;
                s2 = 0.f;
            }
            out = imgptr0[in_j0]*w00*s0 + imgptr0[in_j1]*w01*s1 + imgptr0[in_j2]*w02*s2 +
                  imgptr1[in_j0]*w10*s0 + imgptr1[in_j1]*w11*s1 + imgptr1[in_j2]*w12*s2 +
                  imgptr2[in_j0]*w20*s0 + imgptr2[in_j1]*w21*s1 + imgptr2[in_j2]*w22*s2 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[out_j] = out;
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

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_RVV

void fastGEMM( const float* aptr, size_t astep, const float* bptr,
               size_t bstep, float* cptr, size_t cstep,
               int ma, int na, int nb )
{
    int n = 0;
    int vl = vsetvlmax_e32m4();
    int mvl = vl;
    for( ; n < nb; n += vl )
    {
        if ( n + vl > nb) {
            mvl = nb - n;
        }

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

            vfloat32m4_t d0 = vfmv_v_f_f32m4(0, vl);
            vfloat32m4_t d1 = vfmv_v_f_f32m4(0, vl);
            vfloat32m4_t d2 = vfmv_v_f_f32m4(0, vl);
            vfloat32m4_t d3 = vfmv_v_f_f32m4(0, vl);
            vfloat32m4_t d4 = vfmv_v_f_f32m4(0, vl);
            vfloat32m4_t d5 = vfmv_v_f_f32m4(0, vl);
            vfloat32m4_t d6 = vfmv_v_f_f32m4(0, vl);

            for( int k = 0; k < na; k++ )
            {
                float32_t a0 = aptr0[k];
                float32_t a1 = aptr1[k];
                float32_t a2 = aptr2[k];
                float32_t a3 = aptr3[k];
                float32_t a4 = aptr4[k];
                float32_t a5 = aptr5[k];
                float32_t a6 = aptr6[k];

                vfloat32m4_t b = vle32_v_f32m4(bptr + k*bstep + n, mvl);
                d0 = vfmacc_vf_f32m4(d0, a0, b, mvl);
                d1 = vfmacc_vf_f32m4(d1, a1, b, mvl);
                d2 = vfmacc_vf_f32m4(d2, a2, b, mvl);
                d3 = vfmacc_vf_f32m4(d3, a3, b, mvl);
                d4 = vfmacc_vf_f32m4(d4, a4, b, mvl);
                d5 = vfmacc_vf_f32m4(d5, a5, b, mvl);
                d6 = vfmacc_vf_f32m4(d6, a6, b, mvl);
            }
            vse32_v_f32m4(cptr0 + n, d0, mvl);
            vse32_v_f32m4(cptr1 + n, d1, mvl);
            vse32_v_f32m4(cptr2 + n, d2, mvl);
            vse32_v_f32m4(cptr3 + n, d3, mvl);
            vse32_v_f32m4(cptr4 + n, d4, mvl);
            vse32_v_f32m4(cptr5 + n, d5, mvl);
            vse32_v_f32m4(cptr6 + n, d6, mvl);
        }
    }
}

void fastGEMM1T( const float* vec, const float* weights,
                 size_t wstep, const float* bias,
                 float* dst, int nvecs, int vecsize )
{
    int vlm2 = vsetvlmax_e32m2();
    int i = 0;
    for( ; i <= nvecs - 15; i += 15 )
    {
        const float* wptr = weights + i*wstep;
        vfloat32m2_t
               vs0 = vfmv_v_f_f32m2(0, vlm2), vs1 = vfmv_v_f_f32m2(0, vlm2), vs2 = vfmv_v_f_f32m2(0, vlm2),
               vs3 = vfmv_v_f_f32m2(0, vlm2), vs4 = vfmv_v_f_f32m2(0, vlm2), vs5 = vfmv_v_f_f32m2(0, vlm2),
               vs6 = vfmv_v_f_f32m2(0, vlm2), vs7 = vfmv_v_f_f32m2(0, vlm2), vs8 = vfmv_v_f_f32m2(0, vlm2),
               vs9 = vfmv_v_f_f32m2(0, vlm2), vs10 = vfmv_v_f_f32m2(0, vlm2), vs11 = vfmv_v_f_f32m2(0, vlm2),
               vs12 = vfmv_v_f_f32m2(0, vlm2), vs13 = vfmv_v_f_f32m2(0, vlm2), vs14 = vfmv_v_f_f32m2(0, vlm2);
        int k = 0;
        for( ; k < vecsize - vlm2; k += vlm2, wptr += vlm2 )
        {
            vfloat32m2_t v = vle32_v_f32m2(vec + k, vlm2);

            vs0 = vfmacc_vv_f32m2(vs0, vle32_v_f32m2(wptr, vlm2), v, vlm2);
            vs1 = vfmacc_vv_f32m2(vs1, vle32_v_f32m2(wptr + wstep, vlm2), v, vlm2);
            vs2 = vfmacc_vv_f32m2(vs2, vle32_v_f32m2(wptr + wstep*2, vlm2), v, vlm2);
            vs3 = vfmacc_vv_f32m2(vs3, vle32_v_f32m2(wptr + wstep*3, vlm2), v, vlm2);
            vs4 = vfmacc_vv_f32m2(vs4, vle32_v_f32m2(wptr + wstep*4, vlm2), v, vlm2);
            vs5 = vfmacc_vv_f32m2(vs5, vle32_v_f32m2(wptr + wstep*5, vlm2), v, vlm2);
            vs6 = vfmacc_vv_f32m2(vs6, vle32_v_f32m2(wptr + wstep*6, vlm2), v, vlm2);
            vs7 = vfmacc_vv_f32m2(vs7, vle32_v_f32m2(wptr + wstep*7, vlm2), v, vlm2);
            vs8 = vfmacc_vv_f32m2(vs8, vle32_v_f32m2(wptr + wstep*8, vlm2), v, vlm2);
            vs9 = vfmacc_vv_f32m2(vs9, vle32_v_f32m2(wptr + wstep*9, vlm2), v, vlm2);
            vs10 = vfmacc_vv_f32m2(vs10, vle32_v_f32m2(wptr + wstep*10, vlm2), v, vlm2);
            vs11 = vfmacc_vv_f32m2(vs11, vle32_v_f32m2(wptr + wstep*11, vlm2), v, vlm2);
            vs12 = vfmacc_vv_f32m2(vs12, vle32_v_f32m2(wptr + wstep*12, vlm2), v, vlm2);
            vs13 = vfmacc_vv_f32m2(vs13, vle32_v_f32m2(wptr + wstep*13, vlm2), v, vlm2);
            vs14 = vfmacc_vv_f32m2(vs14, vle32_v_f32m2(wptr + wstep*14, vlm2), v, vlm2);
        }
        int kvl = vecsize - k;
        if (kvl > 0) {
            vfloat32m2_t v = vle32_v_f32m2(vec + k, kvl);
            vs0 = vfmacc_vv_f32m2(vs0, vle32_v_f32m2(wptr, kvl), v, kvl);
            vs1 = vfmacc_vv_f32m2(vs1, vle32_v_f32m2(wptr + wstep*1, kvl), v, kvl);
            vs2 = vfmacc_vv_f32m2(vs2, vle32_v_f32m2(wptr + wstep*2, kvl), v, kvl);
            vs3 = vfmacc_vv_f32m2(vs3, vle32_v_f32m2(wptr + wstep*3, kvl), v, kvl);
            vs4 = vfmacc_vv_f32m2(vs4, vle32_v_f32m2(wptr + wstep*4, kvl), v, kvl);
            vs5 = vfmacc_vv_f32m2(vs5, vle32_v_f32m2(wptr + wstep*5, kvl), v, kvl);
            vs6 = vfmacc_vv_f32m2(vs6, vle32_v_f32m2(wptr + wstep*6, kvl), v, kvl);
            vs7 = vfmacc_vv_f32m2(vs7, vle32_v_f32m2(wptr + wstep*7, kvl), v, kvl);
            vs8 = vfmacc_vv_f32m2(vs8, vle32_v_f32m2(wptr + wstep*8, kvl), v, kvl);
            vs9 = vfmacc_vv_f32m2(vs9, vle32_v_f32m2(wptr + wstep*9, kvl), v, kvl);
            vs10 = vfmacc_vv_f32m2(vs10, vle32_v_f32m2(wptr + wstep*10, kvl), v, kvl);
            vs11 = vfmacc_vv_f32m2(vs11, vle32_v_f32m2(wptr + wstep*11, kvl), v, kvl);
            vs12 = vfmacc_vv_f32m2(vs12, vle32_v_f32m2(wptr + wstep*12, kvl), v, kvl);
            vs13 = vfmacc_vv_f32m2(vs13, vle32_v_f32m2(wptr + wstep*13, kvl), v, kvl);
            vs14 = vfmacc_vv_f32m2(vs14, vle32_v_f32m2(wptr + wstep*14, kvl), v, kvl);
        }

        // Calculate the sum of each vector
        float32_t sum[15];
        vfloat32m1_t zero = vfmv_v_f_f32m1(0, vlm2);
        sum[0] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs0, zero, vlm2));
        sum[1] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs1, zero, vlm2));
        sum[2] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs2, zero, vlm2));
        sum[3] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs3, zero, vlm2));
        sum[4] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs4, zero, vlm2));
        sum[5] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs5, zero, vlm2));
        sum[6] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs6, zero, vlm2));
        sum[7] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs7, zero, vlm2));
        sum[8] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs8, zero, vlm2));
        sum[9] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs9, zero, vlm2));
        sum[10] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs10, zero, vlm2));
        sum[11] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs11, zero, vlm2));
        sum[12] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs12, zero, vlm2));
        sum[13] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs13, zero, vlm2));
        sum[14] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs14, zero, vlm2));

        vfloat32m4_t s0 = vfadd_vv_f32m4(vle32_v_f32m4(sum, 15), vle32_v_f32m4(bias + i, 15), 15);
        vse32_v_f32m4(dst + i, s0, 15);
    }
    int mvl = nvecs - i;
    if (mvl > 0)
    {
        const float* wptr = weights + i*wstep;
        vfloat32m2_t
               vs0 = vfmv_v_f_f32m2(0, vlm2), vs1 = vfmv_v_f_f32m2(0, vlm2), vs2 = vfmv_v_f_f32m2(0, vlm2),
               vs3 = vfmv_v_f_f32m2(0, vlm2), vs4 = vfmv_v_f_f32m2(0, vlm2), vs5 = vfmv_v_f_f32m2(0, vlm2),
               vs6 = vfmv_v_f_f32m2(0, vlm2), vs7 = vfmv_v_f_f32m2(0, vlm2), vs8 = vfmv_v_f_f32m2(0, vlm2),
               vs9 = vfmv_v_f_f32m2(0, vlm2), vs10 = vfmv_v_f_f32m2(0, vlm2), vs11 = vfmv_v_f_f32m2(0, vlm2),
               vs12 = vfmv_v_f_f32m2(0, vlm2), vs13 = vfmv_v_f_f32m2(0, vlm2);
        int k = 0;
        for( ; k <= vecsize - vlm2; k += vlm2, wptr += vlm2 )
        {
            vfloat32m2_t v = vle32_v_f32m2(vec + k, vlm2);
            vs0 = vfmacc_vv_f32m2(vs0, vle32_v_f32m2(wptr, vlm2), v, vlm2);
            vs1 = vfmacc_vv_f32m2(vs1, vle32_v_f32m2(wptr + wstep*std::min(1, mvl-1), vlm2), v, vlm2);
            vs2 = vfmacc_vv_f32m2(vs2, vle32_v_f32m2(wptr + wstep*std::min(2, mvl-1), vlm2), v, vlm2);
            vs3 = vfmacc_vv_f32m2(vs3, vle32_v_f32m2(wptr + wstep*std::min(3, mvl-1), vlm2), v, vlm2);
            vs4 = vfmacc_vv_f32m2(vs4, vle32_v_f32m2(wptr + wstep*std::min(4, mvl-1), vlm2), v, vlm2);
            vs5 = vfmacc_vv_f32m2(vs5, vle32_v_f32m2(wptr + wstep*std::min(5, mvl-1), vlm2), v, vlm2);
            vs6 = vfmacc_vv_f32m2(vs6, vle32_v_f32m2(wptr + wstep*std::min(6, mvl-1), vlm2), v, vlm2);
            vs7 = vfmacc_vv_f32m2(vs7, vle32_v_f32m2(wptr + wstep*std::min(7, mvl-1), vlm2), v, vlm2);
            vs8 = vfmacc_vv_f32m2(vs8, vle32_v_f32m2(wptr + wstep*std::min(8, mvl-1), vlm2), v, vlm2);
            vs9 = vfmacc_vv_f32m2(vs9, vle32_v_f32m2(wptr + wstep*std::min(9, mvl-1), vlm2), v, vlm2);
            vs10 = vfmacc_vv_f32m2(vs10, vle32_v_f32m2(wptr + wstep*std::min(10, mvl-1), vlm2), v, vlm2);
            vs11 = vfmacc_vv_f32m2(vs11, vle32_v_f32m2(wptr + wstep*std::min(11, mvl-1), vlm2), v, vlm2);
            vs12 = vfmacc_vv_f32m2(vs12, vle32_v_f32m2(wptr + wstep*std::min(12, mvl-1), vlm2), v, vlm2);
            vs13 = vfmacc_vv_f32m2(vs13, vle32_v_f32m2(wptr + wstep*std::min(13, mvl-1), vlm2), v, vlm2);
        }
        int kvl = vecsize - k;
        if (kvl > 0) {
            vfloat32m2_t v = vle32_v_f32m2(vec + k, kvl);
            vs0 = vfmacc_vv_f32m2(vs0, vle32_v_f32m2(wptr, kvl), v, kvl);
            vs1 = vfmacc_vv_f32m2(vs1, vle32_v_f32m2(wptr + wstep*std::min(1, mvl-1), kvl), v, kvl);
            vs2 = vfmacc_vv_f32m2(vs2, vle32_v_f32m2(wptr + wstep*std::min(2, mvl-1), kvl), v, kvl);
            vs3 = vfmacc_vv_f32m2(vs3, vle32_v_f32m2(wptr + wstep*std::min(3, mvl-1), kvl), v, kvl);
            vs4 = vfmacc_vv_f32m2(vs4, vle32_v_f32m2(wptr + wstep*std::min(4, mvl-1), kvl), v, kvl);
            vs5 = vfmacc_vv_f32m2(vs5, vle32_v_f32m2(wptr + wstep*std::min(5, mvl-1), kvl), v, kvl);
            vs6 = vfmacc_vv_f32m2(vs6, vle32_v_f32m2(wptr + wstep*std::min(6, mvl-1), kvl), v, kvl);
            vs7 = vfmacc_vv_f32m2(vs7, vle32_v_f32m2(wptr + wstep*std::min(7, mvl-1), kvl), v, kvl);
            vs8 = vfmacc_vv_f32m2(vs8, vle32_v_f32m2(wptr + wstep*std::min(8, mvl-1), kvl), v, kvl);
            vs9 = vfmacc_vv_f32m2(vs9, vle32_v_f32m2(wptr + wstep*std::min(9, mvl-1), kvl), v, kvl);
            vs10 = vfmacc_vv_f32m2(vs10, vle32_v_f32m2(wptr + wstep*std::min(10, mvl-1), kvl), v, kvl);
            vs11 = vfmacc_vv_f32m2(vs11, vle32_v_f32m2(wptr + wstep*std::min(11, mvl-1), kvl), v, kvl);
            vs12 = vfmacc_vv_f32m2(vs12, vle32_v_f32m2(wptr + wstep*std::min(12, mvl-1), kvl), v, kvl);
            vs13 = vfmacc_vv_f32m2(vs13, vle32_v_f32m2(wptr + wstep*std::min(13, mvl-1), kvl), v, kvl);
        }
        // Calculate the sum of each vector
        float32_t sum[14];
        vfloat32m1_t zero = vfmv_v_f_f32m1(0, vlm2);
        sum[0] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs0, zero, vlm2));
        sum[1] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs1, zero, vlm2));
        sum[2] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs2, zero, vlm2));
        sum[3] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs3, zero, vlm2));
        sum[4] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs4, zero, vlm2));
        sum[5] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs5, zero, vlm2));
        sum[6] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs6, zero, vlm2));
        sum[7] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs7, zero, vlm2));
        sum[8] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs8, zero, vlm2));
        sum[9] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs9, zero, vlm2));
        sum[10] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs10, zero, vlm2));
        sum[11] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs11, zero, vlm2));
        sum[12] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs12, zero, vlm2));
        sum[13] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m2_f32m1(zero, vs13, zero, vlm2));

        vfloat32m4_t s0 = vfadd_vv_f32m4(vle32_v_f32m4(sum, mvl), vle32_v_f32m4(bias + i, mvl), mvl);
        vse32_v_f32m4(dst + i, s0, mvl);
    }
}

enum { FASCONV_BASE_VECSZ = 8 };
void fastConv( const float* weights, size_t wstep, const float* bias,
               const float* rowbuf, float* output, const int* outShape,
               int blockSize, int vecsize, int vecsize_aligned,
               const float* relu, bool initOutput )
{
    int vl = FASCONV_BASE_VECSZ;
    int vlm1Max = vsetvlmax_e32m1();
    int outCn = outShape[1];
    size_t outPlaneSize = outShape[2]*outShape[3];
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

        int j = 0;
        for( ; j < blockSize; j += FASCONV_BASE_VECSZ )
        {
            bool tail = false;
            if (j + FASCONV_BASE_VECSZ > blockSize)
            {
                if (j == 0) {
                    vl = blockSize;
                }
                else {
                    j = blockSize - FASCONV_BASE_VECSZ;
                    tail = true;
                }
            }
            int k = 0;
            const float* rptr = rowbuf + j*vecsize_aligned;
            int vlm1 = vsetvlmax_e32m1();
            vfloat32m1_t
                vs00 = vfmv_v_f_f32m1(0, vlm1), vs10 = vfmv_v_f_f32m1(0, vlm1), vs20 = vfmv_v_f_f32m1(0, vlm1),
                vs01 = vfmv_v_f_f32m1(0, vlm1), vs11 = vfmv_v_f_f32m1(0, vlm1), vs21 = vfmv_v_f_f32m1(0, vlm1),
                vs02 = vfmv_v_f_f32m1(0, vlm1), vs12 = vfmv_v_f_f32m1(0, vlm1), vs22 = vfmv_v_f_f32m1(0, vlm1),
                vs03 = vfmv_v_f_f32m1(0, vlm1), vs13 = vfmv_v_f_f32m1(0, vlm1), vs23 = vfmv_v_f_f32m1(0, vlm1),
                vs04 = vfmv_v_f_f32m1(0, vlm1), vs14 = vfmv_v_f_f32m1(0, vlm1), vs24 = vfmv_v_f_f32m1(0, vlm1),
                vs05 = vfmv_v_f_f32m1(0, vlm1), vs15 = vfmv_v_f_f32m1(0, vlm1), vs25 = vfmv_v_f_f32m1(0, vlm1),
                vs06 = vfmv_v_f_f32m1(0, vlm1), vs16 = vfmv_v_f_f32m1(0, vlm1), vs26 = vfmv_v_f_f32m1(0, vlm1),
                vs07 = vfmv_v_f_f32m1(0, vlm1), vs17 = vfmv_v_f_f32m1(0, vlm1), vs27 = vfmv_v_f_f32m1(0, vlm1);

            for (; k < vecsize; k += vlm1, rptr += vlm1 )
            {
                if (k + vlm1 >= vecsize) {
                    vlm1 = vecsize - k;
                }
                vfloat32m1_t w0 = vle32_v_f32m1(wptr0 + k, vlm1);
                vfloat32m1_t w1 = vle32_v_f32m1(wptr1 + k, vlm1);
                vfloat32m1_t w2 = vle32_v_f32m1(wptr2 + k, vlm1);
                vfloat32m1_t r0 = vle32_v_f32m1(rptr, vlm1);

                vs00 = vfmacc_vv_f32m1(vs00, w0, r0, vlm1);
                vs10 = vfmacc_vv_f32m1(vs10, w1, r0, vlm1);
                vs20 = vfmacc_vv_f32m1(vs20, w2, r0, vlm1);

                r0 = vle32_v_f32m1(rptr + vecsize_aligned, vlm1);
                vs01 = vfmacc_vv_f32m1(vs01, w0, r0, vlm1);
                vs11 = vfmacc_vv_f32m1(vs11, w1, r0, vlm1);
                vs21 = vfmacc_vv_f32m1(vs21, w2, r0, vlm1);

                r0 = vle32_v_f32m1(rptr + vecsize_aligned*2, vlm1);
                vs02 = vfmacc_vv_f32m1(vs02, w0, r0, vlm1);
                vs12 = vfmacc_vv_f32m1(vs12, w1, r0, vlm1);
                vs22 = vfmacc_vv_f32m1(vs22, w2, r0, vlm1);

                r0 = vle32_v_f32m1(rptr + vecsize_aligned*3, vlm1);
                vs03 = vfmacc_vv_f32m1(vs03, w0, r0, vlm1);
                vs13 = vfmacc_vv_f32m1(vs13, w1, r0, vlm1);
                vs23 = vfmacc_vv_f32m1(vs23, w2, r0, vlm1);

                r0 = vle32_v_f32m1(rptr + vecsize_aligned*4, vlm1);
                vs04 = vfmacc_vv_f32m1(vs04, w0, r0, vlm1);
                vs14 = vfmacc_vv_f32m1(vs14, w1, r0, vlm1);
                vs24 = vfmacc_vv_f32m1(vs24, w2, r0, vlm1);

                r0 = vle32_v_f32m1(rptr + vecsize_aligned*5, vlm1);
                vs05 = vfmacc_vv_f32m1(vs05, w0, r0, vlm1);
                vs15 = vfmacc_vv_f32m1(vs15, w1, r0, vlm1);
                vs25 = vfmacc_vv_f32m1(vs25, w2, r0, vlm1);

                r0 = vle32_v_f32m1(rptr + vecsize_aligned*6, vlm1);
                vs06 = vfmacc_vv_f32m1(vs06, w0, r0, vlm1);
                vs16 = vfmacc_vv_f32m1(vs16, w1, r0, vlm1);
                vs26 = vfmacc_vv_f32m1(vs26, w2, r0, vlm1);

                r0 = vle32_v_f32m1(rptr + vecsize_aligned*7, vlm1);
                vs07 = vfmacc_vv_f32m1(vs07, w0, r0, vlm1);
                vs17 = vfmacc_vv_f32m1(vs17, w1, r0, vlm1);
                vs27 = vfmacc_vv_f32m1(vs27, w2, r0, vlm1);
            }

            // compute sum of each vs
            vfloat32m1_t zero = vfmv_v_f_f32m1(0, vlm1Max);
            // vl is required here to be at least FASCONV_BASE_VECSZ, aka 8.
            float32_t sum0[FASCONV_BASE_VECSZ], sum1[FASCONV_BASE_VECSZ], sum2[FASCONV_BASE_VECSZ];
            sum0[0] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs00, zero, vlm1Max));
            sum0[1] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs01, zero, vlm1Max));
            sum0[2] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs02, zero, vlm1Max));
            sum0[3] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs03, zero, vlm1Max));
            sum0[4] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs04, zero, vlm1Max));
            sum0[5] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs05, zero, vlm1Max));
            sum0[6] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs06, zero, vlm1Max));
            sum0[7] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs07, zero, vlm1Max));
            sum1[0] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs10, zero, vlm1Max));
            sum1[1] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs11, zero, vlm1Max));
            sum1[2] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs12, zero, vlm1Max));
            sum1[3] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs13, zero, vlm1Max));
            sum1[4] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs14, zero, vlm1Max));
            sum1[5] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs15, zero, vlm1Max));
            sum1[6] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs16, zero, vlm1Max));
            sum1[7] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs17, zero, vlm1Max));
            sum2[0] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs20, zero, vlm1Max));
            sum2[1] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs21, zero, vlm1Max));
            sum2[2] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs22, zero, vlm1Max));
            sum2[3] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs23, zero, vlm1Max));
            sum2[4] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs24, zero, vlm1Max));
            sum2[5] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs25, zero, vlm1Max));
            sum2[6] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs26, zero, vlm1Max));
            sum2[7] = vfmv_f_s_f32m1_f32(vfredsum_vs_f32m1_f32m1(zero, vs27, zero, vlm1Max));

            // if VLEN = 128, so LMUL = 2 for vl = 8.
            // otherwise, VLEN >=256, we only use fist 8 element of the vReg.
            vfloat32m2_t s0, s1, s2;
            if( initOutput )
            {
                s0 = vfmv_v_f_f32m2(bias0, vl);
                s1 = vfmv_v_f_f32m2(bias1, vl);
                s2 = vfmv_v_f_f32m2(bias2, vl);
            }
            else
            {
                s0 = vle32_v_f32m2(outptr0 + j, vl);
                s1 = vle32_v_f32m2(outptr1 + j, vl);
                s2 = vle32_v_f32m2(outptr2 + j, vl);
            }
            s0 = vfadd_vv_f32m2(vle32_v_f32m2(sum0, vl), s0, vl);
            s1 = vfadd_vv_f32m2(vle32_v_f32m2(sum1, vl), s1, vl);
            s2 = vfadd_vv_f32m2(vle32_v_f32m2(sum2, vl), s2, vl);

            if( relu )
            {
                vfloat32m2_t vr0 = vfmv_v_f_f32m2(1, vl), vr1 = vfmv_v_f_f32m2(1, vl), vr2 = vfmv_v_f_f32m2(1, vl);
                float r0 = relu[i], r1 = relu[i+1], r2 = relu[i+2];
                if( i+2 >= outCn )
                {
                    r2 = r1;
                    if( i+1 >= outCn )
                        r2 = r1 = r0;
                }
                vr0 = vfmv_v_f_f32m2(r0, vl);
                vr1 = vfmv_v_f_f32m2(r1, vl);
                vr2 = vfmv_v_f_f32m2(r2, vl);
                vbool16_t m0 = vmfgt_vf_f32m2_b16(s0, 0, vl);
                vbool16_t m1 = vmfgt_vf_f32m2_b16(s1, 0, vl);
                vbool16_t m2 = vmfgt_vf_f32m2_b16(s2, 0, vl);
                s0 = vmerge_vvm_f32m2(m0, vfmul_vv_f32m2(s0, vr0, vl), s0, vl);
                s1 = vmerge_vvm_f32m2(m1, vfmul_vv_f32m2(s1, vr1, vl), s1, vl);
                s2 = vmerge_vvm_f32m2(m2, vfmul_vv_f32m2(s2, vr2, vl), s2, vl);
            }

            if( tail )
            {
                int maskbuf[FASCONV_BASE_VECSZ] = {0};
                int rsz = blockSize % FASCONV_BASE_VECSZ;
                for( int i = 0; i < rsz; i++ )
                    maskbuf[FASCONV_BASE_VECSZ - i - 1] = -1;
                vint32m2_t vmaskbuf = vle32_v_i32m2(maskbuf ,vl);
                vbool16_t mask = vmslt_vx_i32m2_b16(vmaskbuf, 0, vl); // mask for tail
                s0 = vmerge_vvm_f32m2(mask, vle32_v_f32m2(outptr0 + j, vl), s0, vl);
                s1 = vmerge_vvm_f32m2(mask, vle32_v_f32m2(outptr1 + j, vl), s1, vl);
                s2 = vmerge_vvm_f32m2(mask, vle32_v_f32m2(outptr2 + j, vl), s2, vl);
            }

            vse32_v_f32m2(outptr0 + j, s0, vl);
            vse32_v_f32m2(outptr1 + j, s1, vl);
            vse32_v_f32m2(outptr2 + j, s2, vl);
        }
    }
}

/*
Example for load_deinterleave:
    input: ptr[16] = {1,2,3, ... ,14,15,16}
    output: a = {1, 3, 5, 7, 9, 11, 13, 15}
    output: b = {2, 4, 6, 8,10, 12, 14, 16}
*/
static inline void vfloat32m2_load_deinterleave(const float* ptr, vfloat32m2_t& a, vfloat32m2_t& b, int vl)
{
    vuint64m4_t mask = vmv_v_x_u64m4(1,vl*2);
    vuint32m4_t mask_re = vreinterpret_v_u64m4_u32m4(mask);
    vbool8_t mask0 = vmseq_vx_u32m4_b8 (mask_re, 1, vl*2);
    vbool8_t mask1 = vmseq_vx_u32m4_b8 (mask_re, 0, vl*2);
    vfloat32m4_t tempa = vundefined_f32m4(), tempb = vundefined_f32m4();
    vfloat32m4_t vw = vle32_v_f32m4(ptr, vl*2);
    tempa = vcompress_vm_f32m4(mask0, tempa, vw, vl*2);
    tempb = vcompress_vm_f32m4(mask1, tempb, vw, vl*2);
    /* The following instructions have not to be supported by the GNU toolchain.
       So we temporarily use store and load instead.
    // a = vlmul_trunc_v_f32m4_f32m2(tempa);
    // b = vlmul_trunc_v_f32m4_f32m2(tempb);
    */
    cv::AutoBuffer<float> cvBuffer(sizeof(float32_t)*vl*2);
    float* buffer = (float*)cvBuffer.data();
    vse32_v_f32m4(buffer, tempa, vl);
    a = vle32_v_f32m2(buffer, vl);
    vse32_v_f32m4(buffer, tempb, vl);
    b = vle32_v_f32m2(buffer, vl);
}

void fastDepthwiseConv( const float* wptr,
                     int kernel_h, int kernel_w,
                     int stride_h, int stride_w,
                     int dilation_h, int dilation_w,
                     int pad_t, int pad_l,
                     const float* biasptr, const float* relu,
                     const float* inptr_,
                     int height, int width,
                     float* outptr_,
                     int out_d, int outH, int outW )
{
    int vl = vsetvlmax_e32m2();
    const float w00_ = wptr[0], w01_ = wptr[1], w02_ = wptr[2],
                w10 = wptr[3], w11 = wptr[4], w12 = wptr[5],
                w20_ = wptr[6], w21_ = wptr[7], w22_ = wptr[8];
    int outW1 = std::min(outW, (width - dilation_w*(kernel_w - 1) + pad_l)/stride_w);
    float relu_coeff = relu ? relu[out_d] : 1.f, bias = biasptr[out_d];

    for (int out_i = 0; out_i < outH; out_i++)
    {
        int in_i = out_i * stride_h - pad_t, out_j = 0;
        const float* imgptr0 = inptr_ + in_i*width;
        const float* imgptr1 = imgptr0 + dilation_h*width;
        const float* imgptr2 = imgptr0 + (dilation_h*2)*width;
        float out, w00 = w00_, w01 = w01_, w02 = w02_;
        float w20 = w20_, w21 = w21_, w22 = w22_;
        if (in_i < 0)
        {
            w00 = w01 = w02 = 0.f;
            imgptr0 = imgptr1;
        }
        else if (in_i + dilation_h*(kernel_h-1) >= height)
        {
            w20 = w21 = w22 = 0.f;
            imgptr2 = imgptr1;
        }
        float* outptr = outptr_ + out_i*outW;
        if (pad_l > 0)
        {
            out = imgptr0[0]*w01 + imgptr0[dilation_w]*w02 +
                  imgptr1[0]*w11 + imgptr1[dilation_w]*w12 +
                  imgptr2[0]*w21 + imgptr2[dilation_w]*w22 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[0] = out;
            out_j = 1;
        }

        if (stride_w == 1 || (stride_w == 2 && dilation_w == 1))
        {
            if( stride_w == 1 )
                for( ; out_j < outW1; out_j += vl )
                {
                    if (out_j + vl > outW1)
                        vl = outW1 - out_j;
                    int in_j = out_j * stride_w - pad_l;
                    vfloat32m2_t v00 = vle32_v_f32m2(imgptr0 + in_j, vl),
                           v01 = vle32_v_f32m2(imgptr0 + in_j + dilation_w, vl),
                           v02 = vle32_v_f32m2(imgptr0 + in_j + dilation_w*2, vl),
                           v10 = vle32_v_f32m2(imgptr1 + in_j, vl),
                           v11 = vle32_v_f32m2(imgptr1 + in_j + dilation_w, vl),
                           v12 = vle32_v_f32m2(imgptr1 + in_j + dilation_w*2, vl),
                           v20 = vle32_v_f32m2(imgptr2 + in_j, vl),
                           v21 = vle32_v_f32m2(imgptr2 + in_j + dilation_w, vl),
                           v22 = vle32_v_f32m2(imgptr2 + in_j + dilation_w*2, vl);

                    vfloat32m2_t vout0 = vfmul_vf_f32m2(v00, w00, vl);
                    vfloat32m2_t vout1 = vfmul_vf_f32m2(v01, w01, vl);
                    vfloat32m2_t vout2 = vfmul_vf_f32m2(v02, w02, vl);
                    vout0 = vfadd_vf_f32m2(vout0, bias, vl);

                    vout0 = vfmacc_vf_f32m2(vout0, w10, v10, vl);
                    vout1 = vfmacc_vf_f32m2(vout1, w11, v11, vl);
                    vout2 = vfmacc_vf_f32m2(vout2, w12, v12, vl);

                    vout0 = vfmacc_vf_f32m2(vout0, w20, v20, vl);
                    vout1 = vfmacc_vf_f32m2(vout1, w21, v21, vl);
                    vout2 = vfmacc_vf_f32m2(vout2, w22, v22, vl);

                    vout0 = vfadd_vv_f32m2(vfadd_vv_f32m2(vout0, vout1, vl), vout2, vl);
                    if (relu)
                    {
                        vbool16_t m = vmfgt_vf_f32m2_b16(vout0, 0, vl);
                        vout0 = vmerge_vvm_f32m2(m, vfmul_vf_f32m2(vout0, relu_coeff, vl), vout0, vl);
                    }
                    vse32_v_f32m2(outptr + out_j, vout0, vl);
                }
            else //stride_w == 2 && dilation_w == 1
                for( ; out_j < outW1; out_j += vl )
                {
                    if (out_j + vl > outW1)
                        vl = outW1 - out_j;
                    int in_j = out_j * stride_w - pad_l;
                    vfloat32m2_t v00, v01, v02, v10, v11, v12, v20, v21, v22, unused;
                    vfloat32m2_load_deinterleave(imgptr0 + in_j, v00, v01, vl);
                    vfloat32m2_load_deinterleave(imgptr0 + in_j + 2, v02, unused, vl);
                    vfloat32m2_load_deinterleave(imgptr1 + in_j, v10, v11, vl);
                    vfloat32m2_load_deinterleave(imgptr1 + in_j + 2, v12, unused, vl);
                    vfloat32m2_load_deinterleave(imgptr2 + in_j, v20, v21, vl);
                    vfloat32m2_load_deinterleave(imgptr2 + in_j + 2, v22, unused, vl);

                    vfloat32m2_t vout0 = vfmul_vf_f32m2(v00, w00, vl);
                    vfloat32m2_t vout1 = vfmul_vf_f32m2(v01, w01, vl);
                    vfloat32m2_t vout2 = vfmul_vf_f32m2(v02, w02, vl);
                    vout0 = vfadd_vf_f32m2(vout0, bias, vl);

                    vout0 = vfmacc_vf_f32m2(vout0, w10, v10, vl);
                    vout1 = vfmacc_vf_f32m2(vout1, w11, v11, vl);
                    vout2 = vfmacc_vf_f32m2(vout2, w12, v12, vl);

                    vout0 = vfmacc_vf_f32m2(vout0, w20, v20, vl);
                    vout1 = vfmacc_vf_f32m2(vout1, w21, v21, vl);
                    vout2 = vfmacc_vf_f32m2(vout2, w22, v22, vl);

                    vout0 = vfadd_vv_f32m2(vfadd_vv_f32m2(vout0, vout1, vl), vout2, vl);
                    if (relu)
                    {
                        vbool16_t m = vmfgt_vf_f32m2_b16(vout0, 0, vl);
                        vout0 = vmerge_vvm_f32m2(m, vfmul_vf_f32m2(vout0, relu_coeff, vl), vout0, vl);
                    }
                    vse32_v_f32m2(outptr + out_j, vout0, vl);
                }
        }

        for (; out_j < outW1; out_j++)
        {
            int in_j = out_j * stride_w - pad_l;
            out = imgptr0[in_j]*w00 + imgptr0[in_j + dilation_w]*w01 + imgptr0[in_j + dilation_w*2]*w02 +
                  imgptr1[in_j]*w10 + imgptr1[in_j + dilation_w]*w11 + imgptr1[in_j + dilation_w*2]*w12 +
                  imgptr2[in_j]*w20 + imgptr2[in_j + dilation_w]*w21 + imgptr2[in_j + dilation_w*2]*w22 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[out_j] = out;
        }

        for (; out_j < outW; out_j++ )
        {
            int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w*2;
            float s0 = 1.f, s1 = 1.f, s2 = 1.f;
            if (in_j0 >= width)
            {
                in_j0 = 0;
                s0 = 0.f;
            }
            if (in_j1 >= width)
            {
                in_j1 = 0;
                s1 = 0.f;
            }
            if (in_j2 >= width)
            {
                in_j2 = 0;
                s2 = 0.f;
            }
            out = imgptr0[in_j0]*w00*s0 + imgptr0[in_j1]*w01*s1 + imgptr0[in_j2]*w02*s2 +
                  imgptr1[in_j0]*w10*s0 + imgptr1[in_j1]*w11*s1 + imgptr1[in_j2]*w12*s2 +
                  imgptr2[in_j0]*w20*s0 + imgptr2[in_j1]*w21*s1 + imgptr2[in_j2]*w22*s2 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[out_j] = out;
        }
    }
}

#endif // CV_RVV

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace
