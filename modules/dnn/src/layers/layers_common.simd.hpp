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
    size_t vl = 8;
    size_t mvl0 = 8;
    size_t mvl1 = 8;
    for( ; n < nb; n += 16 )
    {
        if ( n + 16 > nb) {
            mvl0 = nb - n;
            mvl1 = (nb - n -8) > 0 ? (nb - n -8) : 0;
        }

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

            vfloat32m2_t d00 = vfmv_v_f_f32m2(0, vl), d01 = vfmv_v_f_f32m2(0, vl);
            vfloat32m2_t d10 = vfmv_v_f_f32m2(0, vl), d11 = vfmv_v_f_f32m2(0, vl);
            vfloat32m2_t d20 = vfmv_v_f_f32m2(0, vl), d21 = vfmv_v_f_f32m2(0, vl);
            vfloat32m2_t d30 = vfmv_v_f_f32m2(0, vl), d31 = vfmv_v_f_f32m2(0, vl);

            for( int k = 0; k < na; k++ )
            {
                vfloat32m2_t a0 = vfmv_v_f_f32m2(aptr0[k], vl);
                vfloat32m2_t a1 = vfmv_v_f_f32m2(aptr1[k], vl);
                vfloat32m2_t a2 = vfmv_v_f_f32m2(aptr2[k], vl);
                vfloat32m2_t a3 = vfmv_v_f_f32m2(aptr3[k], vl);
                vfloat32m2_t b0 = vle32_v_f32m2(bptr + k*bstep + n, mvl0);
                vfloat32m2_t b1 = vle32_v_f32m2(bptr + k*bstep + n + 8, mvl1);
                d00 = vfmacc_vv_f32m2(d00, a0, b0, mvl0);
                d01 = vfmacc_vv_f32m2(d01, a0, b1, mvl1);
                d10 = vfmacc_vv_f32m2(d10, a1, b0, mvl0);
                d11 = vfmacc_vv_f32m2(d11, a1, b1, mvl1);
                d20 = vfmacc_vv_f32m2(d20, a2, b0, mvl0);
                d21 = vfmacc_vv_f32m2(d21, a2, b1, mvl1);
                d30 = vfmacc_vv_f32m2(d30, a3, b0, mvl0);
                d31 = vfmacc_vv_f32m2(d31, a3, b1, mvl1);
            }
            vse32_v_f32m2(cptr0 + n, d00, mvl0);
            vse32_v_f32m2(cptr1 + n, d10, mvl0);
            vse32_v_f32m2(cptr2 + n, d20, mvl0);
            vse32_v_f32m2(cptr3 + n, d30, mvl0);
            vse32_v_f32m2(cptr0 + n + 8, d01, mvl1);
            vse32_v_f32m2(cptr1 + n + 8, d11, mvl1);
            vse32_v_f32m2(cptr2 + n + 8, d21, mvl1);
            vse32_v_f32m2(cptr3 + n + 8, d31, mvl1);
        }
    }
}

void fastGEMM1T( const float* vec, const float* weights,
                 size_t wstep, const float* bias,
                 float* dst, int nvecs, int vecsize )
{
    int i = 0;
    size_t vl = 8;
    for( ; i <= nvecs - 8; i += 8 )
    {
        const float* wptr = weights + i*wstep;
        vfloat32m2_t vs0 = vfmv_v_f_f32m2(0, vl), vs1 = vfmv_v_f_f32m2(0, vl),
               vs2 = vfmv_v_f_f32m2(0, vl), vs3 = vfmv_v_f_f32m2(0, vl),
               vs4 = vfmv_v_f_f32m2(0, vl), vs5 = vfmv_v_f_f32m2(0, vl),
               vs6 = vfmv_v_f_f32m2(0, vl), vs7 = vfmv_v_f_f32m2(0, vl);

        for( int k = 0; k < vecsize; k += 8, wptr += 8 )
        {
            vfloat32m2_t v = vle32_v_f32m2(vec + k, vl);

            vs0 = vfmacc_vv_f32m2(vs0, vle32_v_f32m2(wptr, vl), v, vl);
            vs1 = vfmacc_vv_f32m2(vs1, vle32_v_f32m2(wptr + wstep, vl), v, vl);
            vs2 = vfmacc_vv_f32m2(vs2, vle32_v_f32m2(wptr + wstep*2, vl), v, vl);
            vs3 = vfmacc_vv_f32m2(vs3, vle32_v_f32m2(wptr + wstep*3, vl), v, vl);
            vs4 = vfmacc_vv_f32m2(vs4, vle32_v_f32m2(wptr + wstep*4, vl), v, vl);
            vs5 = vfmacc_vv_f32m2(vs5, vle32_v_f32m2(wptr + wstep*5, vl), v, vl);
            vs6 = vfmacc_vv_f32m2(vs6, vle32_v_f32m2(wptr + wstep*6, vl), v, vl);
            vs7 = vfmacc_vv_f32m2(vs7, vle32_v_f32m2(wptr + wstep*7, vl), v, vl);
        }

        // Calculate the sum of each vector
        vfloat32m1_t zero = vfmv_v_f_f32m1(0, vl);
        vfloat32m1_t temp0 = vfredsum_vs_f32m2_f32m1(temp0, vs0, zero, vl);
        vfloat32m1_t temp1 = vfredsum_vs_f32m2_f32m1(temp1, vs1, zero, vl);
        vfloat32m1_t temp2 = vfredsum_vs_f32m2_f32m1(temp2, vs2, zero, vl);
        vfloat32m1_t temp3 = vfredsum_vs_f32m2_f32m1(temp3, vs3, zero, vl);
        vfloat32m1_t temp4 = vfredsum_vs_f32m2_f32m1(temp4, vs4, zero, vl);
        vfloat32m1_t temp5 = vfredsum_vs_f32m2_f32m1(temp5, vs5, zero, vl);
        vfloat32m1_t temp6 = vfredsum_vs_f32m2_f32m1(temp6, vs6, zero, vl);
        vfloat32m1_t temp7 = vfredsum_vs_f32m2_f32m1(temp7, vs7, zero, vl);
        float32_t sum[8];
        sum[0] = vfmv_f_s_f32m1_f32(temp0);
        sum[1] = vfmv_f_s_f32m1_f32(temp1);
        sum[2] = vfmv_f_s_f32m1_f32(temp2);
        sum[3] = vfmv_f_s_f32m1_f32(temp3);
        sum[4] = vfmv_f_s_f32m1_f32(temp4);
        sum[5] = vfmv_f_s_f32m1_f32(temp5);
        sum[6] = vfmv_f_s_f32m1_f32(temp6);
        sum[7] = vfmv_f_s_f32m1_f32(temp7);
        vfloat32m2_t s0 = vfadd_vv_f32m2(vle32_v_f32m2(sum, vl), vle32_v_f32m2(bias + i, vl), vl);
        vse32_v_f32m2(dst + i, s0, vl);
    }
    int mvl = nvecs - i;
    if (mvl > 0)
    {
        const float* wptr = weights + i*wstep;
        vfloat32m2_t vs0 = vfmv_v_f_f32m2(0, vl), vs1 = vfmv_v_f_f32m2(0, vl),
               vs2 = vfmv_v_f_f32m2(0, vl), vs3 = vfmv_v_f_f32m2(0, vl),
               vs4 = vfmv_v_f_f32m2(0, vl), vs5 = vfmv_v_f_f32m2(0, vl),
               vs6 = vfmv_v_f_f32m2(0, vl), vs7 = vfmv_v_f_f32m2(0, vl);
        int k = 0;
        for( ; k <= vecsize - 8; k += 8, wptr += 8 )
        {
            vfloat32m2_t v = vle32_v_f32m2(vec + k, vl);
            vs0 = vfmacc_vv_f32m2(vs0, vle32_v_f32m2(wptr, vl), v, vl);
            vs1 = vfmacc_vv_f32m2(vs1, vle32_v_f32m2(wptr + wstep*std::min(1, mvl-1), vl), v, vl);
            vs2 = vfmacc_vv_f32m2(vs2, vle32_v_f32m2(wptr + wstep*std::min(2, mvl-1), vl), v, vl);
            vs3 = vfmacc_vv_f32m2(vs3, vle32_v_f32m2(wptr + wstep*std::min(3, mvl-1), vl), v, vl);
            vs4 = vfmacc_vv_f32m2(vs4, vle32_v_f32m2(wptr + wstep*std::min(4, mvl-1), vl), v, vl);
            vs5 = vfmacc_vv_f32m2(vs5, vle32_v_f32m2(wptr + wstep*std::min(5, mvl-1), vl), v, vl);
            vs6 = vfmacc_vv_f32m2(vs6, vle32_v_f32m2(wptr + wstep*std::min(6, mvl-1), vl), v, vl);
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
        }
        // Calculate the sum of each vector
        vfloat32m1_t zero = vfmv_v_f_f32m1(0, vl);
        vfloat32m1_t temp0 = vfmv_v_f_f32m1(0, 4), temp1 = vfmv_v_f_f32m1(0, 4),
                temp2 = vfmv_v_f_f32m1(0, 4), temp3 = vfmv_v_f_f32m1(0, 4),
                temp4 = vfmv_v_f_f32m1(0, 4), temp5 = vfmv_v_f_f32m1(0, 4),
                temp6 = vfmv_v_f_f32m1(0, 4), temp7 = vfmv_v_f_f32m1(0, 4);
        temp0 = vfredsum_vs_f32m2_f32m1(temp0, vs0, zero, vl);
        temp1 = vfredsum_vs_f32m2_f32m1(temp1, vs1, zero, vl);
        temp2 = vfredsum_vs_f32m2_f32m1(temp2, vs2, zero, vl);
        temp3 = vfredsum_vs_f32m2_f32m1(temp3, vs3, zero, vl);
        temp4 = vfredsum_vs_f32m2_f32m1(temp4, vs4, zero, vl);
        temp5 = vfredsum_vs_f32m2_f32m1(temp5, vs5, zero, vl);
        temp6 = vfredsum_vs_f32m2_f32m1(temp6, vs6, zero, vl);
        temp7 = vfredsum_vs_f32m2_f32m1(temp7, vs7, zero, vl);

        float32_t sum[8];
        sum[0] = vfmv_f_s_f32m1_f32(temp0);
        sum[1] = vfmv_f_s_f32m1_f32(temp1);
        sum[2] = vfmv_f_s_f32m1_f32(temp2);
        sum[3] = vfmv_f_s_f32m1_f32(temp3);
        sum[4] = vfmv_f_s_f32m1_f32(temp4);
        sum[5] = vfmv_f_s_f32m1_f32(temp5);
        sum[6] = vfmv_f_s_f32m1_f32(temp6);
        sum[7] = vfmv_f_s_f32m1_f32(temp7);

        vfloat32m2_t s0 = vfadd_vv_f32m2(vle32_v_f32m2(sum, mvl), vle32_v_f32m2(bias + i, mvl), mvl);
        vse32_v_f32m2(dst + i, s0, mvl);
    }
}

enum { FASCONV_BASE_VECSZ = 4 }; // TODO: Large base size.
void fastConv( const float* weights, size_t wstep, const float* bias,
               const float* rowbuf, float* output, const int* outShape,
               int blockSize, int vecsize, int vecsize_aligned,
               const float* relu, bool initOutput )
{
    int vl = 4;
    int outCn = outShape[1];
    size_t outPlaneSize = outShape[2]*outShape[3];
    float r0 = 1.f, r1 = 1.f, r2 = 1.f;
    vfloat32m1_t vr0 = vfmv_v_f_f32m1(1, vl), vr1 = vfmv_v_f_f32m1(1, vl), vr2 = vfmv_v_f_f32m1(1, vl);
    int maskbuf[FASCONV_BASE_VECSZ] = {0};
    int rsz = blockSize % FASCONV_BASE_VECSZ;
    for( int i = 0; i < rsz; i++ )
        maskbuf[FASCONV_BASE_VECSZ - i - 1] = -1;
    vint32m1_t vmaskbuf = vle32_v_i32m1(maskbuf ,vl);
    vbool32_t mask = vmslt_vx_i32m1_b32(vmaskbuf, 0, vl); // mask for tail
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
            vr0 = vfmv_v_f_f32m1(r0, vl);
            vr1 = vfmv_v_f_f32m1(r1, vl);
            vr2 = vfmv_v_f_f32m1(r2, vl);
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
            int vlm2 = 8;
            vfloat32m2_t vs00 = vfmv_v_f_f32m2(0, vlm2), vs01 = vfmv_v_f_f32m2(0, vlm2),
                   vs02 = vfmv_v_f_f32m2(0, vlm2), vs03 = vfmv_v_f_f32m2(0, vlm2),
                   vs10 = vfmv_v_f_f32m2(0, vlm2), vs11 = vfmv_v_f_f32m2(0, vlm2),
                   vs12 = vfmv_v_f_f32m2(0, vlm2), vs13 = vfmv_v_f_f32m2(0, vlm2),
                   vs20 = vfmv_v_f_f32m2(0, vlm2), vs21 = vfmv_v_f_f32m2(0, vlm2),
                   vs22 = vfmv_v_f_f32m2(0, vlm2), vs23 = vfmv_v_f_f32m2(0, vlm2);

            for (; k < vecsize; k += 8, rptr += 8 )
            {
                if (k+8 >= vecsize) {
                    vlm2 = vecsize - k;
                }
                vfloat32m2_t w0 = vle32_v_f32m2(wptr0 + k, vlm2);
                vfloat32m2_t w1 = vle32_v_f32m2(wptr1 + k, vlm2);
                vfloat32m2_t w2 = vle32_v_f32m2(wptr2 + k, vlm2);
                vfloat32m2_t r0 = vle32_v_f32m2(rptr, vlm2);

                vs00 = vfmacc_vv_f32m2(vs00, w0, r0, vlm2);
                vs10 = vfmacc_vv_f32m2(vs10, w1, r0, vlm2);
                vs20 = vfmacc_vv_f32m2(vs20, w2, r0, vlm2);

                r0 = vle32_v_f32m2(rptr + vecsize_aligned, vlm2);
                vs01 = vfmacc_vv_f32m2(vs01, w0, r0, vlm2);
                vs11 = vfmacc_vv_f32m2(vs11, w1, r0, vlm2);
                vs21 = vfmacc_vv_f32m2(vs21, w2, r0, vlm2);

                r0 = vle32_v_f32m2(rptr + vecsize_aligned*2, vlm2);
                vs02 = vfmacc_vv_f32m2(vs02, w0, r0, vlm2);
                vs12 = vfmacc_vv_f32m2(vs12, w1, r0, vlm2);
                vs22 = vfmacc_vv_f32m2(vs22, w2, r0, vlm2);

                r0 = vle32_v_f32m2(rptr + vecsize_aligned*3, vlm2);
                vs03 = vfmacc_vv_f32m2(vs03, w0, r0, vlm2);
                vs13 = vfmacc_vv_f32m2(vs13, w1, r0, vlm2);
                vs23 = vfmacc_vv_f32m2(vs23, w2, r0, vlm2);
            }
            vfloat32m1_t s0, s1, s2;

            if( initOutput )
            {
                s0 = vfmv_v_f_f32m1(bias0, vl);
                s1 = vfmv_v_f_f32m1(bias1, vl);
                s2 = vfmv_v_f_f32m1(bias2, vl);
            }
            else
            {
                s0 = vle32_v_f32m1(outptr0 + j, vl);
                s1 = vle32_v_f32m1(outptr1 + j, vl);
                s2 = vle32_v_f32m1(outptr2 + j, vl);
            }
            // compute sum of each vs
            vfloat32m1_t zero = vfmv_v_f_f32m1(0, vl);
            vfloat32m1_t temp00 = vfredsum_vs_f32m2_f32m1(temp00, vs00, zero, 8);
            vfloat32m1_t temp01 = vfredsum_vs_f32m2_f32m1(temp01, vs01, zero, 8);
            vfloat32m1_t temp02 = vfredsum_vs_f32m2_f32m1(temp02, vs02, zero, 8);
            vfloat32m1_t temp03 = vfredsum_vs_f32m2_f32m1(temp03, vs03, zero, 8);
            vfloat32m1_t temp10 = vfredsum_vs_f32m2_f32m1(temp10, vs10, zero, 8);
            vfloat32m1_t temp11 = vfredsum_vs_f32m2_f32m1(temp11, vs11, zero, 8);
            vfloat32m1_t temp12 = vfredsum_vs_f32m2_f32m1(temp12, vs12, zero, 8);
            vfloat32m1_t temp13 = vfredsum_vs_f32m2_f32m1(temp13, vs13, zero, 8);
            vfloat32m1_t temp20 = vfredsum_vs_f32m2_f32m1(temp20, vs20, zero, 8);
            vfloat32m1_t temp21 = vfredsum_vs_f32m2_f32m1(temp21, vs21, zero, 8);
            vfloat32m1_t temp22 = vfredsum_vs_f32m2_f32m1(temp22, vs22, zero, 8);
            vfloat32m1_t temp23 = vfredsum_vs_f32m2_f32m1(temp23, vs23, zero, 8);
            float32_t sum0[4], sum1[4], sum2[4];
            sum0[0] = vfmv_f_s_f32m1_f32(temp00);
            sum0[1] = vfmv_f_s_f32m1_f32(temp01);
            sum0[2] = vfmv_f_s_f32m1_f32(temp02);
            sum0[3] = vfmv_f_s_f32m1_f32(temp03);
            sum1[0] = vfmv_f_s_f32m1_f32(temp10);
            sum1[1] = vfmv_f_s_f32m1_f32(temp11);
            sum1[2] = vfmv_f_s_f32m1_f32(temp12);
            sum1[3] = vfmv_f_s_f32m1_f32(temp13);
            sum2[0] = vfmv_f_s_f32m1_f32(temp20);
            sum2[1] = vfmv_f_s_f32m1_f32(temp21);
            sum2[2] = vfmv_f_s_f32m1_f32(temp22);
            sum2[3] = vfmv_f_s_f32m1_f32(temp23);

            s0 = vfadd_vv_f32m1(vle32_v_f32m1(sum0, vl), s0, vl);
            s1 = vfadd_vv_f32m1(vle32_v_f32m1(sum1, vl), s1, vl);
            s2 = vfadd_vv_f32m1(vle32_v_f32m1(sum2, vl), s2, vl);


            if( relu )
            {
                vbool32_t m0 = vmfgt_vf_f32m1_b32(s0, 0, vl);
                vbool32_t m1 = vmfgt_vf_f32m1_b32(s1, 0, vl);
                vbool32_t m2 = vmfgt_vf_f32m1_b32(s2, 0, vl);
                s0 = vmerge_vvm_f32m1(m0, vfmul_vv_f32m1(s0, vr0, vl), s0, vl);
                s1 = vmerge_vvm_f32m1(m1, vfmul_vv_f32m1(s1, vr1, vl), s1, vl);
                s2 = vmerge_vvm_f32m1(m2, vfmul_vv_f32m1(s2, vr2, vl), s2, vl);
            }

            if( tail )
            {
                s0 = vmerge_vvm_f32m1(mask, vle32_v_f32m1(outptr0 + j, vl), s0, vl);
                s1 = vmerge_vvm_f32m1(mask, vle32_v_f32m1(outptr1 + j, vl), s1, vl);
                s2 = vmerge_vvm_f32m1(mask, vle32_v_f32m1(outptr2 + j, vl), s2, vl);
            }

            vse32_v_f32m1(outptr0 + j, s0, vl);
            vse32_v_f32m1(outptr1 + j, s1, vl);
            vse32_v_f32m1(outptr2 + j, s2, vl);
        }
    }
}

/*
Example for load_deinterleave:
    input: ptr[16] = {1,2,3, ... ,14,15,16}
    output: a = {1, 3, 5, 7, 9, 11, 13, 15}
    output: b = {2, 4, 6, 8,10, 12, 14, 16}
*/
static inline void vfloat32m2_load_deinterleave(const float* ptr, vfloat32m2_t& a, vfloat32m2_t& b)
{
    int vl = 8;
    uint32_t masks[] = {1,1,1,1,0,0,0,0};
    vuint32m2_t vm = vle32_v_u32m2(masks,vl);
    vbool16_t mask01 = vmseq_vx_u32m2_b16 (vm, 0, vl);
    vbool16_t mask10 = vmseq_vx_u32m2_b16 (vm, 1, vl);
    vfloat32m2_t ta = vle32_v_f32m2(ptr, vl), tb = vle32_v_f32m2(ptr+8, vl);
    uint idx[] = {0,2,4,6,1,3,5,7};
    uint idxa[] = {0,0,0,0,0,1,2,3}, idxb[] = {4,5,6,7,0,0,0,0};
    vuint32m2_t vidxa = vle32_v_u32m2(idxa, 8), vidxb = vle32_v_u32m2(idxb, 8);
    vuint32m2_t vidx = vle32_v_u32m2(idx, 8);
    vfloat32m2_t high = vfmv_v_f_f32m2(0, 8), low = vfmv_v_f_f32m2(0, 8);
    high = vrgather_vv_f32m2(ta, vidx, 8);
    low = vrgather_vv_f32m2(tb, vidx, 8);
    a = vrgather_vv_f32m2_m(mask01, high, low, vidxa, 8);
    b = vrgather_vv_f32m2_m(mask10, low, high, vidxb, 8);
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
    int vl = 8;
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
            const int VECSZ = 8;
            vfloat32m2_t vw00 = vfmv_v_f_f32m2(w00, vl), vw01 = vfmv_v_f_f32m2(w01, vl), vw02 = vfmv_v_f_f32m2(w02, vl),
                      vw10 = vfmv_v_f_f32m2(w10, vl), vw11 = vfmv_v_f_f32m2(w11, vl), vw12 = vfmv_v_f_f32m2(w12, vl),
                      vw20 = vfmv_v_f_f32m2(w20, vl), vw21 = vfmv_v_f_f32m2(w21, vl), vw22 = vfmv_v_f_f32m2(w22, vl);
            vfloat32m2_t vbias = vfmv_v_f_f32m2(bias, vl), vrc = vfmv_v_f_f32m2(relu_coeff, vl);

            if( stride_w == 1 )
                for( ; out_j < outW1; out_j += VECSZ )
                {
                    if (out_j + VECSZ > outW1 && out_j > pad_l)
                        out_j = outW1 - VECSZ;
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

                    vfloat32m2_t vout0 = vfmacc_vv_f32m2(vbias, v00, vw00, vl);
                    vfloat32m2_t vout1 = vfmul_vv_f32m2(v01, vw01, vl);
                    vfloat32m2_t vout2 = vfmul_vv_f32m2(v02, vw02, vl);

                    vout0 = vfmacc_vv_f32m2(vout0, v10, vw10, vl);
                    vout1 = vfmacc_vv_f32m2(vout1, v11, vw11, vl);
                    vout2 = vfmacc_vv_f32m2(vout2, v12, vw12, vl);

                    vout0 = vfmacc_vv_f32m2(vout0, v20, vw20, vl);
                    vout1 = vfmacc_vv_f32m2(vout1, v21, vw21, vl);
                    vout2 = vfmacc_vv_f32m2(vout2, v22, vw22, vl);

                    vout0 = vfadd_vv_f32m2(vfadd_vv_f32m2(vout0, vout1, vl), vout2, vl);
                    if (relu)
                    {
                        vbool16_t m = vmfgt_vf_f32m2_b16(vout0, 0, vl);
                        vout0 = vmerge_vvm_f32m2(m, vfmul_vv_f32m2(vout0, vrc, vl), vout0, vl);
                    }
                    vse32_v_f32m2(outptr + out_j, vout0, vl);
                }
            else
                for( ; out_j < outW1; out_j += VECSZ )
                {
                    if (out_j + VECSZ > outW1 && out_j > pad_l)
                        out_j = outW1 - VECSZ;
                    int in_j = out_j * stride_w - pad_l;
                    vfloat32m2_t v00, v01, v02, v10, v11, v12, v20, v21, v22, unused;
                    vfloat32m2_load_deinterleave(imgptr0 + in_j, v00, v01);
                    vfloat32m2_load_deinterleave(imgptr0 + in_j + 2, v02, unused);
                    vfloat32m2_load_deinterleave(imgptr1 + in_j, v10, v11);
                    vfloat32m2_load_deinterleave(imgptr1 + in_j + 2, v12, unused);
                    vfloat32m2_load_deinterleave(imgptr2 + in_j, v20, v21);
                    vfloat32m2_load_deinterleave(imgptr2 + in_j + 2, v22, unused);

                    vfloat32m2_t vout0 = vfmacc_vv_f32m2(vbias, v00, vw00, vl);
                    vfloat32m2_t vout1 = vfmul_vv_f32m2(v01, vw01, vl);
                    vfloat32m2_t vout2 = vfmul_vv_f32m2(v02, vw02, vl);

                    vout0 = vfmacc_vv_f32m2(vout0, v10, vw10, vl);
                    vout1 = vfmacc_vv_f32m2(vout1, v11, vw11, vl);
                    vout2 = vfmacc_vv_f32m2(vout2, v12, vw12, vl);

                    vout0 = vfmacc_vv_f32m2(vout0, v20, vw20, vl);
                    vout1 = vfmacc_vv_f32m2(vout1, v21, vw21, vl);
                    vout2 = vfmacc_vv_f32m2(vout2, v22, vw22, vl);

                    vout0 = vfadd_vv_f32m2(vfadd_vv_f32m2(vout0, vout1, vl), vout2, vl);
                    if (relu)
                    {
                        vbool16_t m = vmfgt_vf_f32m2_b16(vout0, 0, vl);
                        vout0 = vmerge_vvm_f32m2(m, vfmul_vv_f32m2(vout0, vrc, vl), vout0, vl);
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
