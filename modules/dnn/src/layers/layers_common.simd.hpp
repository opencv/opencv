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
        vl = vsetvl_e32m4(avl);
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
                float a0 = aptr0[k];
                float a1 = aptr1[k];
                float a2 = aptr2[k];
                float a3 = aptr3[k];
                float a4 = aptr4[k];
                float a5 = aptr5[k];
                float a6 = aptr6[k];

                vfloat32m4_t b = vle32_v_f32m4(bptr + k*bstep + n, vl);
                d0 = vfmacc_vf_f32m4(d0, a0, b, vl);
                d1 = vfmacc_vf_f32m4(d1, a1, b, vl);
                d2 = vfmacc_vf_f32m4(d2, a2, b, vl);
                d3 = vfmacc_vf_f32m4(d3, a3, b, vl);
                d4 = vfmacc_vf_f32m4(d4, a4, b, vl);
                d5 = vfmacc_vf_f32m4(d5, a5, b, vl);
                d6 = vfmacc_vf_f32m4(d6, a6, b, vl);
            }
            vse32_v_f32m4(cptr0 + n, d0, vl);
            vse32_v_f32m4(cptr1 + n, d1, vl);
            vse32_v_f32m4(cptr2 + n, d2, vl);
            vse32_v_f32m4(cptr3 + n, d3, vl);
            vse32_v_f32m4(cptr4 + n, d4, vl);
            vse32_v_f32m4(cptr5 + n, d5, vl);
            vse32_v_f32m4(cptr6 + n, d6, vl);
        }
    }
}

void fastGEMM1T( const float* vec, const float* weights,
                 size_t wstep, const float* bias,
                 float* dst, int nvecs, int vecsize )
{
    const int vlm2 = vsetvlmax_e32m2();
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
        int avl = vecsize, vl;
        for(int k = 0 ; k < vecsize; k += vl, wptr += vl, avl -= vl)
        {
            vl = vsetvl_e32m2(avl);
            vfloat32m2_t v = vle32_v_f32m2(vec + k, vl);
            vs0 = vfmacc_vv_f32m2(vs0, vle32_v_f32m2(wptr, vl), v, vl);
            vs1 = vfmacc_vv_f32m2(vs1, vle32_v_f32m2(wptr + wstep, vl), v, vl);
            vs2 = vfmacc_vv_f32m2(vs2, vle32_v_f32m2(wptr + wstep*2, vl), v, vl);
            vs3 = vfmacc_vv_f32m2(vs3, vle32_v_f32m2(wptr + wstep*3, vl), v, vl);
            vs4 = vfmacc_vv_f32m2(vs4, vle32_v_f32m2(wptr + wstep*4, vl), v, vl);
            vs5 = vfmacc_vv_f32m2(vs5, vle32_v_f32m2(wptr + wstep*5, vl), v, vl);
            vs6 = vfmacc_vv_f32m2(vs6, vle32_v_f32m2(wptr + wstep*6, vl), v, vl);
            vs7 = vfmacc_vv_f32m2(vs7, vle32_v_f32m2(wptr + wstep*7, vl), v, vl);
            vs8 = vfmacc_vv_f32m2(vs8, vle32_v_f32m2(wptr + wstep*8, vl), v, vl);
            vs9 = vfmacc_vv_f32m2(vs9, vle32_v_f32m2(wptr + wstep*9, vl), v, vl);
            vs10 = vfmacc_vv_f32m2(vs10, vle32_v_f32m2(wptr + wstep*10, vl), v, vl);
            vs11 = vfmacc_vv_f32m2(vs11, vle32_v_f32m2(wptr + wstep*11, vl), v, vl);
            vs12 = vfmacc_vv_f32m2(vs12, vle32_v_f32m2(wptr + wstep*12, vl), v, vl);
            vs13 = vfmacc_vv_f32m2(vs13, vle32_v_f32m2(wptr + wstep*13, vl), v, vl);
            vs14 = vfmacc_vv_f32m2(vs14, vle32_v_f32m2(wptr + wstep*14, vl), v, vl);
        }

        // Calculate the sum of each vector
        float sum[15];
        vfloat32m1_t zero = vfmv_v_f_f32m1(0, vlm2);
        sum[0] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs0, zero, vlm2));
        sum[1] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs1, zero, vlm2));
        sum[2] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs2, zero, vlm2));
        sum[3] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs3, zero, vlm2));
        sum[4] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs4, zero, vlm2));
        sum[5] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs5, zero, vlm2));
        sum[6] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs6, zero, vlm2));
        sum[7] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs7, zero, vlm2));
        sum[8] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs8, zero, vlm2));
        sum[9] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs9, zero, vlm2));
        sum[10] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs10, zero, vlm2));
        sum[11] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs11, zero, vlm2));
        sum[12] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs12, zero, vlm2));
        sum[13] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs13, zero, vlm2));
        sum[14] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs14, zero, vlm2));

        vfloat32m4_t s0 = vfadd_vv_f32m4(vle32_v_f32m4(sum, 15), vle32_v_f32m4(bias + i, 15), 15);
        vse32_v_f32m4(dst + i, s0, 15);
    }
    int unroll_tail = nvecs - i;
    if (unroll_tail > 0)
    {
        const float* wptr = weights + i*wstep;
        vfloat32m2_t
               vs0 = vfmv_v_f_f32m2(0, vlm2), vs1 = vfmv_v_f_f32m2(0, vlm2), vs2 = vfmv_v_f_f32m2(0, vlm2),
               vs3 = vfmv_v_f_f32m2(0, vlm2), vs4 = vfmv_v_f_f32m2(0, vlm2), vs5 = vfmv_v_f_f32m2(0, vlm2),
               vs6 = vfmv_v_f_f32m2(0, vlm2), vs7 = vfmv_v_f_f32m2(0, vlm2), vs8 = vfmv_v_f_f32m2(0, vlm2),
               vs9 = vfmv_v_f_f32m2(0, vlm2), vs10 = vfmv_v_f_f32m2(0, vlm2), vs11 = vfmv_v_f_f32m2(0, vlm2),
               vs12 = vfmv_v_f_f32m2(0, vlm2), vs13 = vfmv_v_f_f32m2(0, vlm2);
        int avl = vecsize, vl;
        for(int k = 0; k < vecsize; k += vl, wptr += vl, avl -= vl)
        {
            vl = vsetvl_e32m2(avl);
            vfloat32m2_t v = vle32_v_f32m2(vec + k, vl);
            vs0 = vfmacc_vv_f32m2(vs0, vle32_v_f32m2(wptr, vl), v, vl);
            vs1 = vfmacc_vv_f32m2(vs1, vle32_v_f32m2(wptr + wstep*std::min(1, unroll_tail-1), vl), v, vl);
            vs2 = vfmacc_vv_f32m2(vs2, vle32_v_f32m2(wptr + wstep*std::min(2, unroll_tail-1), vl), v, vl);
            vs3 = vfmacc_vv_f32m2(vs3, vle32_v_f32m2(wptr + wstep*std::min(3, unroll_tail-1), vl), v, vl);
            vs4 = vfmacc_vv_f32m2(vs4, vle32_v_f32m2(wptr + wstep*std::min(4, unroll_tail-1), vl), v, vl);
            vs5 = vfmacc_vv_f32m2(vs5, vle32_v_f32m2(wptr + wstep*std::min(5, unroll_tail-1), vl), v, vl);
            vs6 = vfmacc_vv_f32m2(vs6, vle32_v_f32m2(wptr + wstep*std::min(6, unroll_tail-1), vl), v, vl);
            vs7 = vfmacc_vv_f32m2(vs7, vle32_v_f32m2(wptr + wstep*std::min(7, unroll_tail-1), vl), v, vl);
            vs8 = vfmacc_vv_f32m2(vs8, vle32_v_f32m2(wptr + wstep*std::min(8, unroll_tail-1), vl), v, vl);
            vs9 = vfmacc_vv_f32m2(vs9, vle32_v_f32m2(wptr + wstep*std::min(9, unroll_tail-1), vl), v, vl);
            vs10 = vfmacc_vv_f32m2(vs10, vle32_v_f32m2(wptr + wstep*std::min(10, unroll_tail-1), vl), v, vl);
            vs11 = vfmacc_vv_f32m2(vs11, vle32_v_f32m2(wptr + wstep*std::min(11, unroll_tail-1), vl), v, vl);
            vs12 = vfmacc_vv_f32m2(vs12, vle32_v_f32m2(wptr + wstep*std::min(12, unroll_tail-1), vl), v, vl);
            vs13 = vfmacc_vv_f32m2(vs13, vle32_v_f32m2(wptr + wstep*std::min(13, unroll_tail-1), vl), v, vl);
        }

        // Calculate the sum of each vector
        float sum[14];
        vfloat32m1_t zero = vfmv_v_f_f32m1(0, vlm2);
        sum[0] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs0, zero, vlm2));
        sum[1] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs1, zero, vlm2));
        sum[2] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs2, zero, vlm2));
        sum[3] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs3, zero, vlm2));
        sum[4] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs4, zero, vlm2));
        sum[5] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs5, zero, vlm2));
        sum[6] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs6, zero, vlm2));
        sum[7] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs7, zero, vlm2));
        sum[8] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs8, zero, vlm2));
        sum[9] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs9, zero, vlm2));
        sum[10] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs10, zero, vlm2));
        sum[11] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs11, zero, vlm2));
        sum[12] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs12, zero, vlm2));
        sum[13] = vfmv_f_s_f32m1_f32(vfredosum_vs_f32m2_f32m1(zero, vs13, zero, vlm2));

        vfloat32m4_t s0 = vfadd_vv_f32m4(vle32_v_f32m4(sum, unroll_tail), vle32_v_f32m4(bias + i, unroll_tail), unroll_tail);
        vse32_v_f32m4(dst + i, s0, unroll_tail);
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
    cv::AutoBuffer<float> cvBuffer(sizeof(float)*vl*2);
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
    int vl;
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
            int avl = outW1 - out_j;
            if( stride_w == 1 )
                for( ; out_j < outW1; out_j += vl, avl -= vl)
                {
                    vl = vsetvl_e32m2(avl);
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
                for( ; out_j < outW1; out_j += vl, avl -= vl)
                {
                    vl = vsetvl_e32m2(avl);
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

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_LASX

static inline void _v256_load_deinterleave(const float* ptr, __m256& a, __m256& b)
{
    __m256 t0 = (__m256)__lasx_xvld(ptr, 0);
    __m256 t1 = (__m256)__lasx_xvld(ptr, 8*4);

    __m256 lo = (__m256)__lasx_xvpermi_q(t0, t1, 2+0*16);
    __m256 hi = (__m256)__lasx_xvpermi_q(t0, t1, 3+1*16);

    a = (__m256)__lasx_xvpermi_w(hi, lo, 0x88);
    b = (__m256)__lasx_xvpermi_w(hi, lo, 0xdd);
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
            __m256 vw00 = _v256_setall_ps(w00), vw01 = _v256_setall_ps(w01), vw02 = _v256_setall_ps(w02),
                   vw10 = _v256_setall_ps(w10), vw11 = _v256_setall_ps(w11), vw12 = _v256_setall_ps(w12),
                   vw20 = _v256_setall_ps(w20), vw21 = _v256_setall_ps(w21), vw22 = _v256_setall_ps(w22);
            __m256 z = (__m256)__lasx_xvxor_v((__m256i)vw00, (__m256i)vw00),
            vbias = _v256_setall_ps(bias), vrc = _v256_setall_ps(relu_coeff);

            if( stride_w == 1 )
                for( ; out_j < outW1; out_j += VECSZ )
                {
                    if (out_j + VECSZ > outW1 && out_j > pad_l)
                        out_j = outW1 - VECSZ;
                    int in_j = out_j * stride_w - pad_l;
                    __m256 v00 = (__m256)__lasx_xvld(imgptr0 + in_j, 0),
                           v01 = (__m256)__lasx_xvld(imgptr0 + in_j + dilation_w, 0),
                           v02 = (__m256)__lasx_xvld(imgptr0 + in_j + dilation_w*2, 0),
                           v10 = (__m256)__lasx_xvld(imgptr1 + in_j, 0),
                           v11 = (__m256)__lasx_xvld(imgptr1 + in_j + dilation_w, 0),
                           v12 = (__m256)__lasx_xvld(imgptr1 + in_j + dilation_w*2, 0),
                           v20 = (__m256)__lasx_xvld(imgptr2 + in_j, 0),
                           v21 = (__m256)__lasx_xvld(imgptr2 + in_j + dilation_w, 0),
                           v22 = (__m256)__lasx_xvld(imgptr2 + in_j + dilation_w*2, 0);

                    __m256 vout0 = __lasx_xvfmadd_s(v00, vw00, vbias);
                    __m256 vout1 = __lasx_xvfmul_s(v01, vw01);
                    __m256 vout2 = __lasx_xvfmul_s(v02, vw02);

                    vout0 = __lasx_xvfmadd_s(v10, vw10, vout0);
                    vout1 = __lasx_xvfmadd_s(v11, vw11, vout1);
                    vout2 = __lasx_xvfmadd_s(v12, vw12, vout2);

                    vout0 = __lasx_xvfmadd_s(v20, vw20, vout0);
                    vout1 = __lasx_xvfmadd_s(v21, vw21, vout1);
                    vout2 = __lasx_xvfmadd_s(v22, vw22, vout2);

                    vout0 = __lasx_xvfadd_s(__lasx_xvfadd_s(vout0, vout1), vout2);
                    if (relu)
                    {
                        __m256i m = __lasx_xvfcmp_clt_s(z, vout0);
                        vout0 = (__m256)__lasx_xvbitsel_v((__m256i)__lasx_xvfmul_s(vout0, vrc), (__m256i)vout0, m);
                    }
                    __lasx_xvst(vout0, outptr + out_j, 0);
                }
            else
                for( ; out_j < outW1; out_j += VECSZ )
                {
                    if (out_j + VECSZ > outW1 && out_j > pad_l)
                        out_j = outW1 - VECSZ;
                    int in_j = out_j * stride_w - pad_l;
                    __m256 v00, v01, v02, v10, v11, v12, v20, v21, v22, unused;
                    _v256_load_deinterleave(imgptr0 + in_j, v00, v01);
                    _v256_load_deinterleave(imgptr0 + in_j + 2, v02, unused);
                    _v256_load_deinterleave(imgptr1 + in_j, v10, v11);
                    _v256_load_deinterleave(imgptr1 + in_j + 2, v12, unused);
                    _v256_load_deinterleave(imgptr2 + in_j, v20, v21);
                    _v256_load_deinterleave(imgptr2 + in_j + 2, v22, unused);

                    __m256 vout0 = __lasx_xvfmadd_s(v00, vw00, vbias);
                    __m256 vout1 = __lasx_xvfmul_s(v01, vw01);
                    __m256 vout2 = __lasx_xvfmul_s(v02, vw02);

                    vout0 = __lasx_xvfmadd_s(v10, vw10, vout0);
                    vout1 = __lasx_xvfmadd_s(v11, vw11, vout1);
                    vout2 = __lasx_xvfmadd_s(v12, vw12, vout2);

                    vout0 = __lasx_xvfmadd_s(v20, vw20, vout0);
                    vout1 = __lasx_xvfmadd_s(v21, vw21, vout1);
                    vout2 = __lasx_xvfmadd_s(v22, vw22, vout2);

                    vout0 = __lasx_xvfadd_s(__lasx_xvfadd_s(vout0, vout1), vout2);
                    if (relu)
                    {
                        __m256i m = __lasx_xvfcmp_clt_s(z, vout0);
                        vout0 = (__m256)__lasx_xvbitsel_v((__m256i)__lasx_xvfmul_s(vout0, vrc), (__m256i)vout0, m);
                    }
                    __lasx_xvst(vout0, outptr + out_j, 0);
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
