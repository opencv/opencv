// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// AVX-VNNI kernels for the int8 convolution layer. VPDPBUSD multiplies unsigned
// activations by signed weights, so these variants take the uint8 input directly
// and skip the u8 -> s8 rebase the plain AVX2 kernels need.
void fastConvVNNI( const int8_t* weights, size_t wstep, const int* bias,
                   const uint8_t* rowbuf, int* output, const int* outShape,
                   int blockSize, int vecsize, int vecsize_aligned, int outZp,
                   const float* multiplier, bool initOutput, bool finalOutput );
void fastGEMM1TVNNI( const uint8_t* vec, const int8_t* weights,
                     size_t wstep, const int* bias, const float* multiplier,
                     int* dst, int nvecs, int vecsize, int outZp );

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_AVX_VNNI

enum { FASCONV_BASE_VECSZ = 4 };

void fastConvVNNI( const int8_t* weights, size_t wstep, const int* bias,
                   const uint8_t* rowbuf, int* output, const int* outShape,
                   int blockSize, int vecsize, int vecsize_aligned, int outZp,
                   const float* multiplier, bool initOutput, bool finalOutput )
{
    int outCn = outShape[1];
    size_t outPlaneSize = outShape[2]*outShape[3];
    int CV_DECL_ALIGNED(16) maskbuf[FASCONV_BASE_VECSZ] = {0};
    int rsz = blockSize % FASCONV_BASE_VECSZ;
    for( int i = 0; i < rsz; i++ )
        maskbuf[FASCONV_BASE_VECSZ - i - 1] = -1;
    __m128 mask = _mm_loadu_ps((const float*)maskbuf);

    for( int i = 0; i < outCn; i += 3 )
    {
        const int8_t* wptr0 = weights + i*wstep;
        const int8_t* wptr1 = wptr0 + wstep;
        const int8_t* wptr2 = wptr1 + wstep;
        int* outptr0 = output + i*outPlaneSize;
        int* outptr1 = outptr0 + outPlaneSize;
        int* outptr2 = outptr1 + outPlaneSize;
        int bias0 = bias[i], bias1 = bias[i+1], bias2 = bias[i+2];
        float mult0 = multiplier[i], mult1 = multiplier[i+1], mult2 = multiplier[i+2];

        if( i+2 >= outCn )
        {
            wptr2 = wptr1;
            outptr2 = outptr1;
            bias2 = bias1;
            mult2 = mult1;

            if( i+1 >= outCn )
            {
                wptr2 = wptr1 = wptr0;
                outptr2 = outptr1 = outptr0;
                bias2 = bias1 = bias0;
                mult2 = mult1 = mult0;
            }
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
            const uint8_t* rptr = rowbuf + j*vecsize_aligned;

            __m256i vs00 = _mm256_setzero_si256(), vs01 = _mm256_setzero_si256(),
                    vs02 = _mm256_setzero_si256(), vs03 = _mm256_setzero_si256(),
                    vs10 = _mm256_setzero_si256(), vs11 = _mm256_setzero_si256(),
                    vs12 = _mm256_setzero_si256(), vs13 = _mm256_setzero_si256(),
                    vs20 = _mm256_setzero_si256(), vs21 = _mm256_setzero_si256(),
                    vs22 = _mm256_setzero_si256(), vs23 = _mm256_setzero_si256();

            for (; k < vecsize; k += 32, rptr += 32 )
            {
                __m256i w0 = _mm256_load_si256((const __m256i*)(wptr0 + k));
                __m256i w1 = _mm256_load_si256((const __m256i*)(wptr1 + k));
                __m256i w2 = _mm256_load_si256((const __m256i*)(wptr2 + k));
                __m256i r0 = _mm256_load_si256((const __m256i*)rptr);

                vs00 = _mm256_dpbusd_epi32(vs00, r0, w0);
                vs10 = _mm256_dpbusd_epi32(vs10, r0, w1);
                vs20 = _mm256_dpbusd_epi32(vs20, r0, w2);

                r0 = _mm256_load_si256((const __m256i*)(rptr + vecsize_aligned));
                vs01 = _mm256_dpbusd_epi32(vs01, r0, w0);
                vs11 = _mm256_dpbusd_epi32(vs11, r0, w1);
                vs21 = _mm256_dpbusd_epi32(vs21, r0, w2);

                r0 = _mm256_load_si256((const __m256i*)(rptr + vecsize_aligned*2));
                vs02 = _mm256_dpbusd_epi32(vs02, r0, w0);
                vs12 = _mm256_dpbusd_epi32(vs12, r0, w1);
                vs22 = _mm256_dpbusd_epi32(vs22, r0, w2);

                r0 = _mm256_load_si256((const __m256i*)(rptr + vecsize_aligned*3));
                vs03 = _mm256_dpbusd_epi32(vs03, r0, w0);
                vs13 = _mm256_dpbusd_epi32(vs13, r0, w1);
                vs23 = _mm256_dpbusd_epi32(vs23, r0, w2);
            }

            __m256i t0 = _mm256_hadd_epi32(_mm256_hadd_epi32(vs00, vs01), _mm256_hadd_epi32(vs02, vs03));
            __m256i t1 = _mm256_hadd_epi32(_mm256_hadd_epi32(vs10, vs11), _mm256_hadd_epi32(vs12, vs13));
            __m256i t2 = _mm256_hadd_epi32(_mm256_hadd_epi32(vs20, vs21), _mm256_hadd_epi32(vs22, vs23));

            t0 = _mm256_add_epi32(t0, _mm256_permute2x128_si256(t0, t0, 1));
            t1 = _mm256_add_epi32(t1, _mm256_permute2x128_si256(t1, t1, 1));
            t2 = _mm256_add_epi32(t2, _mm256_permute2x128_si256(t2, t2, 1));

            __m128i s0, s1, s2;

            if( initOutput )
            {
                s0 = _mm_set1_epi32(bias0);
                s1 = _mm_set1_epi32(bias1);
                s2 = _mm_set1_epi32(bias2);
            }
            else
            {
                s0 = _mm_loadu_si128((__m128i*)(outptr0 + j));
                s1 = _mm_loadu_si128((__m128i*)(outptr1 + j));
                s2 = _mm_loadu_si128((__m128i*)(outptr2 + j));
            }

            s0 = _mm_add_epi32(s0, _mm256_castsi256_si128(t0));
            s1 = _mm_add_epi32(s1, _mm256_castsi256_si128(t1));
            s2 = _mm_add_epi32(s2, _mm256_castsi256_si128(t2));

            if( finalOutput )
            {
                __m128i voutzp = _mm_set1_epi32(outZp);
                __m128i outmin = _mm_set1_epi32(-128), outmax = _mm_set1_epi32(127);
                s0 = _mm_add_epi32(voutzp, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(s0), _mm_set1_ps(mult0))));
                s1 = _mm_add_epi32(voutzp, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(s1), _mm_set1_ps(mult1))));
                s2 = _mm_add_epi32(voutzp, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(s2), _mm_set1_ps(mult2))));

                s0 = _mm_min_epi32(_mm_max_epi32(s0, outmin), outmax);
                s1 = _mm_min_epi32(_mm_max_epi32(s1, outmin), outmax);
                s2 = _mm_min_epi32(_mm_max_epi32(s2, outmin), outmax);
            }
            if( tail )
            {
                s0 = _mm_castps_si128(_mm_blendv_ps(_mm_loadu_ps((const float*)outptr0 + j), _mm_castsi128_ps(s0), mask));
                s1 = _mm_castps_si128(_mm_blendv_ps(_mm_loadu_ps((const float*)outptr1 + j), _mm_castsi128_ps(s1), mask));
                s2 = _mm_castps_si128(_mm_blendv_ps(_mm_loadu_ps((const float*)outptr2 + j), _mm_castsi128_ps(s2), mask));
            }
            _mm_storeu_si128((__m128i*)(outptr0 + j), s0);
            _mm_storeu_si128((__m128i*)(outptr1 + j), s1);
            _mm_storeu_si128((__m128i*)(outptr2 + j), s2);
        }

        for( ; j <= blockSize - 2; j += 2 )
        {
            const uint8_t* rptr0 = rowbuf + j*vecsize_aligned;
            const uint8_t* rptr1 = rowbuf + (j+1)*vecsize_aligned;
            int s00, s01, s10, s11, s20, s21;

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
                int8_t w0 = wptr0[k], w1 = wptr1[k], w2 = wptr2[k];
                int r = (int)rptr0[k];
                s00 += (int)w0*r; s10 += (int)w1*r; s20 += (int)w2*r;
                r = (int)rptr1[k];
                s01 += (int)w0*r; s11 += (int)w1*r; s21 += (int)w2*r;
            }

            if( finalOutput )
            {
                s00 = std::min(std::max(outZp + (int)std::round(s00*mult0), -128), 127);
                s01 = std::min(std::max(outZp + (int)std::round(s01*mult0), -128), 127);
                s10 = std::min(std::max(outZp + (int)std::round(s10*mult1), -128), 127);
                s11 = std::min(std::max(outZp + (int)std::round(s11*mult1), -128), 127);
                s20 = std::min(std::max(outZp + (int)std::round(s20*mult2), -128), 127);
                s21 = std::min(std::max(outZp + (int)std::round(s21*mult2), -128), 127);
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
            const uint8_t* rptr0 = rowbuf + j*vecsize_aligned;
            int s00, s10, s20;

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
                int8_t w0 = wptr0[k], w1 = wptr1[k], w2 = wptr2[k];
                int r = (int)rptr0[k];
                s00 += (int)w0*r; s10 += (int)w1*r; s20 += (int)w2*r;
            }

            if( finalOutput )
            {
                s00 = std::min(std::max(outZp + (int)std::round(s00*mult0), -128), 127);
                s10 = std::min(std::max(outZp + (int)std::round(s10*mult1), -128), 127);
                s20 = std::min(std::max(outZp + (int)std::round(s20*mult2), -128), 127);
            }
            outptr0[j] = s00;
            outptr1[j] = s10;
            outptr2[j] = s20;
        }
    }
    _mm256_zeroupper();
}

void fastGEMM1TVNNI( const uint8_t* vec, const int8_t* weights,
                     size_t wstep, const int* bias, const float* multiplier,
                     int* dst, int nvecs, int vecsize, int outZp )
{
    int i = 0;

    for( ; i <= nvecs - 8; i += 8 )
    {
        const int8_t* wptr = weights + i*wstep;
        __m256i vs0 = _mm256_setzero_si256(), vs1 = _mm256_setzero_si256(),
                vs2 = _mm256_setzero_si256(), vs3 = _mm256_setzero_si256(),
                vs4 = _mm256_setzero_si256(), vs5 = _mm256_setzero_si256(),
                vs6 = _mm256_setzero_si256(), vs7 = _mm256_setzero_si256();

        __m128i voutzp = _mm_set1_epi32(outZp);
        __m128i outmin = _mm_set1_epi32(-128), outmax = _mm_set1_epi32(127);

        for( int k = 0; k < vecsize; k += 32, wptr += 32 )
        {
            __m256i v = _mm256_load_si256((const __m256i*)(vec + k));

            vs0 = _mm256_dpbusd_epi32(vs0, v, _mm256_load_si256((const __m256i*)wptr));
            vs1 = _mm256_dpbusd_epi32(vs1, v, _mm256_load_si256((const __m256i*)(wptr + wstep)));
            vs2 = _mm256_dpbusd_epi32(vs2, v, _mm256_load_si256((const __m256i*)(wptr + wstep*2)));
            vs3 = _mm256_dpbusd_epi32(vs3, v, _mm256_load_si256((const __m256i*)(wptr + wstep*3)));
            vs4 = _mm256_dpbusd_epi32(vs4, v, _mm256_load_si256((const __m256i*)(wptr + wstep*4)));
            vs5 = _mm256_dpbusd_epi32(vs5, v, _mm256_load_si256((const __m256i*)(wptr + wstep*5)));
            vs6 = _mm256_dpbusd_epi32(vs6, v, _mm256_load_si256((const __m256i*)(wptr + wstep*6)));
            vs7 = _mm256_dpbusd_epi32(vs7, v, _mm256_load_si256((const __m256i*)(wptr + wstep*7)));
        }

        __m256i s0 = _mm256_hadd_epi32(_mm256_hadd_epi32(vs0, vs1), _mm256_hadd_epi32(vs2, vs3));
        __m256i s1 = _mm256_hadd_epi32(_mm256_hadd_epi32(vs4, vs5), _mm256_hadd_epi32(vs6, vs7));

        s0 = _mm256_add_epi32(s0, _mm256_permute2x128_si256(s0, s0, 1));
        s1 = _mm256_add_epi32(s1, _mm256_permute2x128_si256(s1, s1, 1));

        __m128i t0 = _mm_add_epi32(_mm256_castsi256_si128(s0), _mm_loadu_si128((__m128i*)(bias + i)));
        __m128i t1 = _mm_add_epi32(_mm256_castsi256_si128(s1), _mm_loadu_si128((__m128i*)(bias + i + 4)));

        t0 = _mm_add_epi32(voutzp, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(t0), _mm_loadu_ps(multiplier + i))));
        t1 = _mm_add_epi32(voutzp, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(t1), _mm_loadu_ps(multiplier + i + 4))));

        t0 = _mm_min_epi32(_mm_max_epi32(t0, outmin), outmax);
        t1 = _mm_min_epi32(_mm_max_epi32(t1, outmin), outmax);

        _mm_storeu_si128((__m128i*)(dst + i), t0);
        _mm_storeu_si128((__m128i*)(dst + i + 4), t1);
    }

    for( ; i < nvecs; i++ )
    {
        const int8_t* wptr = weights + i*wstep;
        __m256i vs0 = _mm256_setzero_si256();

        for( int k = 0; k < vecsize; k += 32, wptr += 32 )
        {
            __m256i v = _mm256_load_si256((const __m256i*)(vec + k));
            vs0 = _mm256_dpbusd_epi32(vs0, v, _mm256_load_si256((const __m256i*)wptr));
        }

        __m256i s0 = _mm256_hadd_epi32(_mm256_hadd_epi32(vs0, vs0), vs0);
        s0 = _mm256_add_epi32(s0, _mm256_permute2x128_si256(s0, s0, 1));
        int temp = _mm_extract_epi32(_mm256_castsi256_si128(s0), 0);
        dst[i] = outZp + (int)std::round((temp + bias[i]) * multiplier[i]);
    }

    _mm256_zeroupper();
}

#endif // !CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY && CV_AVX_VNNI

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace cv::dnn
