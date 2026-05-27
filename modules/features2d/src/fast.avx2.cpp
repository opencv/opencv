/* This is FAST corner detector, contributed to OpenCV by the author, Edward Rosten.
   Below is the original copyright and the references */
/*
Copyright (c) 2006, 2008 Edward Rosten
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
    *Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
    *Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
    *Neither the name of the University of Cambridge nor the names of
     its contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/*
The references are:
 * Machine learning for high-speed corner detection,
   E. Rosten and T. Drummond, ECCV 2006
 * Faster and better: A machine learning approach to corner detection
   E. Rosten, R. Porter and T. Drummond, PAMI, 2009
*/
#include "precomp.hpp"
#include "fast.hpp"
#include "opencv2/core/hal/intrin.hpp"
namespace cv
{
namespace opt_AVX2
{
/* Load 16 uchar values and zero-extend to int16 */
static inline __m256i loadu_i16(const uchar* p)
{
    return _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)p));
}
/* Extract 16-bit movemask from __m256i (AVX2 has no native 16-bit movemask) */
static inline uint32_t movemask_epi16(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extractf128_si256(v, 1);
    uint32_t mlo = (uint32_t)_mm_movemask_epi8(lo);
    uint32_t mhi = (uint32_t)_mm_movemask_epi8(hi);
    uint32_t r = 0;
    for (int i = 0; i < 8; i++)
    {
        r |= ((mlo >> (2*i)) & 1u) << i;
        r |= ((mhi >> (2*i)) & 1u) << (8+i);
    }
    return r;
}
class FAST_t_patternSize16_AVX2_Impl CV_FINAL: public FAST_t_patternSize16_AVX2
{
public:
    FAST_t_patternSize16_AVX2_Impl(int _cols, int _threshold, bool _nonmax_suppression, const int* _pixel)
        : cols(_cols), nonmax_suppression(_nonmax_suppression), pixel(_pixel)
    {
        t256c = (char)_threshold;
        threshold = std::min(std::max(_threshold, 0), 255);
    }

    virtual void process(int &j, const uchar* &ptr, uchar* curr, int* cornerpos, int &ncorners) CV_OVERRIDE
    {
        if (nonmax_suppression)
            processFusion(j, ptr, curr, cornerpos, ncorners);
        else
            processCounting(j, ptr, cornerpos, ncorners);
        _mm256_zeroupper();
    }

    /* NMS=true path: fusion score tree with prescreen (16-wide int16)
     *
     * Computes corner scores for all pixels in a 16-wide block using a
     * multi-level binary min/max tree (derived from the __E macro in the
     * scalar reference). This avoids the per-corner score loop of the
     * counting method, which is the bottleneck when NMS is enabled.
     */
    void processFusion(int &j, const uchar* &ptr, uchar* curr, int* cornerpos, int &ncorners)
    {
        const __m256i vthr = _mm256_set1_epi16((short)(threshold - 1));
        for (; j <= cols - 3 - 16; j += 16, ptr += 16)
        {
            __m256i vcen = loadu_i16(ptr);
            __m256i vlo = _mm256_sub_epi16(vcen, _mm256_set1_epi16((short)threshold));
            __m256i vhi = _mm256_add_epi16(vcen, _mm256_set1_epi16((short)threshold));

            /* Quick pre-filter: check 4 diametrically opposed pairs */
            __m256i vk0  = loadu_i16(ptr + pixel[0]);
            __m256i vk4  = loadu_i16(ptr + pixel[4]);
            __m256i vk8  = loadu_i16(ptr + pixel[8]);
            __m256i vk12 = loadu_i16(ptr + pixel[12]);

            __m256i bright = _mm256_and_si256(
                _mm256_cmpgt_epi16(vk0, vhi), _mm256_cmpgt_epi16(vk4, vhi));
            __m256i dark = _mm256_and_si256(
                _mm256_cmpgt_epi16(vlo, vk0), _mm256_cmpgt_epi16(vlo, vk4));
            bright = _mm256_or_si256(bright, _mm256_and_si256(
                _mm256_cmpgt_epi16(vk4, vhi), _mm256_cmpgt_epi16(vk8, vhi)));
            dark   = _mm256_or_si256(dark, _mm256_and_si256(
                _mm256_cmpgt_epi16(vlo, vk4), _mm256_cmpgt_epi16(vlo, vk8)));
            bright = _mm256_or_si256(bright, _mm256_and_si256(
                _mm256_cmpgt_epi16(vk8, vhi), _mm256_cmpgt_epi16(vk12, vhi)));
            dark   = _mm256_or_si256(dark, _mm256_and_si256(
                _mm256_cmpgt_epi16(vlo, vk8), _mm256_cmpgt_epi16(vlo, vk12)));
            bright = _mm256_or_si256(bright, _mm256_and_si256(
                _mm256_cmpgt_epi16(vk12, vhi), _mm256_cmpgt_epi16(vk0, vhi)));
            dark   = _mm256_or_si256(dark, _mm256_and_si256(
                _mm256_cmpgt_epi16(vlo, vk12), _mm256_cmpgt_epi16(vlo, vk0)));

            if (_mm256_testz_si256(_mm256_or_si256(bright, dark), _mm256_or_si256(bright, dark)))
                continue;

            __m256i d0  = _mm256_sub_epi16(vcen, vk0);
            __m256i d1  = _mm256_sub_epi16(vcen, loadu_i16(ptr + pixel[1]));
            __m256i d2  = _mm256_sub_epi16(vcen, loadu_i16(ptr + pixel[2]));
            __m256i d3  = _mm256_sub_epi16(vcen, loadu_i16(ptr + pixel[3]));
            __m256i d4  = _mm256_sub_epi16(vcen, vk4);
            __m256i d5  = _mm256_sub_epi16(vcen, loadu_i16(ptr + pixel[5]));
            __m256i d6  = _mm256_sub_epi16(vcen, loadu_i16(ptr + pixel[6]));
            __m256i d7  = _mm256_sub_epi16(vcen, loadu_i16(ptr + pixel[7]));
            __m256i d8  = _mm256_sub_epi16(vcen, vk8);
            __m256i d9  = _mm256_sub_epi16(vcen, loadu_i16(ptr + pixel[9]));
            __m256i d10 = _mm256_sub_epi16(vcen, loadu_i16(ptr + pixel[10]));
            __m256i d11 = _mm256_sub_epi16(vcen, loadu_i16(ptr + pixel[11]));
            __m256i d12 = _mm256_sub_epi16(vcen, vk12);
            __m256i d13 = _mm256_sub_epi16(vcen, loadu_i16(ptr + pixel[13]));
            __m256i d14 = _mm256_sub_epi16(vcen, loadu_i16(ptr + pixel[14]));
            __m256i d15 = _mm256_sub_epi16(vcen, loadu_i16(ptr + pixel[15]));

            /* Bright score: multi-level binary tree (min tree) */
            __m256i va0 = _mm256_min_epi16(d7, d8);
            __m256i va00 = _mm256_min_epi16(va0, d6);
            va00 = _mm256_min_epi16(va00, d5); va00 = _mm256_min_epi16(va00, d4); va00 = _mm256_min_epi16(va00, d3);
            __m256i va000 = _mm256_min_epi16(va00, d2); va000 = _mm256_min_epi16(va000, d1);
            __m256i va001 = _mm256_min_epi16(va00, d9); va001 = _mm256_min_epi16(va001, d10);
            __m256i va01 = _mm256_min_epi16(va0, d9); va01 = _mm256_min_epi16(va01, d10);
            va01 = _mm256_min_epi16(va01, d11); va01 = _mm256_min_epi16(va01, d12);
            __m256i va010 = _mm256_min_epi16(va01, d6); va010 = _mm256_min_epi16(va010, d5);
            __m256i va011 = _mm256_min_epi16(va01, d13); va011 = _mm256_min_epi16(va011, d14);

            __m256i min_max = _mm256_max_epi16(_mm256_min_epi16(va000, d0), _mm256_min_epi16(va000, d9));
            min_max = _mm256_max_epi16(min_max, _mm256_max_epi16(_mm256_min_epi16(va001, d2), _mm256_min_epi16(va001, d11)));
            min_max = _mm256_max_epi16(min_max, _mm256_max_epi16(_mm256_min_epi16(va010, d4), _mm256_min_epi16(va010, d13)));
            min_max = _mm256_max_epi16(min_max, _mm256_max_epi16(_mm256_min_epi16(va011, d6), _mm256_min_epi16(va011, d15)));

            __m256i va1 = _mm256_min_epi16(d15, d0);
            __m256i va10 = _mm256_min_epi16(va1, d14); va10 = _mm256_min_epi16(va10, d13);
            va10 = _mm256_min_epi16(va10, d12); va10 = _mm256_min_epi16(va10, d11);
            __m256i va100 = _mm256_min_epi16(va10, d10); va100 = _mm256_min_epi16(va100, d9);
            __m256i va101 = _mm256_min_epi16(va10, d1); va101 = _mm256_min_epi16(va101, d2);
            __m256i va11 = _mm256_min_epi16(va1, d1); va11 = _mm256_min_epi16(va11, d2);
            va11 = _mm256_min_epi16(va11, d3); va11 = _mm256_min_epi16(va11, d4);
            __m256i va110 = _mm256_min_epi16(va11, d14); va110 = _mm256_min_epi16(va110, d13);
            __m256i va111 = _mm256_min_epi16(va11, d5); va111 = _mm256_min_epi16(va111, d6);

            min_max = _mm256_max_epi16(min_max, _mm256_max_epi16(_mm256_min_epi16(va100, d8), _mm256_min_epi16(va100, d1)));
            min_max = _mm256_max_epi16(min_max, _mm256_max_epi16(_mm256_min_epi16(va101, d10), _mm256_min_epi16(va101, d3)));
            min_max = _mm256_max_epi16(min_max, _mm256_max_epi16(_mm256_min_epi16(va110, d12), _mm256_min_epi16(va110, d5)));
            min_max = _mm256_max_epi16(min_max, _mm256_max_epi16(_mm256_min_epi16(va111, d14), _mm256_min_epi16(va111, d7)));

            /* Dark score: binary tree (max tree) */
            __m256i vb0 = _mm256_max_epi16(d7, d8);
            __m256i vb00 = _mm256_max_epi16(vb0, d6); vb00 = _mm256_max_epi16(vb00, d5);
            vb00 = _mm256_max_epi16(vb00, d4); vb00 = _mm256_max_epi16(vb00, d3);
            __m256i vb000 = _mm256_max_epi16(vb00, d2); vb000 = _mm256_max_epi16(vb000, d1);
            __m256i vb001 = _mm256_max_epi16(vb00, d9); vb001 = _mm256_max_epi16(vb001, d10);
            __m256i vb01 = _mm256_max_epi16(vb0, d9); vb01 = _mm256_max_epi16(vb01, d10);
            vb01 = _mm256_max_epi16(vb01, d11); vb01 = _mm256_max_epi16(vb01, d12);
            __m256i vb010 = _mm256_max_epi16(vb01, d6); vb010 = _mm256_max_epi16(vb010, d5);
            __m256i vb011 = _mm256_max_epi16(vb01, d13); vb011 = _mm256_max_epi16(vb011, d14);

            __m256i max_min = _mm256_min_epi16(_mm256_max_epi16(vb000, d0), _mm256_max_epi16(vb000, d9));
            max_min = _mm256_min_epi16(max_min, _mm256_min_epi16(_mm256_max_epi16(vb001, d2), _mm256_max_epi16(vb001, d11)));
            max_min = _mm256_min_epi16(max_min, _mm256_min_epi16(_mm256_max_epi16(vb010, d4), _mm256_max_epi16(vb010, d13)));
            max_min = _mm256_min_epi16(max_min, _mm256_min_epi16(_mm256_max_epi16(vb011, d6), _mm256_max_epi16(vb011, d15)));

            __m256i vb1 = _mm256_max_epi16(d15, d0);
            __m256i vb10 = _mm256_max_epi16(vb1, d14); vb10 = _mm256_max_epi16(vb10, d13);
            vb10 = _mm256_max_epi16(vb10, d12); vb10 = _mm256_max_epi16(vb10, d11);
            __m256i vb100 = _mm256_max_epi16(vb10, d10); vb100 = _mm256_max_epi16(vb100, d9);
            __m256i vb101 = _mm256_max_epi16(vb10, d1); vb101 = _mm256_max_epi16(vb101, d2);
            __m256i vb11 = _mm256_max_epi16(vb1, d1); vb11 = _mm256_max_epi16(vb11, d2);
            vb11 = _mm256_max_epi16(vb11, d3); vb11 = _mm256_max_epi16(vb11, d4);
            __m256i vb110 = _mm256_max_epi16(vb11, d14); vb110 = _mm256_max_epi16(vb110, d13);
            __m256i vb111 = _mm256_max_epi16(vb11, d5); vb111 = _mm256_max_epi16(vb111, d6);

            max_min = _mm256_min_epi16(max_min, _mm256_min_epi16(_mm256_max_epi16(vb100, d8), _mm256_max_epi16(vb100, d1)));
            max_min = _mm256_min_epi16(max_min, _mm256_min_epi16(_mm256_max_epi16(vb101, d10), _mm256_max_epi16(vb101, d3)));
            max_min = _mm256_min_epi16(max_min, _mm256_min_epi16(_mm256_max_epi16(vb110, d12), _mm256_max_epi16(vb110, d5)));
            max_min = _mm256_min_epi16(max_min, _mm256_min_epi16(_mm256_max_epi16(vb111, d14), _mm256_max_epi16(vb111, d7)));

            __m256i score_v = _mm256_sub_epi16(
                _mm256_max_epi16(min_max, _mm256_sub_epi16(_mm256_setzero_si256(), max_min)),
                _mm256_set1_epi16(1));

            /* Pack scores to uint8 for NMS */
            __m128i slo = _mm256_castsi256_si128(score_v);
            __m128i shi = _mm256_extractf128_si256(score_v, 1);
            _mm_storeu_si128((__m128i*)(curr + j), _mm_packus_epi16(slo, shi));

            uint32_t mask = movemask_epi16(_mm256_cmpgt_epi16(score_v, vthr));
            while (mask)
            {
                int k = __builtin_ctz(mask);
                cornerpos[ncorners++] = j + k;
                mask &= mask - 1;
            }
        }
    }

    /* NMS=false path: original counting method (32-wide uint8) */
    void processCounting(int &j, const uchar* &ptr, int* cornerpos, int &ncorners)
    {
        static const __m256i delta256 = _mm256_broadcastsi128_si256(_mm_set1_epi8((char)(-128))), K16_256 = _mm256_broadcastsi128_si256(_mm_set1_epi8((char)8));
        const __m256i t256 = _mm256_broadcastsi128_si256(_mm_set1_epi8(t256c));
        for (; j < cols - 32 - 3; j += 32, ptr += 32)
        {
            __m256i m0, m1;
            __m256i v0 = _mm256_loadu_si256((const __m256i*)ptr);

            __m256i v1 = _mm256_xor_si256(_mm256_subs_epu8(v0, t256), delta256);
            v0 = _mm256_xor_si256(_mm256_adds_epu8(v0, t256), delta256);

            __m256i x0 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i*)(ptr + pixel[0])), delta256);
            __m256i x1 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i*)(ptr + pixel[4])), delta256);
            __m256i x2 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i*)(ptr + pixel[8])), delta256);
            __m256i x3 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i*)(ptr + pixel[12])), delta256);

            m0 = _mm256_and_si256(_mm256_cmpgt_epi8(x0, v0), _mm256_cmpgt_epi8(x1, v0));
            m1 = _mm256_and_si256(_mm256_cmpgt_epi8(v1, x0), _mm256_cmpgt_epi8(v1, x1));
            m0 = _mm256_or_si256(m0, _mm256_and_si256(_mm256_cmpgt_epi8(x1, v0), _mm256_cmpgt_epi8(x2, v0)));
            m1 = _mm256_or_si256(m1, _mm256_and_si256(_mm256_cmpgt_epi8(v1, x1), _mm256_cmpgt_epi8(v1, x2)));
            m0 = _mm256_or_si256(m0, _mm256_and_si256(_mm256_cmpgt_epi8(x2, v0), _mm256_cmpgt_epi8(x3, v0)));
            m1 = _mm256_or_si256(m1, _mm256_and_si256(_mm256_cmpgt_epi8(v1, x2), _mm256_cmpgt_epi8(v1, x3)));
            m0 = _mm256_or_si256(m0, _mm256_and_si256(_mm256_cmpgt_epi8(x3, v0), _mm256_cmpgt_epi8(x0, v0)));
            m1 = _mm256_or_si256(m1, _mm256_and_si256(_mm256_cmpgt_epi8(v1, x3), _mm256_cmpgt_epi8(v1, x0)));
            m0 = _mm256_or_si256(m0, m1);

            unsigned int mask = _mm256_movemask_epi8(m0);
            if (mask == 0)
                continue;
            if ((mask & 0xffff) == 0)
            {
                j -= 16;
                ptr -= 16;
                continue;
            }

            __m256i c0 = _mm256_setzero_si256(), c1 = c0, max0 = c0, max1 = c0;
            for (int k = 0; k < 25; k++)
            {
                __m256i x = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(ptr + pixel[k])), delta256);
                m0 = _mm256_cmpgt_epi8(x, v0);
                m1 = _mm256_cmpgt_epi8(v1, x);

                c0 = _mm256_and_si256(_mm256_sub_epi8(c0, m0), m0);
                c1 = _mm256_and_si256(_mm256_sub_epi8(c1, m1), m1);

                max0 = _mm256_max_epu8(max0, c0);
                max1 = _mm256_max_epu8(max1, c1);
            }

            max0 = _mm256_max_epu8(max0, max1);
            unsigned int m = _mm256_movemask_epi8(_mm256_cmpgt_epi8(max0, K16_256));

            for (int k = 0; m > 0 && k < 32; k++, m >>= 1)
                if (m & 1)
                    cornerpos[ncorners++] = j + k;
        }
    }

    virtual ~FAST_t_patternSize16_AVX2_Impl() CV_OVERRIDE {}
private:
    int cols;
    char t256c;
    int threshold;
    bool nonmax_suppression;
    const int* pixel;
};
Ptr<FAST_t_patternSize16_AVX2> FAST_t_patternSize16_AVX2::getImpl(int _cols, int _threshold, bool _nonmax_suppression, const int* _pixel)
{
    return Ptr<FAST_t_patternSize16_AVX2>(new FAST_t_patternSize16_AVX2_Impl(_cols, _threshold, _nonmax_suppression, _pixel));
}
}
}
