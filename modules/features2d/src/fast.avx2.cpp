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

class FAST_t_patternSize16_AVX2_Impl CV_FINAL: public FAST_t_patternSize16_AVX2
{
public:
    FAST_t_patternSize16_AVX2_Impl(int _cols, int _threshold, bool _nonmax_suppression, const int* _pixel):
                                   cols(_cols), nonmax_suppression(_nonmax_suppression), pixel(_pixel)
    {
        //patternSize = 16
        t256c = (char)_threshold;
        threshold = std::min(std::max(_threshold, 0), 255);
    }

    virtual void process(int &j, const uchar* &ptr, uchar* curr, int* cornerpos, int &ncorners) CV_OVERRIDE
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

            unsigned int mask = _mm256_movemask_epi8(m0); //unsigned is important!
            if (mask == 0){
                continue;
            }
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
                {
                    cornerpos[ncorners++] = j + k;
                    if (nonmax_suppression)
                    {
                        short d[25];
                        for (int q = 0; q < 25; q++)
                            d[q] = (short)(ptr[k] - ptr[k + pixel[q]]);
                        v_int16x8 q0 = v_setall_s16(-1000), q1 = v_setall_s16(1000);
                        for (int q = 0; q < 16; q += 8)
                        {
                            v_int16x8 v0_ = v_load(d + q + 1);
                            v_int16x8 v1_ = v_load(d + q + 2);
                            v_int16x8 a = v_min(v0_, v1_);
                            v_int16x8 b = v_max(v0_, v1_);
                            v0_ = v_load(d + q + 3);
                            a = v_min(a, v0_);
                            b = v_max(b, v0_);
                            v0_ = v_load(d + q + 4);
                            a = v_min(a, v0_);
                            b = v_max(b, v0_);
                            v0_ = v_load(d + q + 5);
                            a = v_min(a, v0_);
                            b = v_max(b, v0_);
                            v0_ = v_load(d + q + 6);
                            a = v_min(a, v0_);
                            b = v_max(b, v0_);
                            v0_ = v_load(d + q + 7);
                            a = v_min(a, v0_);
                            b = v_max(b, v0_);
                            v0_ = v_load(d + q + 8);
                            a = v_min(a, v0_);
                            b = v_max(b, v0_);
                            v0_ = v_load(d + q);
                            q0 = v_max(q0, v_min(a, v0_));
                            q1 = v_min(q1, v_max(b, v0_));
                            v0_ = v_load(d + q + 9);
                            q0 = v_max(q0, v_min(a, v0_));
                            q1 = v_min(q1, v_max(b, v0_));
                        }
                        q0 = v_max(q0, v_setzero_s16() - q1);
                        curr[j + k] = (uchar)(v_reduce_max(q0) - 1);
                    }
                }
        }
        _mm256_zeroupper();
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
