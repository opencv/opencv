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
namespace opt_NEON
{

/* Load 8 uchar and zero-extend to int16x8 */
static inline int16x8_t loadu_i16(const uchar* p)
{
    return vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p)));
}

/* Extract 8-bit bitmask from int16x8 comparison result */
static inline uint8_t movemask_s16(int16x8_t v)
{
    uint16x8_t u = vreinterpretq_u16_s16(v);
    uint8_t mask = 0;
    mask |= (vgetq_lane_u16(u, 0) >> 15) << 0;
    mask |= (vgetq_lane_u16(u, 1) >> 15) << 1;
    mask |= (vgetq_lane_u16(u, 2) >> 15) << 2;
    mask |= (vgetq_lane_u16(u, 3) >> 15) << 3;
    mask |= (vgetq_lane_u16(u, 4) >> 15) << 4;
    mask |= (vgetq_lane_u16(u, 5) >> 15) << 5;
    mask |= (vgetq_lane_u16(u, 6) >> 15) << 6;
    mask |= (vgetq_lane_u16(u, 7) >> 15) << 7;
    return mask;
}

class FAST_t_patternSize16_NEON_Impl CV_FINAL: public FAST_t_patternSize16_NEON
{
public:
    FAST_t_patternSize16_NEON_Impl(int _cols, int _threshold, bool _nonmax_suppression, const int* _pixel)
        : cols(_cols), nonmax_suppression(_nonmax_suppression), pixel(_pixel)
    {
        threshold = std::min(std::max(_threshold, 0), 255);
    }

    virtual void process(int &j, const uchar* &ptr, uchar* curr, int* cornerpos, int &ncorners) CV_OVERRIDE
    {
        /* Fused arc-detection-and-score approach using shared binary tree:
         * Compute the full corner score for all pixels simultaneously.
         * The score itself determines corner presence (score >= threshold).
         * Process 8 pixels per iteration in int16 via NEON. */
        const int16x8_t vthr = vdupq_n_s16((short)(threshold - 1));

        for (; j <= cols - 3 - 8; j += 8, ptr += 8)
        {
            /* Load center and 16 neighbors, compute differences d[i] = center - neighbor[i] */
            int16x8_t vcen = loadu_i16(ptr);
            int16x8_t d0  = vsubq_s16(vcen, loadu_i16(ptr + pixel[0]));
            int16x8_t d1  = vsubq_s16(vcen, loadu_i16(ptr + pixel[1]));
            int16x8_t d2  = vsubq_s16(vcen, loadu_i16(ptr + pixel[2]));
            int16x8_t d3  = vsubq_s16(vcen, loadu_i16(ptr + pixel[3]));
            int16x8_t d4  = vsubq_s16(vcen, loadu_i16(ptr + pixel[4]));
            int16x8_t d5  = vsubq_s16(vcen, loadu_i16(ptr + pixel[5]));
            int16x8_t d6  = vsubq_s16(vcen, loadu_i16(ptr + pixel[6]));
            int16x8_t d7  = vsubq_s16(vcen, loadu_i16(ptr + pixel[7]));
            int16x8_t d8  = vsubq_s16(vcen, loadu_i16(ptr + pixel[8]));
            int16x8_t d9  = vsubq_s16(vcen, loadu_i16(ptr + pixel[9]));
            int16x8_t d10 = vsubq_s16(vcen, loadu_i16(ptr + pixel[10]));
            int16x8_t d11 = vsubq_s16(vcen, loadu_i16(ptr + pixel[11]));
            int16x8_t d12 = vsubq_s16(vcen, loadu_i16(ptr + pixel[12]));
            int16x8_t d13 = vsubq_s16(vcen, loadu_i16(ptr + pixel[13]));
            int16x8_t d14 = vsubq_s16(vcen, loadu_i16(ptr + pixel[14]));
            int16x8_t d15 = vsubq_s16(vcen, loadu_i16(ptr + pixel[15]));

            /* Bright score: shared binary tree (min tree) */
            int16x8_t va0 = vminq_s16(d7, d8);
            int16x8_t va00 = vminq_s16(va0, d6);
            va00 = vminq_s16(va00, d5); va00 = vminq_s16(va00, d4);
            va00 = vminq_s16(va00, d3);

            int16x8_t va000 = vminq_s16(va00, d2); va000 = vminq_s16(va000, d1);
            int16x8_t va001 = vminq_s16(va00, d9); va001 = vminq_s16(va001, d10);

            int16x8_t va01 = vminq_s16(va0, d9); va01 = vminq_s16(va01, d10);
            va01 = vminq_s16(va01, d11); va01 = vminq_s16(va01, d12);

            int16x8_t va010 = vminq_s16(va01, d6); va010 = vminq_s16(va010, d5);
            int16x8_t va011 = vminq_s16(va01, d13); va011 = vminq_s16(va011, d14);

            int16x8_t min_max = vmaxq_s16(vminq_s16(va000, d0), vminq_s16(va000, d9));
            min_max = vmaxq_s16(min_max, vmaxq_s16(vminq_s16(va001, d2), vminq_s16(va001, d11)));
            min_max = vmaxq_s16(min_max, vmaxq_s16(vminq_s16(va010, d4), vminq_s16(va010, d13)));
            min_max = vmaxq_s16(min_max, vmaxq_s16(vminq_s16(va011, d6), vminq_s16(va011, d15)));

            int16x8_t va1 = vminq_s16(d15, d0);
            int16x8_t va10 = vminq_s16(va1, d14); va10 = vminq_s16(va10, d13);
            va10 = vminq_s16(va10, d12); va10 = vminq_s16(va10, d11);

            int16x8_t va100 = vminq_s16(va10, d10); va100 = vminq_s16(va100, d9);
            int16x8_t va101 = vminq_s16(va10, d1); va101 = vminq_s16(va101, d2);

            int16x8_t va11 = vminq_s16(va1, d1); va11 = vminq_s16(va11, d2);
            va11 = vminq_s16(va11, d3); va11 = vminq_s16(va11, d4);

            int16x8_t va110 = vminq_s16(va11, d14); va110 = vminq_s16(va110, d13);
            int16x8_t va111 = vminq_s16(va11, d5); va111 = vminq_s16(va111, d6);

            min_max = vmaxq_s16(min_max, vmaxq_s16(vminq_s16(va100, d8), vminq_s16(va100, d1)));
            min_max = vmaxq_s16(min_max, vmaxq_s16(vminq_s16(va101, d10), vminq_s16(va101, d3)));
            min_max = vmaxq_s16(min_max, vmaxq_s16(vminq_s16(va110, d12), vminq_s16(va110, d5)));
            min_max = vmaxq_s16(min_max, vmaxq_s16(vminq_s16(va111, d14), vminq_s16(va111, d7)));

            /* Dark score: shared binary tree (max tree) */
            int16x8_t vb0 = vmaxq_s16(d7, d8);
            int16x8_t vb00 = vmaxq_s16(vb0, d6);
            vb00 = vmaxq_s16(vb00, d5); vb00 = vmaxq_s16(vb00, d4);
            vb00 = vmaxq_s16(vb00, d3);

            int16x8_t vb000 = vmaxq_s16(vb00, d2); vb000 = vmaxq_s16(vb000, d1);
            int16x8_t vb001 = vmaxq_s16(vb00, d9); vb001 = vmaxq_s16(vb001, d10);

            int16x8_t vb01 = vmaxq_s16(vb0, d9); vb01 = vmaxq_s16(vb01, d10);
            vb01 = vmaxq_s16(vb01, d11); vb01 = vmaxq_s16(vb01, d12);

            int16x8_t vb010 = vmaxq_s16(vb01, d6); vb010 = vmaxq_s16(vb010, d5);
            int16x8_t vb011 = vmaxq_s16(vb01, d13); vb011 = vmaxq_s16(vb011, d14);

            int16x8_t max_min = vminq_s16(vmaxq_s16(vb000, d0), vmaxq_s16(vb000, d9));
            max_min = vminq_s16(max_min, vminq_s16(vmaxq_s16(vb001, d2), vmaxq_s16(vb001, d11)));
            max_min = vminq_s16(max_min, vminq_s16(vmaxq_s16(vb010, d4), vmaxq_s16(vb010, d13)));
            max_min = vminq_s16(max_min, vminq_s16(vmaxq_s16(vb011, d6), vmaxq_s16(vb011, d15)));

            int16x8_t vb1 = vmaxq_s16(d15, d0);
            int16x8_t vb10 = vmaxq_s16(vb1, d14); vb10 = vmaxq_s16(vb10, d13);
            vb10 = vmaxq_s16(vb10, d12); vb10 = vmaxq_s16(vb10, d11);

            int16x8_t vb100 = vmaxq_s16(vb10, d10); vb100 = vmaxq_s16(vb100, d9);
            int16x8_t vb101 = vmaxq_s16(vb10, d1); vb101 = vmaxq_s16(vb101, d2);

            int16x8_t vb11 = vmaxq_s16(vb1, d1); vb11 = vmaxq_s16(vb11, d2);
            vb11 = vmaxq_s16(vb11, d3); vb11 = vmaxq_s16(vb11, d4);

            int16x8_t vb110 = vmaxq_s16(vb11, d14); vb110 = vmaxq_s16(vb110, d13);
            int16x8_t vb111 = vmaxq_s16(vb11, d5); vb111 = vmaxq_s16(vb111, d6);

            max_min = vminq_s16(max_min, vminq_s16(vmaxq_s16(vb100, d8), vmaxq_s16(vb100, d1)));
            max_min = vminq_s16(max_min, vminq_s16(vmaxq_s16(vb101, d10), vmaxq_s16(vb101, d3)));
            max_min = vminq_s16(max_min, vminq_s16(vmaxq_s16(vb110, d12), vmaxq_s16(vb110, d5)));
            max_min = vminq_s16(max_min, vminq_s16(vmaxq_s16(vb111, d14), vmaxq_s16(vb111, d7)));

            /* Final score: max(bright, -dark) - 1 */
            int16x8_t score_v = vsubq_s16(
                vmaxq_s16(min_max, vnegq_s16(max_min)),
                vdupq_n_s16(1));

            /* Store scores as uchar for NMS (saturating narrow int16 -> uint8) */
            if (nonmax_suppression)
            {
                vst1_u8(curr + j, vqmovn_u16(vreinterpretq_u16_s16(
                    vmaxq_s16(vminq_s16(score_v, vdupq_n_s16(255)), vdupq_n_s16(0)))));
            }

            /* Corner detection: score >= threshold */
            uint16x8_t cmp_result = vcgtq_s16(score_v, vthr);
            uint8_t mask = movemask_s16(vreinterpretq_s16_u16(cmp_result));
            while (mask)
            {
                int k = __builtin_ctz(mask);
                cornerpos[ncorners++] = j + k;
                mask &= mask - 1;
            }
        }
    }

    virtual ~FAST_t_patternSize16_NEON_Impl() CV_OVERRIDE {}

private:
    int cols;
    int threshold;
    bool nonmax_suppression;
    const int* pixel;
};

Ptr<FAST_t_patternSize16_NEON> FAST_t_patternSize16_NEON::getImpl(int _cols, int _threshold, bool _nonmax_suppression, const int* _pixel)
{
    return Ptr<FAST_t_patternSize16_NEON>(new FAST_t_patternSize16_NEON_Impl(_cols, _threshold, _nonmax_suppression, _pixel));
}

}
}
