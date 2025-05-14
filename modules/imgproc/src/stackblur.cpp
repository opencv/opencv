// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
StackBlur - a fast almost Gaussian Blur
Theory: http://underdestruction.com/2004/02/25/stackblur-2004
The code has been borrowed from (https://github.com/flozz/StackBlur)
and adapted for OpenCV by Zihao Mu.

Below is the original copyright
*/

/*
Copyright (c) 2010 Mario Klingemann

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
 */


#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

#include <iostream>

using namespace std;

#define STACKBLUR_MAX_RADIUS 254

static unsigned short const stackblurMul[255] =
        {
                512,512,456,512,328,456,335,512,405,328,271,456,388,335,292,512,
                454,405,364,328,298,271,496,456,420,388,360,335,312,292,273,512,
                482,454,428,405,383,364,345,328,312,298,284,271,259,496,475,456,
                437,420,404,388,374,360,347,335,323,312,302,292,282,273,265,512,
                497,482,468,454,441,428,417,405,394,383,373,364,354,345,337,328,
                320,312,305,298,291,284,278,271,265,259,507,496,485,475,465,456,
                446,437,428,420,412,404,396,388,381,374,367,360,354,347,341,335,
                329,323,318,312,307,302,297,292,287,282,278,273,269,265,261,512,
                505,497,489,482,475,468,461,454,447,441,435,428,422,417,411,405,
                399,394,389,383,378,373,368,364,359,354,350,345,341,337,332,328,
                324,320,316,312,309,305,301,298,294,291,287,284,281,278,274,271,
                268,265,262,259,257,507,501,496,491,485,480,475,470,465,460,456,
                451,446,442,437,433,428,424,420,416,412,408,404,400,396,392,388,
                385,381,377,374,370,367,363,360,357,354,350,347,344,341,338,335,
                332,329,326,323,320,318,315,312,310,307,304,302,299,297,294,292,
                289,287,285,282,280,278,275,273,271,269,267,265,263,261,259
        };

static unsigned char const stackblurShr[255] =
        {
                9, 11, 12, 13, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17,
                17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19,
                19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20,
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21,
                21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
                22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23,
                23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
                23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
                23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
                23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24
        };

namespace cv{

#if (CV_SIMD || CV_SIMD_SCALABLE)
template<typename T>
inline int opRow(const T* , T* , const std::vector<ushort>& , const float , const int radius, const int CN, const int )
{
    return radius * CN;
}

template<>
inline int opRow<uchar>(const uchar* srcPtr, uchar* dstPtr, const std::vector<ushort>& kVec, const float , const int radius, const int CN, const int widthCN)
{
    int kernelSize = (int)kVec.size();

    int i = radius * CN;
    if (radius > STACKBLUR_MAX_RADIUS)
        return i;

    const int mulValTab= stackblurMul[radius];
    const int shrValTab= stackblurShr[radius];

    const int VEC_LINE = VTraits<v_uint8>::vlanes();

    if (kernelSize == 3)
    {
        v_uint32 v_mulVal = vx_setall_u32(mulValTab);
        for (; i <= widthCN - VEC_LINE; i += VEC_LINE)
        {
            v_uint16 x0l, x0h, x1l, x1h, x2l, x2h;
            v_expand(vx_load(srcPtr + i - CN), x0l, x0h);
            v_expand(vx_load(srcPtr + i), x1l, x1h);
            v_expand(vx_load(srcPtr + i + CN), x2l, x2h);

            x1l = v_add_wrap(v_add_wrap(x1l, x1l), v_add_wrap(x0l, x2l));
            x1h = v_add_wrap(v_add_wrap(x1h, x1h), v_add_wrap(x0h, x2h));

            v_uint32 y00, y01, y10, y11;
            v_expand(x1l, y00, y01);
            v_expand(x1h, y10, y11);

            y00 = v_shr(v_mul(y00, v_mulVal), shrValTab);
            y01 = v_shr(v_mul(y01, v_mulVal), shrValTab);
            y10 = v_shr(v_mul(y10, v_mulVal), shrValTab);
            y11 = v_shr(v_mul(y11, v_mulVal), shrValTab);

            v_store(dstPtr + i, v_pack(v_pack(y00, y01), v_pack(y10, y11)));
        }
    }
    else
    {
        const ushort * kx = kVec.data() + kernelSize/2;
        v_int32 v_mulVal = vx_setall_s32(mulValTab);
        v_int16 k0 = vx_setall_s16((short)(kx[0]));

        srcPtr += i;
        for( ; i <= widthCN - VEC_LINE; i += VEC_LINE, srcPtr += VEC_LINE)
        {
            v_uint8 v_src = vx_load(srcPtr);
            v_int32 s0, s1, s2, s3;
            v_mul_expand(v_reinterpret_as_s16(v_expand_low(v_src)), k0, s0, s1);
            v_mul_expand(v_reinterpret_as_s16(v_expand_high(v_src)), k0, s2, s3);

            int k = 1, j = CN;
            for (; k <= kernelSize / 2 - 1; k += 2, j += 2 * CN)
            {
                v_int16 k12 = v_reinterpret_as_s16(vx_setall_s32(((int)kx[k] & 0xFFFF) | ((int)kx[k + 1] << 16)));

                v_uint8 v_src0 = vx_load(srcPtr - j);
                v_uint8 v_src1 = vx_load(srcPtr - j - CN);
                v_uint8 v_src2 = vx_load(srcPtr + j);
                v_uint8 v_src3 = vx_load(srcPtr + j + CN);

                v_int16 xl, xh;
                v_zip(v_reinterpret_as_s16(v_add(v_expand_low(v_src0), v_expand_low(v_src2))), v_reinterpret_as_s16(v_add(v_expand_low(v_src1), v_expand_low(v_src3))), xl, xh);
                s0 = v_add(s0, v_dotprod(xl, k12));
                s1 = v_add(s1, v_dotprod(xh, k12));
                v_zip(v_reinterpret_as_s16(v_add(v_expand_high(v_src0), v_expand_high(v_src2))), v_reinterpret_as_s16(v_add(v_expand_high(v_src1), v_expand_high(v_src3))), xl, xh);
                s2 = v_add(s2, v_dotprod(xl, k12));
                s3 = v_add(s3, v_dotprod(xh, k12));
            }
            if( k < kernelSize / 2 + 1 )
            {
                v_int16 k1 = vx_setall_s16((short)(kx[k]));

                v_uint8 v_src0 = vx_load(srcPtr - j);
                v_uint8 v_src1 = vx_load(srcPtr + j);

                v_int16 xl, xh;
                v_zip(v_reinterpret_as_s16(v_expand_low(v_src0)), v_reinterpret_as_s16(v_expand_low(v_src1)), xl, xh);
                s0 = v_add(s0, v_dotprod(xl, k1));
                s1 = v_add(s1, v_dotprod(xh, k1));
                v_zip(v_reinterpret_as_s16(v_expand_high(v_src0)), v_reinterpret_as_s16(v_expand_high(v_src1)), xl, xh);
                s2 = v_add(s2, v_dotprod(xl, k1));
                s3 = v_add(s3, v_dotprod(xh, k1));
            }

            s0 = v_shr(v_mul(s0, v_mulVal), shrValTab);
            s1 = v_shr(v_mul(s1, v_mulVal), shrValTab);
            s2 = v_shr(v_mul(s2, v_mulVal), shrValTab);
            s3 = v_shr(v_mul(s3, v_mulVal), shrValTab);

            v_store(dstPtr + i, v_pack(v_reinterpret_as_u16(v_pack(s0, s1)), v_reinterpret_as_u16(v_pack(s2, s3))));
        }
    }
    return i;
}

template<>
inline int opRow<ushort>(const ushort* srcPtr, ushort* dstPtr, const std::vector<ushort>& kVec, const float , const int radius, const int CN, const int widthCN)
{
    int kernelSize = (int)kVec.size();

    int i = radius * CN;
    if (radius > STACKBLUR_MAX_RADIUS)
        return i;

    const int mulValTab= stackblurMul[radius];
    const int shrValTab= stackblurShr[radius];

    const int VEC_LINE = VTraits<v_uint16>::vlanes();

    v_uint32 v_mulVal = vx_setall_u32(mulValTab);
    if (kernelSize == 3)
    {
        for (; i <= widthCN - VEC_LINE; i += VEC_LINE)
        {
            v_uint32 x0l, x0h, x1l, x1h, x2l, x2h;
            v_expand(vx_load(srcPtr + i - CN), x0l, x0h);
            v_expand(vx_load(srcPtr + i), x1l, x1h);
            v_expand(vx_load(srcPtr + i + CN), x2l, x2h);

            x1l = v_add(v_add(x1l, x1l), v_add(x0l, x2l));
            x1h = v_add(v_add(x1h, x1h), v_add(x0h, x2h));

            v_store(dstPtr + i, v_pack(v_shr(v_mul(x1l, v_mulVal), shrValTab), v_shr(v_mul(x1h, v_mulVal), shrValTab)));
        }
    }
    else
    {
        const ushort * kx = kVec.data() + kernelSize/2;
        v_uint16 k0 = vx_setall_u16(kx[0]);

        srcPtr += i;
        for( ; i <= widthCN - VEC_LINE; i += VEC_LINE, srcPtr += VEC_LINE)
        {
            v_uint16 v_src = vx_load(srcPtr);
            v_uint32 s0, s1;

            v_mul_expand(v_src, k0, s0, s1);

            int k = 1, j = CN;
            for (; k <= kernelSize / 2 - 1; k += 2, j += 2*CN)
            {
                v_uint16 k1 = vx_setall_u16(kx[k]);
                v_uint16 k2 = vx_setall_u16(kx[k + 1]);

                v_uint32 y0, y1;
                v_mul_expand(v_add(vx_load(srcPtr - j), vx_load(srcPtr + j)), k1, y0, y1);
                s0 = v_add(s0, y0);
                s1 = v_add(s1, y1);
                v_mul_expand(v_add(vx_load(srcPtr - j - CN), vx_load(srcPtr + j + CN)), k2, y0, y1);
                s0 = v_add(s0, y0);
                s1 = v_add(s1, y1);
            }
            if( k < kernelSize / 2 + 1 )
            {
                v_uint16 k1 = vx_setall_u16(kx[k]);

                v_uint32 y0, y1;
                v_mul_expand(v_add(vx_load(srcPtr - j), vx_load(srcPtr + j)), k1, y0, y1);
                s0 = v_add(s0, y0);
                s1 = v_add(s1, y1);
            }

            s0 = v_shr(v_mul(s0, v_mulVal), shrValTab);
            s1 = v_shr(v_mul(s1, v_mulVal), shrValTab);

            v_store(dstPtr + i, v_pack(s0, s1));
        }
    }

    return i;
}

template<>
inline int opRow<short>(const short* srcPtr, short* dstPtr, const std::vector<ushort>& kVec, const float , const int radius, const int CN, const int widthCN)
{
    int kernelSize = (int)kVec.size();
    int i = radius * CN;

    if (radius > STACKBLUR_MAX_RADIUS)
        return i;

    const int mulValTab= stackblurMul[radius];
    const int shrValTab= stackblurShr[radius];

    const int VEC_LINE = VTraits<v_int16>::vlanes();
    v_int32 v_mulVal = vx_setall_s32(mulValTab);

    if (kernelSize == 3)
    {
        for (; i <= widthCN - VEC_LINE; i += VEC_LINE)
        {
            v_int32 x0l, x0h, x1l, x1h, x2l, x2h;
            v_expand(vx_load(srcPtr + i - CN), x0l, x0h);
            v_expand(vx_load(srcPtr + i), x1l, x1h);
            v_expand(vx_load(srcPtr + i + CN), x2l, x2h);

            x1l = v_add(v_add(x1l, x1l), v_add(x0l, x2l));
            x1h = v_add(v_add(x1h, x1h), v_add(x0h, x2h));

            v_store(dstPtr + i, v_pack(v_shr(v_mul(x1l, v_mulVal), shrValTab), v_shr(v_mul(x1h, v_mulVal), shrValTab)));
        }
    }
    else
    {
        const ushort * kx = kVec.data() + kernelSize/2;
        v_int16 k0 = vx_setall_s16((short)(kx[0]));

        srcPtr += i;
        for( ; i <= widthCN - VEC_LINE; i += VEC_LINE, srcPtr += VEC_LINE)
        {
            v_int16 v_src = vx_load(srcPtr);
            v_int32 s0, s1;
            v_mul_expand(v_src, k0, s0, s1);

            int k = 1, j = CN;
            for (; k <= kernelSize / 2 - 1; k += 2, j += 2 * CN)
            {
                v_int16 k1 = vx_setall_s16((short)kx[k]);
                v_int16 k2 = vx_setall_s16((short)kx[k + 1]);

                v_int32 y0, y1;

                v_mul_expand(v_add(vx_load(srcPtr - j), vx_load(srcPtr + j)), k1, y0, y1);
                s0 = v_add(s0, y0);
                s1 = v_add(s1, y1);
                v_mul_expand(v_add(vx_load(srcPtr - j - CN), vx_load(srcPtr + j + CN)), k2, y0, y1);
                s0 = v_add(s0, y0);
                s1 = v_add(s1, y1);
            }
            if( k < kernelSize / 2 + 1 )
            {
                v_int16 k1 = vx_setall_s16((short)kx[k]);
                v_int32 y0, y1;
                v_mul_expand(v_add(vx_load(srcPtr - j), vx_load(srcPtr + j)), k1, y0, y1);
                s0 = v_add(s0, y0);
                s1 = v_add(s1, y1);
            }

            s0 = v_shr(v_mul(s0, v_mulVal), shrValTab);
            s1 = v_shr(v_mul(s1, v_mulVal), shrValTab);

            v_store(dstPtr + i, v_pack(s0, s1));
        }
    }
    return i;
}

template<>
inline int opRow<float>(const float* srcPtr, float* dstPtr, const std::vector<ushort>& kVec, const float mulVal, const int radius, const int CN, const int widthCN)
{
    int kernelSize = (int)kVec.size();
    int i = radius * CN;

    v_float32 v_mulVal = vx_setall_f32(mulVal);
    const int VEC_LINE = VTraits<v_float32>::vlanes();
    const int VEC_LINE4 = VEC_LINE * 4;

    if (kernelSize == 3)
    {
        for (; i <= widthCN - VEC_LINE4; i += VEC_LINE4)
        {
            v_float32 v_srcPtr0 = vx_load(srcPtr + i);
            v_float32 v_srcPtr1 = vx_load(srcPtr + VEC_LINE + i) ;
            v_float32 v_srcPtr2 = vx_load(srcPtr + VEC_LINE * 2 + i);
            v_float32 v_srcPtr3 = vx_load(srcPtr + VEC_LINE * 3 + i);

            v_float32 v_sumVal0 =  v_add(v_add(v_add(v_srcPtr0, v_srcPtr0), vx_load(srcPtr + i - CN)), vx_load(srcPtr + i + CN));
            v_float32 v_sumVal1 =  v_add(v_add(v_add(v_srcPtr1, v_srcPtr1), vx_load(srcPtr + VEC_LINE + i - CN)), vx_load(srcPtr + VEC_LINE + i + CN));
            v_float32 v_sumVal2 =  v_add(v_add(v_add(v_srcPtr2, v_srcPtr2), vx_load(srcPtr + VEC_LINE * 2 + i - CN)), vx_load(srcPtr + VEC_LINE * 2 + i + CN));
            v_float32 v_sumVal3 =  v_add(v_add(v_add(v_srcPtr3, v_srcPtr3), vx_load(srcPtr + VEC_LINE * 3 + i - CN)), vx_load(srcPtr + VEC_LINE * 3 + i + CN));

            v_store(dstPtr + i, v_mul(v_sumVal0, v_mulVal));
            v_store(dstPtr + i + VEC_LINE, v_mul(v_sumVal1, v_mulVal));
            v_store(dstPtr + i + VEC_LINE * 2, v_mul(v_sumVal2, v_mulVal));
            v_store(dstPtr + i + VEC_LINE * 3, v_mul(v_sumVal3, v_mulVal));
        }

        for (; i <= widthCN - VEC_LINE; i += VEC_LINE)
        {
            v_float32 v_srcPtr = vx_load(srcPtr + i);
            v_float32 v_sumVal = v_add(v_add(v_add(v_srcPtr, v_srcPtr), vx_load(srcPtr + i - CN)), vx_load(srcPtr + i + CN));
            v_store(dstPtr + i, v_mul(v_sumVal, v_mulVal));
        }
    }
    else
    {
        const ushort * kx = kVec.data() + kernelSize/2;
        v_float32 k0 = vx_setall_f32((float)(kx[0]));

        srcPtr += i;
        for( ; i <= widthCN - VEC_LINE; i += VEC_LINE, srcPtr += VEC_LINE)
        {
            v_float32 v_src = vx_load(srcPtr);
            v_float32 s0;
            s0 = v_mul(v_src, k0);

            int k = 1, j = CN;
            for (; k <= kernelSize / 2 - 1; k += 2, j += 2 * CN)
            {
                v_float32 k1 = vx_setall_f32((float)kx[k]);
                v_float32 k2 = vx_setall_f32((float)kx[k + 1]);

                s0 = v_add(s0, v_mul(v_add(vx_load(srcPtr - j), vx_load(srcPtr + j)), k1));
                s0 = v_add(s0, v_mul(v_add(vx_load(srcPtr - j - CN), vx_load(srcPtr + j + CN)), k2));
            }
            if( k < kernelSize / 2 + 1 )
            {
                v_float32 k1 = vx_setall_f32((float)kx[k]);

                s0 = v_add(s0, v_mul(v_add(vx_load(srcPtr - j), vx_load(srcPtr + j)), k1));
            }

            v_store(dstPtr + i, v_mul(s0, v_mulVal));
        }
    }
    return i;
}

template<typename T, typename TBuf>
inline int opComputeDiff(const T*& , TBuf*& , const int , const int)
{
    return 0;
}

template<>
inline int opComputeDiff<uchar, int>(const uchar*& srcPtr, int*& diff0, const int w, const int CNR1)
{
    int index = 0;
    const int VEC_LINE_8 = VTraits<v_uint8>::vlanes();
    const int VEC_LINE_32 = VTraits<v_int32>::vlanes();
    for (; index <= w - VEC_LINE_8; index += VEC_LINE_8, diff0+=VEC_LINE_8, srcPtr+=VEC_LINE_8)
    {
        v_uint16 x0l, x0h, x1l, x1h;
        v_expand(vx_load(srcPtr + CNR1), x0l, x0h);
        v_expand(vx_load(srcPtr), x1l, x1h);

        v_int32 y0, y1, y2, y3;
        v_expand(v_sub(v_reinterpret_as_s16(x0l), v_reinterpret_as_s16(x1l)), y0, y1);
        v_expand(v_sub(v_reinterpret_as_s16(x0h), v_reinterpret_as_s16(x1h)), y2, y3);

        v_store(diff0, y0);
        v_store(diff0 + VEC_LINE_32, y1);
        v_store(diff0 + VEC_LINE_32 * 2, y2);
        v_store(diff0 + VEC_LINE_32 * 3, y3);
    }
    return index;
}
#endif

template<typename T, typename TBuf>
class ParallelStackBlurRow : public ParallelLoopBody
{
public:
    ParallelStackBlurRow (const Mat &_src, Mat &_dst, int _radius): src(_src), dst(_dst) ,radius(_radius)
    {
        width= dst.cols;
        wm = width - 1;
        mulVal = 1.0f / ((radius + 1) * (radius + 1));
        CN = src.channels();
    }

    ~ParallelStackBlurRow() {}

    /*
     * The idea is as follows:
     * The stack can be understood as a sliding window of length kernel size.
     * The sliding window moves one element at a time from left to right.
     * The sumIn stores the elements added to the stack each time,
     * and sumOut stores the subtracted elements. Every time stack moves, stack, sumIn and sumOut are updated.
     * The dst will be calculated using the following formula:
     * dst[i] = (stack + sumIn - sumOut) / stack_num
     * In the Row direction, in order to avoid redundant computation,
     * we save the sumIn - sumOut as a diff vector.
     * So the new formula is:
     * dst[i] = (stack + diff[i]) / stack_num.
     * In practice, we use multiplication and bit shift right to simulate integer division:
     * dst[i] = ((stack + diff[i]) * mulVal) >> shrVal.
     * */
    virtual void operator ()(const Range& range) const CV_OVERRIDE
    {
        const int kernelSize = 2 * radius + 1;

        if (kernelSize <= 9 && width > kernelSize) // Special branch for small kernel
        {
            std::vector<ushort> kVec;
            for (int i = 0; i < kernelSize; i++)
            {
                if (i <= radius)
                    kVec.push_back(ushort(i + 1));
                else
                    kVec.push_back(ushort(2 * radius - i + 1));
            }

            const ushort * kx = kVec.data() + kernelSize/2;
            for (int row = range.start; row < range.end; row++)
            {
                const T* srcPtr = src.ptr<T>(row);
                T* dstPtr = dst.ptr<T>(row);
                TBuf sumVal;

                // init
                for (int i = 0; i < radius; i++)
                {
                    for (int ci = 0; ci < CN; ci++)
                    {
                        sumVal = 0;
                        for (int k = 0; k < kernelSize; k++)
                        {
                            int index = std::max(k - radius + i, 0);
                            sumVal += (TBuf)srcPtr[index * CN + ci] * (TBuf)kVec[k];
                        }
                        dstPtr[i*CN + ci] = (T)(sumVal * mulVal);
                    }
                }

                int widthCN = (width - radius) * CN;

                // middle
                int wc = radius * CN;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                wc = opRow<T>(srcPtr, dstPtr, kVec, mulVal, radius, CN, widthCN);
#endif
                for (; wc < widthCN; wc++)
                {
                    sumVal = srcPtr[wc] * kx[0];
                    for (int k = 1; k <= radius; k++)
                        sumVal += ((TBuf)(srcPtr[wc + k * CN])+(TBuf)(srcPtr[wc - k * CN])) * (TBuf)kx[k];
                    dstPtr[wc] = (T)(sumVal * mulVal);
                }

                // tail
                for (int i = wc / CN; i < width; i++)
                {
                    for (int ci = 0; ci < CN; ci++)
                    {
                        sumVal = 0;
                        for (int k = 0; k < kernelSize; k++)
                        {
                            int index = std::min(k - radius + i, wm);
                            sumVal += (TBuf)srcPtr[index * CN + ci] * (TBuf)kVec[k];
                        }
                        dstPtr[i*CN + ci] = (T)(sumVal * mulVal);
                    }
                }

            }
        }
        else
        {
            size_t bufSize = CN * (width + kernelSize) * sizeof(TBuf) + 2 * CN * sizeof(TBuf);
            AutoBuffer<uchar> _buf(bufSize + 16);
            uchar* bufptr = alignPtr(_buf.data(), 16);
            TBuf* diffVal = (TBuf*)bufptr;
            TBuf* sum = diffVal+CN;
            TBuf* diff = sum + CN;

            const int CNR1 = CN * (radius + 1);
            const int widthCN = (width - radius - 1) * CN;

            for (int row = range.start; row < range.end; row++)
            {
                memset(bufptr, 0, bufSize);

                const T* srcPtr = src.ptr<T>(row);
                T* dstPtr = dst.ptr<T>(row);

                int radiusMul = (radius + 2) * (radius + 1) / 2;
                for (int ci = 0; ci < CN; ci++)
                    sum[ci] += (TBuf)srcPtr[ci] * radiusMul;

                // compute diff
                const T* srcPtr0 = srcPtr;

                // init
                for (int i = 0; i < radius; i++)
                {
                    if (i < wm) srcPtr0 += CN;
                    for (int ci = 0; ci < CN; ci++)
                    {
                        diff[i*CN + ci] = (TBuf)srcPtr0[ci] - (TBuf)srcPtr[ci];
                        diffVal[ci] += diff[i*CN + ci];
                        sum[ci] += srcPtr0[ci] * (radius - i);
                    }
                }

                // middle
                auto diff0 = diff + radius * CN;
                int index = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                index = opComputeDiff(srcPtr, diff0, widthCN, CNR1);
#endif

                for (; index < widthCN; index++, diff0++, srcPtr++)
                    diff0[0] = (TBuf)(srcPtr[CNR1]) - (TBuf)(srcPtr[0]);

                // tails
                srcPtr0 = src.ptr<T>(row) + index;
                const T* srcPtr1 = src.ptr<T>(row) + (width - 1) * CN;
                int dist = width - index/CN;
                for (int r = 0; r < radius; r++, diff0 += CN)
                {
                    for (int ci = 0; ci < CN; ci++)
                        diff0[ci] = (TBuf)(srcPtr1[ci]) - (TBuf)(srcPtr0[ci]);

                    if (dist >= r)
                    {
                        srcPtr0 += CN;
                        dist--;
                    }
                }

                srcPtr = src.ptr<T>(row);
                diff0 = diff + radius * CN;
                for (int ci = 0; ci < CN; ci++)
                    diffVal[ci] += diff0[ci];
                diff0 += CN;

                if (CN == 1)
                {
                    for (int i = 0; i < width; i++, diff0 ++, dstPtr ++, srcPtr ++)
                    {
                        *(dstPtr) = saturate_cast<T>((sum[0] * mulVal));
                        sum[0] += diffVal[0];
                        diffVal[0] += (diff0[0] - diff0[-CNR1]);
                    }
                }
                else if (CN == 3)
                {
                    for (int i = 0; i < width; i++, diff0 += CN, dstPtr += CN, srcPtr += CN)
                    {
                        *(dstPtr + 0) = saturate_cast<T>((sum[0] * mulVal));
                        *(dstPtr + 1) = saturate_cast<T>((sum[1] * mulVal));
                        *(dstPtr + 2) = saturate_cast<T>((sum[2] * mulVal));

                        sum[0] += diffVal[0];
                        sum[1] += diffVal[1];
                        sum[2] += diffVal[2];

                        diffVal[0] += (diff0[0] - diff0[0 - CNR1]);
                        diffVal[1] += (diff0[1] - diff0[1 - CNR1]);
                        diffVal[2] += (diff0[2] - diff0[2 - CNR1]);
                    }
                }
                else if (CN == 4)
                {
                    for (int i = 0; i < width; i++, diff0 += CN, dstPtr += CN, srcPtr += CN)
                    {
                        *(dstPtr + 0) = saturate_cast<T>((sum[0] * mulVal));
                        *(dstPtr + 1) = saturate_cast<T>((sum[1] * mulVal));
                        *(dstPtr + 2) = saturate_cast<T>((sum[2] * mulVal));
                        *(dstPtr + 3) = saturate_cast<T>((sum[3] * mulVal));

                        sum[0] += diffVal[0];
                        sum[1] += diffVal[1];
                        sum[2] += diffVal[2];
                        sum[3] += diffVal[3];

                        diffVal[0] += (diff0[0] - diff0[0 - CNR1]);
                        diffVal[1] += (diff0[1] - diff0[1 - CNR1]);
                        diffVal[2] += (diff0[2] - diff0[2 - CNR1]);
                        diffVal[3] += (diff0[3] - diff0[3 - CNR1]);
                    }
                }
                else
                {
                    int i = 0;
                    for (; i < width; i++, diff0 += CN, dstPtr += CN, srcPtr += CN)
                    {
                        for (int ci = 0; ci < CN; ci++)
                        {
                            *(dstPtr + ci) = saturate_cast<T>((sum[ci] * mulVal));
                            sum[ci] += diffVal[ci];
                            diffVal[ci] += (diff0[ci] - diff0[ci - CNR1]);
                        }
                    }
                }
            }
        }
    }

private:
    const Mat &src;
    Mat &dst;
    int radius;
    int width;
    int wm;
    int CN;
    float mulVal;
};

#if (CV_SIMD || CV_SIMD_SCALABLE)
template<typename T, typename TBuf>
inline int opColumn(const T* , T* , T* , TBuf* , TBuf* , TBuf* , const float ,
                    const int , const int , const int , const int , const int )
{
    return 0;
}

template<>
inline int opColumn<float, float>(const float* srcPtr, float* dstPtr, float* stack, float* sum, float* sumIn,
                                  float* sumOut, const float mulVal, const int , const int ,
                                  const int widthLen, const int ss, const int sp1)
{
    int k = 0;
    v_float32 v_mulVal = vx_setall_f32(mulVal);
    const int VEC_LINE = VTraits<v_float32>::vlanes();
    const int VEC_LINE4 = 4 * VEC_LINE;

    auto stackStartPtr = stack + ss * widthLen;
    auto stackSp1Ptr = stack + sp1 * widthLen;

    for (;k <= widthLen - VEC_LINE4; k += VEC_LINE4)
    {
        v_float32 v_sum0 = vx_load(sum + k);
        v_float32 v_sum1 = vx_load(sum + VEC_LINE + k);
        v_float32 v_sum2 = vx_load(sum + VEC_LINE * 2 + k);
        v_float32 v_sum3 = vx_load(sum + VEC_LINE * 3 + k);

        v_float32 v_sumOut0 = vx_load(sumOut + k);
        v_float32 v_sumOut1 = vx_load(sumOut + VEC_LINE + k);
        v_float32 v_sumOut2 = vx_load(sumOut + VEC_LINE * 2 + k);
        v_float32 v_sumOut3 = vx_load(sumOut + VEC_LINE * 3 + k);

        v_float32 v_sumIn0 = vx_load(sumIn + k);
        v_float32 v_sumIn1 = vx_load(sumIn + VEC_LINE + k);
        v_float32 v_sumIn2 = vx_load(sumIn + VEC_LINE * 2 + k);
        v_float32 v_sumIn3 = vx_load(sumIn + VEC_LINE * 3+ k);

        v_store(dstPtr + k, v_mul(v_sum0, v_mulVal));
        v_store(dstPtr + VEC_LINE + k, v_mul(v_sum1, v_mulVal));
        v_store(dstPtr + VEC_LINE * 2 + k, v_mul(v_sum2, v_mulVal));
        v_store(dstPtr + VEC_LINE * 3 + k, v_mul(v_sum3, v_mulVal));

        v_sum0 = v_sub(v_sum0, v_sumOut0);
        v_sum1 = v_sub(v_sum1, v_sumOut1);
        v_sum2 = v_sub(v_sum2, v_sumOut2);
        v_sum3 = v_sub(v_sum3, v_sumOut3);

        v_sumOut0 = v_sub(v_sumOut0, vx_load(stackStartPtr + k));
        v_sumOut1 = v_sub(v_sumOut1, vx_load(stackStartPtr + VEC_LINE + k));
        v_sumOut2 = v_sub(v_sumOut2, vx_load(stackStartPtr + VEC_LINE * 2 + k));
        v_sumOut3 = v_sub(v_sumOut3, vx_load(stackStartPtr + VEC_LINE * 3 + k));

        v_float32 v_srcPtr0 = vx_load(srcPtr + k);
        v_float32 v_srcPtr1 = vx_load(srcPtr + VEC_LINE + k);
        v_float32 v_srcPtr2 = vx_load(srcPtr + VEC_LINE * 2 + k);
        v_float32 v_srcPtr3 = vx_load(srcPtr + VEC_LINE * 3 + k);

        v_store(stackStartPtr + k, v_srcPtr0);
        v_store(stackStartPtr + VEC_LINE + k, v_srcPtr1);
        v_store(stackStartPtr + VEC_LINE * 2 + k, v_srcPtr2);
        v_store(stackStartPtr + VEC_LINE * 3 + k, v_srcPtr3);

        v_sumIn0 = v_add(v_sumIn0, v_srcPtr0);
        v_sumIn1 = v_add(v_sumIn1, v_srcPtr1);
        v_sumIn2 = v_add(v_sumIn2, v_srcPtr2);
        v_sumIn3 = v_add(v_sumIn3, v_srcPtr3);

        v_store(sum + k, v_add(v_sum0, v_sumIn0));
        v_store(sum + VEC_LINE + k, v_add(v_sum1, v_sumIn1));
        v_store(sum + VEC_LINE * 2 + k, v_add(v_sum2, v_sumIn2));
        v_store(sum + VEC_LINE * 3 + k, v_add(v_sum3, v_sumIn3));

        v_srcPtr0 = vx_load(stackSp1Ptr + k);
        v_srcPtr1 = vx_load(stackSp1Ptr + VEC_LINE + k);
        v_srcPtr2 = vx_load(stackSp1Ptr + VEC_LINE * 2 +  k);
        v_srcPtr3 = vx_load(stackSp1Ptr + VEC_LINE * 3 + k);

        v_sumOut0 = v_add(v_sumOut0, v_srcPtr0);
        v_sumOut1 = v_add(v_sumOut1, v_srcPtr1);
        v_sumOut2 = v_add(v_sumOut2, v_srcPtr2);
        v_sumOut3 = v_add(v_sumOut3, v_srcPtr3);

        v_store(sumOut + k, v_sumOut0);
        v_store(sumOut + VEC_LINE + k, v_sumOut1);
        v_store(sumOut + VEC_LINE * 2 + k, v_sumOut2);
        v_store(sumOut + VEC_LINE * 3 + k, v_sumOut3);

        v_sumIn0 = v_sub(v_sumIn0, v_srcPtr0);
        v_sumIn1 = v_sub(v_sumIn1, v_srcPtr1);
        v_sumIn2 = v_sub(v_sumIn2, v_srcPtr2);
        v_sumIn3 = v_sub(v_sumIn3, v_srcPtr3);

        v_store(sumIn + k, v_sumIn0);
        v_store(sumIn + VEC_LINE + k, v_sumIn1);
        v_store(sumIn + VEC_LINE * 2 + k, v_sumIn2);
        v_store(sumIn + VEC_LINE * 3 + k, v_sumIn3);
    }

    for (;k <= widthLen - VEC_LINE; k += VEC_LINE)
    {
        v_float32 v_sum = vx_load(sum + k);
        v_float32 v_sumOut = vx_load(sumOut + k);
        v_float32 v_sumIn = vx_load(sumIn + k);

        v_store(dstPtr + k, v_mul(v_sum, v_mulVal));
        v_sum = v_sub(v_sum, v_sumOut);
        v_sumOut = v_sub(v_sumOut, vx_load(stackStartPtr + k));

        v_float32 v_srcPtr = vx_load(srcPtr + k);
        v_store(stackStartPtr + k, v_srcPtr);

        v_sumIn = v_add(v_sumIn, v_srcPtr);
        v_store(sum + k, v_add(v_sum, v_sumIn));

        v_srcPtr = vx_load(stackSp1Ptr + k);
        v_sumOut = v_add(v_sumOut, v_srcPtr);
        v_store(sumOut + k, v_sumOut);
        v_sumIn = v_sub(v_sumIn, v_srcPtr);
        v_store(sumIn + k, v_sumIn);
    }
    return k;
}

template<>
inline int opColumn<uchar, int>(const uchar* srcPtr, uchar* dstPtr, uchar* stack, int* sum, int* sumIn,
                                int* sumOut, const float , const int mulValTab, const int shrValTab,
                                const int widthLen, const int ss, const int sp1)
{
    int k = 0;
    if (mulValTab != 0 && shrValTab != 0)
    {
        const int VEC_LINE_8 = VTraits<v_uint8>::vlanes();
        const int VEC_LINE_32 = VTraits<v_int32>::vlanes();
        v_int32 v_mulVal = vx_setall_s32(mulValTab);

        auto stackStartPtr = stack + ss * widthLen;
        auto stackSp1Ptr = stack + sp1 * widthLen;

        for (;k <= widthLen - VEC_LINE_8; k += VEC_LINE_8)
        {
            v_int32 v_sum0, v_sum1, v_sum2, v_sum3;
            v_int32 v_sumIn0, v_sumIn1, v_sumIn2, v_sumIn3;
            v_int32 v_sumOut0, v_sumOut1, v_sumOut2, v_sumOut3;

            v_sum0 = vx_load(sum + k);
            v_sum1 = vx_load(sum + k + VEC_LINE_32);
            v_sum2 = vx_load(sum + k + VEC_LINE_32 * 2);
            v_sum3 = vx_load(sum + k + VEC_LINE_32 * 3);

            v_sumIn0 = vx_load(sumIn + k);
            v_sumIn1 = vx_load(sumIn + k + VEC_LINE_32);
            v_sumIn2 = vx_load(sumIn + k + VEC_LINE_32 * 2);
            v_sumIn3 = vx_load(sumIn + k + VEC_LINE_32 * 3);

            v_sumOut0 = vx_load(sumOut + k);
            v_sumOut1 = vx_load(sumOut + k + VEC_LINE_32);
            v_sumOut2 = vx_load(sumOut + k + VEC_LINE_32 * 2);
            v_sumOut3 = vx_load(sumOut + k + VEC_LINE_32 * 3);

            v_store(dstPtr + k,
                    v_pack(
                            v_reinterpret_as_u16(v_pack(v_shr(v_mul(v_sum0, v_mulVal), shrValTab), v_shr(v_mul(v_sum1, v_mulVal), shrValTab))),
                            v_reinterpret_as_u16(v_pack(v_shr(v_mul(v_sum2, v_mulVal), shrValTab), v_shr(v_mul(v_sum3, v_mulVal), shrValTab)))));

            v_sum0 = v_sub(v_sum0, v_sumOut0);
            v_sum1 = v_sub(v_sum1, v_sumOut1);
            v_sum2 = v_sub(v_sum2, v_sumOut2);
            v_sum3 = v_sub(v_sum3, v_sumOut3);

            v_uint16 x0l, x0h;
            v_int32 v_ss0, v_ss1, v_ss2, v_ss3;

            v_expand(vx_load(stackStartPtr + k), x0l, x0h);
            v_expand(v_reinterpret_as_s16(x0l), v_ss0, v_ss1);
            v_expand(v_reinterpret_as_s16(x0h), v_ss2, v_ss3);

            v_sumOut0 = v_sub(v_sumOut0, v_ss0);
            v_sumOut1 = v_sub(v_sumOut1, v_ss1);
            v_sumOut2 = v_sub(v_sumOut2, v_ss2);
            v_sumOut3 = v_sub(v_sumOut3, v_ss3);

            v_expand(vx_load(srcPtr + k), x0l, x0h);
            v_expand(v_reinterpret_as_s16(x0l), v_ss0, v_ss1);
            v_expand(v_reinterpret_as_s16(x0h), v_ss2, v_ss3);

            memcpy(stackStartPtr + k,srcPtr + k, VEC_LINE_8 * sizeof (uchar));

            v_sumIn0 = v_add(v_sumIn0, v_ss0);
            v_sumIn1 = v_add(v_sumIn1, v_ss1);
            v_sumIn2 = v_add(v_sumIn2, v_ss2);
            v_sumIn3 = v_add(v_sumIn3, v_ss3);

            v_store(sum + k, v_add(v_sum0, v_sumIn0));
            v_store(sum + VEC_LINE_32 + k, v_add(v_sum1, v_sumIn1));
            v_store(sum + VEC_LINE_32 * 2 + k, v_add(v_sum2, v_sumIn2));
            v_store(sum + VEC_LINE_32 * 3 + k, v_add(v_sum3, v_sumIn3));

            v_expand(vx_load(stackSp1Ptr + k), x0l, x0h);
            v_expand(v_reinterpret_as_s16(x0l), v_ss0, v_ss1);
            v_expand(v_reinterpret_as_s16(x0h), v_ss2, v_ss3);

            v_sumOut0 = v_add(v_sumOut0, v_ss0);
            v_sumOut1 = v_add(v_sumOut1, v_ss1);
            v_sumOut2 = v_add(v_sumOut2, v_ss2);
            v_sumOut3 = v_add(v_sumOut3, v_ss3);

            v_store(sumOut + k, v_sumOut0);
            v_store(sumOut + VEC_LINE_32 + k, v_sumOut1);
            v_store(sumOut + VEC_LINE_32 * 2 + k, v_sumOut2);
            v_store(sumOut + VEC_LINE_32 * 3 + k, v_sumOut3);

            v_sumIn0 = v_sub(v_sumIn0, v_ss0);
            v_sumIn1 = v_sub(v_sumIn1, v_ss1);
            v_sumIn2 = v_sub(v_sumIn2, v_ss2);
            v_sumIn3 = v_sub(v_sumIn3, v_ss3);

            v_store(sumIn + k, v_sumIn0);
            v_store(sumIn + VEC_LINE_32 + k, v_sumIn1);
            v_store(sumIn + VEC_LINE_32 * 2 + k, v_sumIn2);
            v_store(sumIn + VEC_LINE_32 * 3 + k, v_sumIn3);
        }
    }
    return k;
}

template<>
inline int opColumn<short, int>(const short* srcPtr, short* dstPtr, short* stack, int* sum, int* sumIn,
                                int* sumOut, const float , const int mulValTab, const int shrValTab,
                                const int widthLen, const int ss, const int sp1)
{
    int k = 0;
    if (mulValTab != 0 && shrValTab != 0)
    {
        const int VEC_LINE_16 = VTraits<v_int16>::vlanes();
        const int VEC_LINE_32 = VTraits<v_int32>::vlanes();
        v_int32 v_mulVal = vx_setall_s32(mulValTab);

        auto stackStartPtr = stack + ss * widthLen;
        auto stackSp1Ptr = stack + sp1 * widthLen;
        for (;k <= widthLen - VEC_LINE_16; k += VEC_LINE_16)
        {
            v_int32 v_sum0, v_sum1;
            v_int32 v_sumIn0, v_sumIn1;
            v_int32 v_sumOut0, v_sumOut1;

            v_sum0 = vx_load(sum + k);
            v_sum1 = vx_load(sum + k + VEC_LINE_32);

            v_sumIn0 = vx_load(sumIn + k);
            v_sumIn1 = vx_load(sumIn + k + VEC_LINE_32);

            v_sumOut0 = vx_load(sumOut + k);
            v_sumOut1 = vx_load(sumOut + k + VEC_LINE_32);

            v_store(dstPtr + k,v_pack(v_shr(v_mul(v_sum0, v_mulVal), shrValTab), v_shr(v_mul(v_sum1, v_mulVal), shrValTab)));

            v_sum0 = v_sub(v_sum0, v_sumOut0);
            v_sum1 = v_sub(v_sum1, v_sumOut1);

            v_int32 v_ss0, v_ss1;
            v_expand(vx_load(stackStartPtr + k), v_ss0, v_ss1);

            v_sumOut0 = v_sub(v_sumOut0, v_ss0);
            v_sumOut1 = v_sub(v_sumOut1, v_ss1);

            v_expand(vx_load(srcPtr + k), v_ss0, v_ss1);
            memcpy(stackStartPtr + k,srcPtr + k, VEC_LINE_16 * sizeof (short));

            v_sumIn0 = v_add(v_sumIn0, v_ss0);
            v_sumIn1 = v_add(v_sumIn1, v_ss1);

            v_sum0 = v_add(v_sum0, v_sumIn0);
            v_sum1 = v_add(v_sum1, v_sumIn1);

            v_store(sum + k, v_sum0);
            v_store(sum + VEC_LINE_32 + k, v_sum1);

            v_expand(vx_load(stackSp1Ptr + k), v_ss0, v_ss1);

            v_sumOut0 = v_add(v_sumOut0, v_ss0);
            v_sumOut1 = v_add(v_sumOut1, v_ss1);

            v_store(sumOut + k, v_sumOut0);
            v_store(sumOut + VEC_LINE_32 + k, v_sumOut1);

            v_sumIn0 = v_sub(v_sumIn0, v_ss0);
            v_sumIn1 = v_sub(v_sumIn1, v_ss1);

            v_store(sumIn + k, v_sumIn0);
            v_store(sumIn + VEC_LINE_32 + k, v_sumIn1);
        }
    }
    return k;
}

template<>
inline int opColumn<ushort, int>(const ushort* srcPtr, ushort* dstPtr, ushort* stack, int* sum, int* sumIn,
                                int* sumOut, const float , const int mulValTab, const int shrValTab,
                                const int widthLen, const int ss, const int sp1)
{
    int k = 0;
    if (mulValTab != 0 && shrValTab != 0)
    {
        const int VEC_LINE_16 = VTraits<v_uint16>::vlanes();
        const int VEC_LINE_32 = VTraits<v_int32>::vlanes();
        v_uint32 v_mulVal = vx_setall_u32((uint32_t)mulValTab);

        auto stackStartPtr = stack + ss * widthLen;
        auto stackSp1Ptr = stack + sp1 * widthLen;
        for (;k <= widthLen - VEC_LINE_16; k += VEC_LINE_16)
        {
            v_int32 v_sum0, v_sum1;
            v_int32 v_sumIn0, v_sumIn1;
            v_int32 v_sumOut0, v_sumOut1;

            v_sum0 = vx_load(sum + k);
            v_sum1 = vx_load(sum + k + VEC_LINE_32);

            v_sumIn0 = vx_load(sumIn + k);
            v_sumIn1 = vx_load(sumIn + k + VEC_LINE_32);

            v_sumOut0 = vx_load(sumOut + k);
            v_sumOut1 = vx_load(sumOut + k + VEC_LINE_32);

            v_store(dstPtr + k, v_pack(v_shr(v_mul(v_reinterpret_as_u32(v_sum0), v_mulVal), shrValTab), v_shr(v_mul(v_reinterpret_as_u32(v_sum1), v_mulVal), shrValTab)));

            v_sum0 = v_sub(v_sum0, v_sumOut0);
            v_sum1 = v_sub(v_sum1, v_sumOut1);

            v_uint32 v_ss0, v_ss1;
            v_expand(vx_load(stackStartPtr + k), v_ss0, v_ss1);

            v_sumOut0 = v_sub(v_sumOut0, v_reinterpret_as_s32(v_ss0));
            v_sumOut1 = v_sub(v_sumOut1, v_reinterpret_as_s32(v_ss1));

            v_expand(vx_load(srcPtr + k), v_ss0, v_ss1);

            memcpy(stackStartPtr + k,srcPtr + k, VEC_LINE_16 * sizeof (ushort));

            v_sumIn0 = v_add(v_sumIn0, v_reinterpret_as_s32(v_ss0));
            v_sumIn1 = v_add(v_sumIn1, v_reinterpret_as_s32(v_ss1));

            v_sum0 = v_add(v_sum0, v_sumIn0);
            v_sum1 = v_add(v_sum1, v_sumIn1);

            v_store(sum + k, v_sum0);
            v_store(sum + VEC_LINE_32 + k, v_sum1);

            v_expand(vx_load(stackSp1Ptr + k), v_ss0, v_ss1);

            v_sumOut0 = v_add(v_sumOut0, v_reinterpret_as_s32(v_ss0));
            v_sumOut1 = v_add(v_sumOut1, v_reinterpret_as_s32(v_ss1));

            v_store(sumOut + k, v_sumOut0);
            v_store(sumOut + VEC_LINE_32 + k, v_sumOut1);

            v_sumIn0 = v_sub(v_sumIn0, v_reinterpret_as_s32(v_ss0));
            v_sumIn1 = v_sub(v_sumIn1, v_reinterpret_as_s32(v_ss1));

            v_store(sumIn + k, v_sumIn0);
            v_store(sumIn + VEC_LINE_32 + k, v_sumIn1);
        }
    }
    return k;
}
#endif

template<typename T, typename TBuf>
class ParallelStackBlurColumn:
        public ParallelLoopBody
{
public:
    ParallelStackBlurColumn (const Mat & _src, Mat &_dst, int _radius):src(_src), dst(_dst) ,radius(_radius)
    {
        CN = src.channels();
        widthElem = CN * src.cols;
        height = src.rows;
        hm = src.rows - 1;
        mulVal = 1.0f / ((radius + 1)*(radius + 1));
        if (radius <= STACKBLUR_MAX_RADIUS)
        {
            shrValTab = stackblurShr[radius];
            mulValTab = stackblurMul[radius];
        }
        else
        {
            shrValTab = 0;
            mulValTab = 0;
        }
    }

    ~ParallelStackBlurColumn() {}

    virtual void operator ()(const Range& range) const CV_OVERRIDE
    {
        if (radius == 0)
            return;

        const int kernelSize = 2 * radius + 1;
        int widthImg = std::min(range.end, src.cols * CN);
        int widthLen = widthImg - range.start;

        size_t bufSize = 3 * widthLen * sizeof(TBuf) + kernelSize * widthLen * sizeof(T);

        AutoBuffer<uchar> _buf(bufSize + 16);
        uchar* bufptr = alignPtr(_buf.data(), 16);

        TBuf* sum = (TBuf *)bufptr;
        TBuf* sumIn = sum + widthLen;
        TBuf* sumOut = sumIn + widthLen;
        T* stack = (T* )(sumOut + widthLen);

        memset(bufptr, 0, bufSize);

        const T* srcPtr =dst.ptr<T>() + range.start;

        for (int i = 0; i <= radius; i++)
        {
            for (int k = 0; k < widthLen; k++)
            {
                stack[i * widthLen + k] = *(srcPtr + k);
                sum[k] += *(srcPtr + k) * (i + 1);
                sumOut[k] += *(srcPtr + k);
            }
        }

        for (int i = 1; i <= radius; i++)
        {
            if (i <= hm) srcPtr += widthElem;
            for (int k = 0; k < widthLen; k++)
            {
                T tmp = *(srcPtr + k);
                stack[(i + radius) * widthLen + k] = tmp;
                sum[k] += tmp * (radius - i + 1);
                sumIn[k] += tmp;
            }
        }

        int sp = radius;
        int yp = radius;

        if (yp > hm) yp = hm;

        T* dstPtr = dst.ptr<T>() + range.start;
        srcPtr = dst.ptr<T>(yp) + range.start;
        int stackStart = 0;

        for(int i = 0; i < height; i++)
        {
            stackStart = sp + kernelSize - radius;
            if (stackStart >= kernelSize) stackStart -= kernelSize;

            int sp1 = sp + 1;
            if (sp1 >= kernelSize)
                sp1 = 0;

            if (yp < hm)
            {
                yp++;
                srcPtr += widthElem;
            }

            int k = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            k = opColumn<T, TBuf>(srcPtr, dstPtr, stack, sum, sumIn, sumOut, mulVal, mulValTab, shrValTab,
                                      widthLen, stackStart, sp1);
#endif

            for (; k < widthLen; k++)
            {
                *(dstPtr + k) = static_cast<T>(sum[k] * mulVal);
                sum[k] -= sumOut[k];
                sumOut[k] -= stack[stackStart * widthLen + k];

                stack[stackStart * widthLen + k] = *(srcPtr + k);
                sumIn[k] += *(srcPtr + k);
                sum[k] += sumIn[k];

                sumOut[k] += stack[sp1 * widthLen + k];
                sumIn[k] -= stack[sp1 * widthLen + k];
            }

            dstPtr += widthElem;
            ++sp;
            if (sp >= kernelSize)
                sp = 0;
        }
    }

private:
    const Mat &src;
    Mat &dst;
    int radius;
    int CN;
    int height;
    int widthElem;
    int hm;
    float mulVal;
    int mulValTab;
    int shrValTab;
};

void stackBlur(InputArray _src, OutputArray _dst, Size ksize)
{
    CV_INSTRUMENT_REGION();
    CV_Assert(!_src.empty());

    CV_Assert( ksize.width  > 0 && ksize.width  % 2 == 1 &&
               ksize.height > 0 && ksize.height % 2 == 1 );

    int radiusH = ksize.height / 2;
    int radiusW = ksize.width / 2;

    int stype = _src.type(), sdepth = _src.depth();
    Mat src = _src.getMat();

    if (ksize.width == 1)
    {
        _src.copyTo(_dst);

        if (ksize.height == 1)
            return;
    }
    else
    {
        _dst.create( src.size(), stype);
    }

    Mat dst = _dst.getMat();
    int numOfThreads = getNumThreads();
    int widthElem = src.cols * src.channels();

    if (dst.rows / numOfThreads < 3)
        numOfThreads = std::max(1, dst.rows / 3);

    if (sdepth == CV_8U)
    {
        if (ksize.width != 1)
            parallel_for_(Range(0, src.rows), ParallelStackBlurRow<uchar, int>(src, dst, radiusW), numOfThreads);
        if (ksize.height != 1)
            parallel_for_(Range(0, widthElem), ParallelStackBlurColumn<uchar, int>(dst, dst, radiusH), numOfThreads);
    }
    else if (sdepth == CV_16S)
    {
        if (ksize.width != 1)
            parallel_for_(Range(0, src.rows), ParallelStackBlurRow<short, int>(src, dst, radiusW), numOfThreads);
        if (ksize.height != 1)
            parallel_for_(Range(0, widthElem), ParallelStackBlurColumn<short, int>(dst, dst, radiusH), numOfThreads);
    }
    else if (sdepth == CV_16U)
    {
        if (ksize.width != 1)
            parallel_for_(Range(0, src.rows), ParallelStackBlurRow<ushort, int>(src, dst, radiusW), numOfThreads);
        if (ksize.height != 1)
            parallel_for_(Range(0, widthElem), ParallelStackBlurColumn<ushort, int>(dst, dst, radiusH), numOfThreads);
    }
    else if (sdepth == CV_32F)
    {
        if (ksize.width != 1)
            parallel_for_(Range(0, src.rows), ParallelStackBlurRow<float, float>(src, dst, radiusW), numOfThreads);
        if (ksize.height != 1)
            parallel_for_(Range(0, widthElem), ParallelStackBlurColumn<float, float>(dst, dst, radiusH), numOfThreads);
    }
    else
        CV_Error(Error::StsNotImplemented,
                   ("Unsupported input format in StackBlur, the supported formats are: CV_8U, CV_16U, CV_16S and CV_32F."));
}
} //namespace
