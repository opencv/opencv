/*
 * By downloading, copying, installing or using the software you agree to this license.
 * If you do not agree to this license, do not download, install,
 * copy or use the software.
 *
 *
 *                           License Agreement
 *                For Open Source Computer Vision Library
 *                        (3-clause BSD License)
 *
 * Copyright (C) 2012-2015, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the names of the copyright holders nor the names of the contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * This software is provided by the copyright holders and contributors "as is" and
 * any express or implied warranties, including, but not limited to, the implied
 * warranties of merchantability and fitness for a particular purpose are disclaimed.
 * In no event shall copyright holders or contributors be liable for any direct,
 * indirect, incidental, special, exemplary, or consequential damages
 * (including, but not limited to, procurement of substitute goods or services;
 * loss of use, data, or profits; or business interruption) however caused
 * and on any theory of liability, whether in contract, strict liability,
 * or tort (including negligence or otherwise) arising in any way out of
 * the use of this software, even if advised of the possibility of such damage.
 */

#include <vector>

#include "common.hpp"

namespace CAROTENE_NS {

bool isSobel3x3Supported(const Size2D &size, BORDER_MODE border,
                         s32 dx, s32 dy, Margin borderMargin)
{
    return dx < 3 && dx >= 0 &&
           dy < 3 && dy >= 0 &&
           (dx + dy) > 0 &&
           isSeparableFilter3x3Supported(size, border, dx, dy, borderMargin);
}

void Sobel3x3(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              s16 * dstBase, ptrdiff_t dstStride,
              s32 dx, s32 dy,
              BORDER_MODE borderType, u8 borderValue, Margin borderMargin)
{
    internal::assertSupportedConfiguration(isSobel3x3Supported(size, borderType, dx, dy, borderMargin));
#ifdef CAROTENE_NEON
    SeparableFilter3x3(size, srcBase, srcStride, dstBase, dstStride,
                       dx, dy, 0, 0,
                       borderType, borderValue, borderMargin);
#else
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)borderValue;
#endif
}

bool isSobel3x3f32Supported(const Size2D &size, BORDER_MODE border,
                            s32 dx, s32 dy)
{
    return isSupportedConfiguration() &&
           dx < 3 && dx >= 0 &&
           dy < 3 && dy >= 0 &&
           (dx + dy) > 0 &&
           size.width >= 4 && size.height >= 2 &&
           (border == BORDER_MODE_CONSTANT   ||
            border == BORDER_MODE_REFLECT    ||
            border == BORDER_MODE_REFLECT101 ||
            border == BORDER_MODE_REPLICATE   );
}

void Sobel3x3(const Size2D &size,
              const f32 * srcBase, ptrdiff_t srcStride,
              f32 * dstBase, ptrdiff_t dstStride,
              s32 dx, s32 dy,
              BORDER_MODE borderType, f32 borderValue)
{
    internal::assertSupportedConfiguration(isSobel3x3f32Supported(size, borderType, dx, dy));
#ifdef CAROTENE_NEON
    std::vector<f32> _tmp;
    f32 *tmp = 0;
    if (borderType == BORDER_MODE_CONSTANT)
    {
        _tmp.assign(size.width + 2, borderValue);
        tmp = &_tmp[1];
    }

    ptrdiff_t delta = (ptrdiff_t)((size.width + 2 + 31) & -32);//align size
    std::vector<f32> _tempBuf((delta << 1) + 64);
    f32 *trow0 = internal::alignPtr(&_tempBuf[1], 32), *trow1 = internal::alignPtr(trow0 + delta, 32);

    for( size_t y = 0; y < size.height; y++ )
    {
        const f32* srow0;
        const f32* srow1 = internal::getRowPtr(srcBase, srcStride, y);
        const f32* srow2;
        f32* drow = internal::getRowPtr(dstBase, dstStride, y > 0 ? y-1 : 0);
        f32* drow1 = internal::getRowPtr(dstBase, dstStride, y);
        if (borderType == BORDER_MODE_REFLECT101) {
            srow0 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 1);
            srow2 = internal::getRowPtr(srcBase, srcStride, y < size.height-1 ? y+1 : size.height-2);
        } else  if (borderType == BORDER_MODE_CONSTANT) {
            srow0 = y > 0 ? internal::getRowPtr(srcBase, srcStride, y-1) : tmp;
            srow2 =  y < size.height-1 ? internal::getRowPtr(srcBase, srcStride, y+1) : tmp;
        } else { // BORDER_MODE_REFLECT || BORDER_MODE_REPLICATE
            srow0 = internal::getRowPtr(srcBase, srcStride, y > 0 ? y-1 : 0);
            srow2 = internal::getRowPtr(srcBase, srcStride, y < size.height-1 ? y+1 : size.height-1);
        }

        float32x4_t tprev = vmovq_n_f32(0.f);
        float32x4_t tcurr = vmovq_n_f32(0.f);
        float32x4_t tnext = vmovq_n_f32(0.f);
        float32x4_t t0, t1, t2;
        // do vertical convolution
        size_t x = 0, bcolsn = y + 2 < size.height ? size.width : (size.width - 4);
        for( ; x <= bcolsn; x += 4 )
        {
            internal::prefetch(srow0 + x);
            internal::prefetch(srow1 + x);
            internal::prefetch(srow2 + x);

            float32x4_t x0 = vld1q_f32(srow0 + x);
            float32x4_t x1 = vld1q_f32(srow1 + x);
            float32x4_t x2 = vld1q_f32(srow2 + x);

            tprev = tcurr;
            tcurr = tnext;
            if(!dy)
            {
                tnext = vaddq_f32(vaddq_f32(vaddq_f32(x1, x1), x2), x0);
            }
            else if(dy == 2)
            {
                tnext = vsubq_f32(vsubq_f32(x2, x1), vsubq_f32(x1, x0));
            }
            else
            {
                tnext = vsubq_f32(x2, x0);
            }

            if(!x) {
                tcurr = tnext;
                // make border
                if (borderType == BORDER_MODE_CONSTANT)
                {
                    tcurr = vsetq_lane_f32(borderValue,tcurr, 3);
                }
                else if (borderType == BORDER_MODE_REFLECT101)
                {
                    tcurr = vsetq_lane_f32(vgetq_lane_f32(tcurr, 1),tcurr, 3);
                }
                else // BORDER_MODE_REFLECT || BORDER_MODE_REPLICATE
                {
                    tcurr = vsetq_lane_f32(vgetq_lane_f32(tcurr, 0),tcurr, 3);
                }
                continue;
            }

            internal::prefetch(trow0 + x);
            internal::prefetch(trow1 + x);

            t0 = vextq_f32(tprev, tcurr, 3);
            t1 = tcurr;
            t2 = vextq_f32(tcurr, tnext, 1);
            if(!dx)
            {
                t0 = vaddq_f32(t0, vaddq_f32(vaddq_f32(t1, t1), t2));
            }
            else if(dx == 2)
            {
                t0 = vsubq_f32(vsubq_f32(t2, t1), vsubq_f32(t1, t0));
            }
            else
            {
                t0 = vsubq_f32(t2, t0);
            }

            if(!(y%2))
            {
                vst1q_f32(trow0 + x - 4, t0);
            }
            else
            {
                vst1q_f32(trow1 + x - 4, t0);
            }
        }
        x -= 4;
        if(x == size.width){
            x--;
        }
        f32 prevx = 0, rowx = 0, nextx = 0;
        if(!dy)
        {
            prevx = x > 0 ? srow2[x-1] + 2*srow1[x-1] + srow0[x-1] :
                    (borderType == BORDER_MODE_REFLECT101 ? srow2[1] + 2*srow1[1] + srow0[1] :
                    (borderType == BORDER_MODE_CONSTANT   ? 4*borderValue :
                                                            srow2[0] + 2*srow1[0] + srow0[0]) );
            rowx  = srow2[x] + 2*srow1[x] + srow0[x];
        }
        else if(dy == 2)
        {
            prevx = x > 0 ? srow2[x-1] - 2*srow1[x-1] + srow0[x-1] :
                    (borderType == BORDER_MODE_REFLECT101 ? srow2[1] - 2*srow1[1] + srow0[1] :
                    (borderType == BORDER_MODE_CONSTANT   ? 0.f :
                                                            srow2[0] - 2*srow1[0] + srow0[0]) );
            rowx  = srow2[x] - 2*srow1[x] + srow0[x];
        }
        else
        {
            prevx = x > 0 ? srow2[x-1] - srow0[x-1] :
                    (borderType == BORDER_MODE_REFLECT101 ? srow2[1] - srow0[1] :
                    (borderType == BORDER_MODE_CONSTANT   ? 0.f :
                                                            srow2[0] - srow0[0]) );
            rowx  = srow2[x] - srow0[x];
        }

        for( ; x < size.width; x++ )
        {
            if(x+1 == size.width) {
                // make border
                if (borderType == BORDER_MODE_CONSTANT)
                {
                    if(!dy) {
                        nextx = 4*borderValue;
                    } else {
                        nextx = 0.f;
                    }
                } else if (borderType == BORDER_MODE_REFLECT101)
                {
                    if(!dy) {
                        nextx = srow2[x-1] + 2*srow1[x-1] + srow0[x-1];
                    } else if(dy == 2) {
                        nextx = srow2[x-1] - 2*srow1[x-1] + srow0[x-1];
                    } else {
                        nextx = srow2[x-1] - srow0[x-1];
                    }
                } else {
                    if(!dy) {
                        nextx = srow2[x] + 2*srow1[x] + srow0[x];
                    } else if(dy == 2) {
                        nextx = srow2[x] - 2*srow1[x] + srow0[x];
                    } else {
                        nextx = srow2[x] - srow0[x];
                    }
                }
            } else {
                if(!dy) {
                    nextx = srow2[x+1] + 2*srow1[x+1] + srow0[x+1];
                } else if(dy == 2) {
                    nextx = srow2[x+1] - 2*srow1[x+1] + srow0[x+1];
                } else {
                    nextx = srow2[x+1] - srow0[x+1];
                }
            }
            f32 res;
            if(dx==1) {
                res = nextx - prevx;
            } else if(!dx) {
                res = prevx + 2*rowx + nextx;
            } else {
                res = prevx - 2*rowx + nextx;
            }
            if(!(y%2)) {
                *(trow0+x) = res;
            } else {
                *(trow1+x) = res;
            }
            prevx = rowx;
            rowx = nextx;
        }

        if(y>0) {
            for(size_t x1 = 0; x1 < size.width; x1++ )
            {
                if(y%2)
                    *(drow + x1) = trow0[x1];
                else
                    *(drow + x1) = trow1[x1];
            }
        }
        if(y == size.height-1) {
            for(size_t x1 = 0; x1 < size.width; x1++ )
            {
                if(!(y%2))
                    *(drow1 + x1) = trow0[x1];
                else
                    *(drow1 + x1) = trow1[x1];
            }
        }
    }
#else
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)borderValue;
#endif
}

} // namespace CAROTENE_NS
