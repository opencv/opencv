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
 * Copyright (C) 2014-2015, NVIDIA Corporation, all rights reserved.
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

#include "common.hpp"

#include "separable_filter.hpp"

namespace CAROTENE_NS {

bool isSeparableFilter3x3Supported(const Size2D &size, BORDER_MODE border, s32 dx, s32 dy, Margin borderMargin)
{
    return isSupportedConfiguration() &&
        size.width >= 9 && size.height >= 1 &&
        (size.height + borderMargin.top + borderMargin.bottom) >= 2  &&
        (dx >= 0) && (dx < 4) && (dy >= 0) && (dy < 4) &&
        (border == BORDER_MODE_CONSTANT   ||
         border == BORDER_MODE_REFLECT    ||
         border == BORDER_MODE_REFLECT101 ||
         border == BORDER_MODE_REPLICATE   );
}

void SeparableFilter3x3(const Size2D &size,
                        const u8 * srcBase, ptrdiff_t srcStride,
                        s16 * dstBase, ptrdiff_t dstStride,
                        const u8 rowFilter, const u8 colFilter, const s16 *xw, const s16 *yw,
                        BORDER_MODE border, u8 borderValue, Margin borderMargin)
{
    internal::assertSupportedConfiguration(isSeparableFilter3x3Supported(size, border, rowFilter, colFilter, borderMargin));
#ifdef CAROTENE_NEON
    if(!((xw || rowFilter < 3) && (yw || colFilter < 3)))
        std::abort();//Couldn't call generic filter without provided weights

    typedef void (*sepFilter3x3_8u16s_func)(const Size2D&, const u8*, ptrdiff_t, s16*, ptrdiff_t,
                                            const s16*, const s16*, BORDER_MODE, u8, Margin);

    static sepFilter3x3_8u16s_func quickFilters[4][4]=
    {
    /*d0y*/{ /*d0x*/ internal::sepFilter3x3<internal::RowFilter3x3S16_121,    internal::ColFilter3x3S16_121>::process,
             /*dx*/  internal::sepFilter3x3<internal::RowFilter3x3S16_m101,   internal::ColFilter3x3S16_121>::process,
             /*d2x*/ internal::sepFilter3x3<internal::RowFilter3x3S16_1m21,   internal::ColFilter3x3S16_121>::process,
             /*dNx*/ internal::sepFilter3x3<internal::RowFilter3x3S16Generic, internal::ColFilter3x3S16_121>::process},

    /*dy */{ /*d0x*/ internal::sepFilter3x3<internal::RowFilter3x3S16_121,    internal::ColFilter3x3S16_m101>::process,
             /*dx*/  internal::sepFilter3x3<internal::RowFilter3x3S16_m101,   internal::ColFilter3x3S16_m101>::process,
             /*d2x*/ internal::sepFilter3x3<internal::RowFilter3x3S16_1m21,   internal::ColFilter3x3S16_m101>::process,
             /*dNx*/ internal::sepFilter3x3<internal::RowFilter3x3S16Generic, internal::ColFilter3x3S16_m101>::process},

    /*d2y*/{ /*d0x*/ internal::sepFilter3x3<internal::RowFilter3x3S16_121,    internal::ColFilter3x3S16_1m21>::process,
             /*dx*/  internal::sepFilter3x3<internal::RowFilter3x3S16_m101,   internal::ColFilter3x3S16_1m21>::process,
             /*d2x*/ internal::sepFilter3x3<internal::RowFilter3x3S16_1m21,   internal::ColFilter3x3S16_1m21>::process,
             /*dNx*/ internal::sepFilter3x3<internal::RowFilter3x3S16Generic, internal::ColFilter3x3S16_1m21>::process},

    /*dNy*/{ /*d0x*/ internal::sepFilter3x3<internal::RowFilter3x3S16_121,    internal::ColFilter3x3S16Generic>::process,
             /*dx*/  internal::sepFilter3x3<internal::RowFilter3x3S16_m101,   internal::ColFilter3x3S16Generic>::process,
             /*d2x*/ internal::sepFilter3x3<internal::RowFilter3x3S16_1m21,   internal::ColFilter3x3S16Generic>::process,
             /*dNx*/ internal::sepFilter3x3<internal::RowFilter3x3S16Generic, internal::ColFilter3x3S16Generic>::process}
    };

    quickFilters[colFilter][rowFilter](size, srcBase, srcStride, dstBase, dstStride,
                                       xw, yw, border, borderValue, borderMargin);
#else
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)xw;
    (void)yw;
    (void)borderValue;
#endif
}


} // namespace CAROTENE_NS
