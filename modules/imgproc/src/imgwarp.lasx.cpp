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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

/* ////////////////////////////////////////////////////////////////////
//
//  Geometrical transforms on images and matrices: rotation, zoom etc.
//
// */

#include "precomp.hpp"
#include "imgwarp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv
{
namespace opt_LASX
{

int warpAffineBlockline(int *adelta, int *bdelta, short* xy, short* alpha, int X0, int Y0, int bw)
{
    const int AB_BITS = MAX(10, (int)INTER_BITS);
    int x1 = 0;
    __m256i fxy_mask = _v256_setall_w(INTER_TAB_SIZE - 1);
    __m256i XX = _v256_setall_w(X0), YY = _v256_setall_w(Y0);
    for (; x1 <= bw - 16; x1 += 16)
    {
        __m256i tx0, tx1, ty0, ty1;
        tx0 = __lasx_xvadd_w(__lasx_xvld((const __m256i*)(adelta + x1), 0), XX);
        ty0 = __lasx_xvadd_w(__lasx_xvld((const __m256i*)(bdelta + x1), 0), YY);
        tx1 = __lasx_xvadd_w(__lasx_xvld((const __m256i*)(adelta + x1), 8*4), XX);
        ty1 = __lasx_xvadd_w(__lasx_xvld((const __m256i*)(bdelta + x1), 8*4), YY);

        tx0 = __lasx_xvsrai_w(tx0, AB_BITS - INTER_BITS);
        ty0 = __lasx_xvsrai_w(ty0, AB_BITS - INTER_BITS);
        tx1 = __lasx_xvsrai_w(tx1, AB_BITS - INTER_BITS);
        ty1 = __lasx_xvsrai_w(ty1, AB_BITS - INTER_BITS);

        __m256i fx_ = _lasx_packs_w(__lasx_xvand_v(tx0, fxy_mask),
            __lasx_xvand_v(tx1, fxy_mask));
        __m256i fy_ = _lasx_packs_w(__lasx_xvand_v(ty0, fxy_mask),
            __lasx_xvand_v(ty1, fxy_mask));
        tx0 = _lasx_packs_w(__lasx_xvsrai_w(tx0, INTER_BITS),
            __lasx_xvsrai_w(tx1, INTER_BITS));
        ty0 = _lasx_packs_w(__lasx_xvsrai_w(ty0, INTER_BITS),
            __lasx_xvsrai_w(ty1, INTER_BITS));
        fx_ = __lasx_xvsadd_h(fx_, __lasx_xvslli_h(fy_, INTER_BITS));
        fx_ = __lasx_xvpermi_d(fx_, (3 << 6) + (1 << 4) + (2 << 2) + 0);

        __lasx_xvst(__lasx_xvilvl_h(ty0, tx0), (__m256i*)(xy + x1 * 2), 0);
        __lasx_xvst(__lasx_xvilvh_h(ty0, tx0), (__m256i*)(xy + x1 * 2), 16*2);
        __lasx_xvst(fx_, (__m256i*)(alpha + x1), 0);
    }
    return x1;
}

}
}
/* End of file. */
