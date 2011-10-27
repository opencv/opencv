/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING. 
// 
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2009-2010, NVIDIA Corporation, all rights reserved.
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

#ifndef _ncv_color_conversion_hpp_
#define _ncv_color_conversion_hpp_

#include "NCVPixelOperations.hpp"

enum NCVColorSpace
{
    NCVColorSpaceGray,
    NCVColorSpaceRGBA,
};

template<NCVColorSpace CSin, NCVColorSpace CSout, typename Tin, typename Tout> struct __pixColorConv {
static void _pixColorConv(const Tin &pixIn, Tout &pixOut);
};

template<typename Tin, typename Tout> struct __pixColorConv<NCVColorSpaceRGBA, NCVColorSpaceGray, Tin, Tout> {
static void _pixColorConv(const Tin &pixIn, Tout &pixOut)
{
    Ncv32f luma = 0.299f * pixIn.x + 0.587f * pixIn.y + 0.114f * pixIn.z;
    _TDemoteClampNN(luma, pixOut.x);
}};

template<typename Tin, typename Tout> struct __pixColorConv<NCVColorSpaceGray, NCVColorSpaceRGBA, Tin, Tout> {
static void _pixColorConv(const Tin &pixIn, Tout &pixOut)
{
    _TDemoteClampNN(pixIn.x, pixOut.x);
    _TDemoteClampNN(pixIn.x, pixOut.y);
    _TDemoteClampNN(pixIn.x, pixOut.z);
    pixOut.w = 0;
}};

template<NCVColorSpace CSin, NCVColorSpace CSout, typename Tin, typename Tout>
static
NCVStatus _ncvColorConv_host(const NCVMatrix<Tin> &h_imgIn,
                             const NCVMatrix<Tout> &h_imgOut)
{
    ncvAssertReturn(h_imgIn.size() == h_imgOut.size(), NCV_DIMENSIONS_INVALID);
    ncvAssertReturn(h_imgIn.memType() == h_imgOut.memType() &&
                    (h_imgIn.memType() == NCVMemoryTypeHostPinned || h_imgIn.memType() == NCVMemoryTypeNone), NCV_MEM_RESIDENCE_ERROR);
    NCV_SET_SKIP_COND(h_imgIn.memType() == NCVMemoryTypeNone);
    NCV_SKIP_COND_BEGIN

    for (Ncv32u i=0; i<h_imgIn.height(); i++)
    {
        for (Ncv32u j=0; j<h_imgIn.width(); j++)
        {
            __pixColorConv<CSin, CSout, Tin, Tout>::_pixColorConv(h_imgIn.at(j,i), h_imgOut.at(j,i));
        }
    }

    NCV_SKIP_COND_END
    return NCV_SUCCESS;
}

#endif //_ncv_color_conversion_hpp_
