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

#include "precomp.hpp"
#include "deriv.hpp"

namespace cv
{

// 3x3 CV_8U -> CV_16S via cv_hal_spatialGradient when a platform HAL provides it
// (e.g. RISC-V RVV). Returns false if the HAL is not implemented so the caller
// can fall back to Sobel2D_impl.
static bool spatialGradient3x3_tryHal( InputArray _src, OutputArray _dx, OutputArray _dy,
                                       int borderType )
{
    Mat src = _src.getMat();
    CV_Assert( src.type() == CV_8UC1 );
    CV_Assert( borderType == BORDER_DEFAULT || borderType == BORDER_REPLICATE );

    _dx.create( src.size(), CV_16SC1 );
    _dy.create( src.size(), CV_16SC1 );
    Mat dx = _dx.getMat(), dy = _dy.getMat();

    int res = cv_hal_spatialGradient( src.data, src.step,
                                      dx.ptr<short>(), dx.step,
                                      dy.ptr<short>(), dy.step,
                                      src.cols, src.rows, 3, borderType );
    if ( res == CV_HAL_ERROR_OK )
        return true;
    if ( res != CV_HAL_ERROR_NOT_IMPLEMENTED )
        CV_Error_( cv::Error::StsInternal,
                   ("HAL implementation spatialGradient ==> cv_hal_spatialGradient returned %d (0x%08x)", res, res) );
    return false;
}

void spatialGradient( InputArray _src, OutputArray _dx, OutputArray _dy,
                      int ksize, int borderType, int ddepth, double scale )
{
    CV_INSTRUMENT_REGION();

    CV_Assert( ksize == 3 || ksize == 5 );
    CV_Assert( ddepth == CV_16S || ddepth == CV_32F );

    Mat src = _src.getMat();
    CV_Assert( !src.empty() );

    Size wholeSize;
    Point ofs;
    src.locateROI(wholeSize, ofs);
    const bool entireParent = (ofs.x == 0 && ofs.y == 0 &&
        src.cols == wholeSize.width && src.rows == wholeSize.height);
    const bool isolated = (borderType & BORDER_ISOLATED) != 0;

    const int bt = borderType & ~BORDER_ISOLATED;
    const bool halEligible = (ksize == 3 && ddepth == CV_16S && scale == 1.0
        && src.type() == CV_8UC1 && entireParent && !isolated
        && (bt == BORDER_REFLECT_101 || bt == BORDER_REPLICATE));

    if ( halEligible && spatialGradient3x3_tryHal( _src, _dx, _dy, borderType ) )
        return;

    Sobel2D_impl( _src, _dx, _dy, ksize, ddepth, scale, borderType );
}

}
