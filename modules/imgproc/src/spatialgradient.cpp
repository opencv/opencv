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
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.
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
#include "opencv2/core/hal/intrin.hpp"

#include "spatialgradient.simd.hpp"
#include "spatialgradient.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL based on CMakeLists.txt content

namespace cv
{

// Baseline dispatchers: pick the best CPU variant of each fused kernel at runtime.
static void dispatchSep3x3_16s(const uchar* src, size_t src_step, int srcRows, int srcCols,
                               int rowStart, int rowEnd,
                               short* dx, short* dy, size_t dst_step, int borderType)
{
    CV_CPU_DISPATCH(spatialGradientSep3x3_16s,
        (src, src_step, srcRows, srcCols, rowStart, rowEnd, dx, dy, dst_step, borderType),
        CV_CPU_DISPATCH_MODES_ALL);
}

static void dispatchSep5x5_16s(const uchar* src, size_t src_step, int srcRows, int srcCols,
                               int rowStart, int rowEnd,
                               short* dx, short* dy, size_t dst_step, int borderType)
{
    CV_CPU_DISPATCH(spatialGradientSep5x5_16s,
        (src, src_step, srcRows, srcCols, rowStart, rowEnd, dx, dy, dst_step, borderType),
        CV_CPU_DISPATCH_MODES_ALL);
}

static void dispatchSep3x3_32f(const uchar* src, size_t src_step, int srcRows, int srcCols,
                               int rowStart, int rowEnd,
                               float* dx, float* dy, size_t dst_step, float scale, int borderType)
{
    CV_CPU_DISPATCH(spatialGradientSep3x3_32f,
        (src, src_step, srcRows, srcCols, rowStart, rowEnd, dx, dy, dst_step, scale, borderType),
        CV_CPU_DISPATCH_MODES_ALL);
}

static void dispatchSep5x5_32f(const uchar* src, size_t src_step, int srcRows, int srcCols,
                               int rowStart, int rowEnd,
                               float* dx, float* dy, size_t dst_step, float scale, int borderType)
{
    CV_CPU_DISPATCH(spatialGradientSep5x5_32f,
        (src, src_step, srcRows, srcCols, rowStart, rowEnd, dx, dy, dst_step, scale, borderType),
        CV_CPU_DISPATCH_MODES_ALL);
}

// 3x3 CV_8U -> CV_16S via cv_hal_spatialGradient when a platform HAL provides it
// (e.g. RISC-V RVV). Returns false if the HAL is not implemented so the caller
// falls back to the fused separable kernel.
static bool spatialGradient3x3_tryHal( const Mat& src, OutputArray _dx, OutputArray _dy, int bt )
{
    _dx.create( src.size(), CV_16SC1 );
    _dy.create( src.size(), CV_16SC1 );
    Mat dx = _dx.getMat(), dy = _dy.getMat();

    int res = cv_hal_spatialGradient( src.data, src.step,
                                      dx.ptr<short>(), dx.step,
                                      dy.ptr<short>(), dy.step,
                                      src.cols, src.rows, 3, bt );
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

    Mat src = _src.getMat();
    CV_Assert( !src.empty() );
    CV_Assert( ksize == -1 || ksize == 1 || ksize == 3 || ksize == 5 || ksize == 7 );
    CV_Assert( ddepth == CV_16S || ddepth == CV_32F );

    Size wholeSize;
    Point ofs;
    src.locateROI( wholeSize, ofs );
    const bool entireParent = ( ofs.x == 0 && ofs.y == 0 &&
        src.cols == wholeSize.width && src.rows == wholeSize.height );
    const bool isolated = ( borderType & BORDER_ISOLATED ) != 0;
    const int  bt = borderType & ~BORDER_ISOLATED;

    // (1) Optional platform HAL for whole-image 3x3 CV_8U->CV_16S at unit scale.
    const bool halEligible = ( ksize == 3 && ddepth == CV_16S && scale == 1.0
        && src.type() == CV_8UC1 && entireParent && !isolated
        && ( bt == BORDER_REFLECT_101 || bt == BORDER_REPLICATE ) );
    if ( halEligible && spatialGradient3x3_tryHal( src, _dx, _dy, bt ) )
        return;

    // (2) Fused separable single-pass path (CV_8U source): 3x3/5x5, CV_16S (unit scale) and
    //     CV_32F (any scale), reflect/replicate borders and full-width row-range ROIs. Each
    //     source sample is read once and shared between dx and dy; output is bit-identical to
    //     two cv::Sobel() passes. CV_16S fuses only at unit scale (the kernel emits the
    //     unscaled int16 result); a scaled int16 request falls back to keep cv::Sobel rounding.
    const bool fusableBorder = ( bt == BORDER_REPLICATE || bt == BORDER_REFLECT
                                 || bt == BORDER_REFLECT_101 );
    const bool fullWidthRoi  = ( ofs.x == 0 && src.cols == wholeSize.width );
    const bool scaleOk       = ( ddepth == CV_32F ) || ( scale == 1.0 );

    if ( (ksize == 3 || ksize == 5) && src.type() == CV_8UC1 && fusableBorder && fullWidthRoi && !isolated && scaleOk )
    {
        _dx.create( src.size(), CV_MAKETYPE(ddepth, 1) );
        _dy.create( src.size(), CV_MAKETYPE(ddepth, 1) );
        Mat dx = _dx.getMat(), dy = _dy.getMat();

        const uchar* base = src.ptr<uchar>(0) - (size_t)ofs.y * src.step;
        const int parentRows = wholeSize.height;
        const int rowStart = ofs.y;
        const int rowEnd   = ofs.y + src.rows;

        if ( ddepth == CV_32F )
        {
            const float fscale = (float)scale;
            if ( ksize == 3 )
                dispatchSep3x3_32f( base, src.step, parentRows, src.cols, rowStart, rowEnd,
                                    dx.ptr<float>(), dy.ptr<float>(), dx.step1(), fscale, bt );
            else
                dispatchSep5x5_32f( base, src.step, parentRows, src.cols, rowStart, rowEnd,
                                    dx.ptr<float>(), dy.ptr<float>(), dx.step1(), fscale, bt );
        }
        else
        {
            if ( ksize == 3 )
                dispatchSep3x3_16s( base, src.step, parentRows, src.cols, rowStart, rowEnd,
                                    dx.ptr<short>(), dy.ptr<short>(), dx.step1(), bt );
            else
                dispatchSep5x5_16s( base, src.step, parentRows, src.cols, rowStart, rowEnd,
                                    dx.ptr<short>(), dy.ptr<short>(), dx.step1(), bt );
        }
        return;
    }

    // (3) General fallback: two separable cv::Sobel passes.
    Sobel( _src, _dx, ddepth, 1, 0, ksize, scale, 0, borderType );
    Sobel( _src, _dy, ddepth, 0, 1, ksize, scale, 0, borderType );
}

}
