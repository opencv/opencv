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
// Copyright (C) 2000-2008, 2018, Intel Corporation, all rights reserved.
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

#include "precomp.hpp"

#include <vector>

#include "opencv2/core/hal/intrin.hpp"
#include "opencl_kernels_imgproc.hpp"

#include "median_blur.simd.hpp"
#include "median_blur.simd_declarations.hpp"

namespace cv {

#ifdef HAVE_OPENCL

#define DIVUP(total, grain) ((total + grain - 1) / (grain))

static bool ocl_medianFilter(InputArray _src, OutputArray _dst, int m)
{
    size_t localsize[2] = { 16, 16 };
    size_t globalsize[2];
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    if ( !((depth == CV_8U || depth == CV_16U || depth == CV_16S || depth == CV_32F) && cn <= 4 && (m == 3 || m == 5)) )
        return false;

    Size imgSize = _src.size();
    bool useOptimized = (1 == cn) &&
                        (size_t)imgSize.width >= localsize[0] * 8  &&
                        (size_t)imgSize.height >= localsize[1] * 8 &&
                        imgSize.width % 4 == 0 &&
                        imgSize.height % 4 == 0 &&
                        (ocl::Device::getDefault().isIntel());

    cv::String kname = format( useOptimized ? "medianFilter%d_u" : "medianFilter%d", m) ;
    cv::String kdefs = useOptimized ?
                         format("-D T=%s -D T1=%s -D T4=%s%d -D cn=%d -D USE_4OPT", ocl::typeToStr(type),
                         ocl::typeToStr(depth), ocl::typeToStr(depth), cn*4, cn)
                         :
                         format("-D T=%s -D T1=%s -D cn=%d", ocl::typeToStr(type), ocl::typeToStr(depth), cn) ;

    ocl::Kernel k(kname.c_str(), ocl::imgproc::medianFilter_oclsrc, kdefs.c_str() );

    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(src.size(), type);
    UMat dst = _dst.getUMat();

    k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst));

    if( useOptimized )
    {
        globalsize[0] = DIVUP(src.cols / 4, localsize[0]) * localsize[0];
        globalsize[1] = DIVUP(src.rows / 4, localsize[1]) * localsize[1];
    }
    else
    {
        globalsize[0] = (src.cols + localsize[0] + 2) / localsize[0] * localsize[0];
        globalsize[1] = (src.rows + localsize[1] - 1) / localsize[1] * localsize[1];
    }

    return k.run(2, globalsize, localsize, false);
}

#undef DIVUP

#endif


void medianBlur( InputArray _src0, OutputArray _dst, int ksize )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!_src0.empty());
    CV_Assert( (ksize % 2 == 1) && (_src0.dims() <= 2 ));

    if( ksize <= 1 || _src0.empty() )
    {
        _src0.copyTo(_dst);
        return;
    }

    CV_OCL_RUN(_dst.isUMat(),
               ocl_medianFilter(_src0,_dst, ksize))

    Mat src0 = _src0.getMat();

    // ----------- SAFETY CHECK (ADDED FIX) ----------------
    const int64 pixels = (int64)src0.rows * (int64)src0.cols;
    if (pixels > (int64)2e9)
    {
        CV_Error(cv::Error::StsOutOfRange,
                 "medianBlur: image too large, may cause internal overflow. "
                 "Please process the image in tiles.");
    }
    // -----------------------------------------------------

    _dst.create( src0.size(), src0.type() );
    Mat dst = _dst.getMat();

    CALL_HAL(medianBlur, cv_hal_medianBlur, src0.data, src0.step, dst.data, dst.step,
             src0.cols, src0.rows, src0.depth(), src0.channels(), ksize);

    CV_CPU_DISPATCH(medianBlur, (src0, dst, ksize),
        CV_CPU_DISPATCH_MODES_ALL);
}

}  // namespace

/* End of file. */
