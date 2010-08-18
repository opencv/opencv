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

using namespace cv;
using namespace cv::gpu;

#if !defined (HAVE_CUDA)

void cv::gpu::remap(const GpuMat& /*src*/, const GpuMat& /*xmap*/, const GpuMat& /*ymap*/, GpuMat& /*dst*/) { throw_nogpu(); }
void cv::gpu::meanShiftFiltering_GPU(const GpuMat&, GpuMat&, int, int, TermCriteria ) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu 
{ 
    namespace impl 
    {
        extern "C" void remap_gpu(const DevMem2D& src, const DevMem2D_<float>& xmap, const DevMem2D_<float>& ymap, DevMem2D dst);

        extern "C" void meanShiftFiltering_gpu(const DevMem2D& src, DevMem2D dst, int sp, int sr, int maxIter, float eps);
    }
}}

void cv::gpu::remap(const GpuMat& src, const GpuMat& xmap, const GpuMat& ymap, GpuMat& dst)
{ 
    CV_DbgAssert(xmap.data && xmap.cols == ymap.cols && xmap.rows == ymap.rows);
    CV_Assert(xmap.type() == CV_32F && ymap.type() == CV_32F);

    dst.create(xmap.size(), src.type());
    CV_Assert(dst.data != src.data);   
    
    impl::remap_gpu(src, xmap, ymap, dst);
}



void cv::gpu::meanShiftFiltering_GPU(const GpuMat& src, GpuMat& dst, int sp, int sr, TermCriteria criteria)
{       
    if( src.empty() )
        CV_Error( CV_StsBadArg, "The input image is empty" );

    if( src.depth() != CV_8U || src.channels() != 4 )
        CV_Error( CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported" );

    dst.create( src.size(), CV_8UC3 );
    
    float eps;
    if( !(criteria.type & TermCriteria::MAX_ITER) )
        criteria.maxCount = 5;
    
    int maxIter = std::min(std::max(criteria.maxCount, 1), 100);
    
    if( !(criteria.type & TermCriteria::EPS) )
        eps = 1.f;

    eps = (float)std::max(criteria.epsilon, 0.0);        
    impl::meanShiftFiltering_gpu(src, dst, sp, sr, maxIter, eps);    
}


#endif /* !defined (HAVE_CUDA) */