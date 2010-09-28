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

void cv::gpu::erode( const GpuMat&, GpuMat&, const Mat&, Point, int) { throw_nogpu(); }
void cv::gpu::dilate( const GpuMat&, GpuMat&, const Mat&, Point, int) { throw_nogpu(); }
void cv::gpu::morphologyEx( const GpuMat&, GpuMat&, int, const Mat&, Point, int) { throw_nogpu(); }

#else

namespace 
{
    typedef NppStatus (*npp_morf_func)(const Npp8u*, Npp32s, Npp8u*, Npp32s, NppiSize, const Npp8u*, NppiSize, NppiPoint);


    void morphoogy_caller(npp_morf_func func, const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor, int iterations)
    {
        CV_Assert(src.type() == CV_8U || src.type() == CV_8UC4);        
        CV_Assert(kernel.type() == CV_8U && (kernel.cols & 1) != 0 && (kernel.rows & 1) != 0);

        // in NPP for Cuda 3.1 only such anchor is supported.
        CV_Assert(anchor.x == kernel.cols/2 && anchor.y == kernel.rows/2);

        const Mat& cont_krnl = (kernel.isContinuous() ? kernel : kernel.clone()).reshape(1, 1);
        GpuMat gpu_krnl(cont_krnl);
                
        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        NppiSize mask_sz;
        mask_sz.width = kernel.cols;
        mask_sz.height = kernel.rows;

        NppiPoint anc;
        anc.x = anchor.x;
        anc.y = anchor.y;
        
        dst.create(src.size(), src.type());

        for(int i = 0; i < iterations; ++i)
            nppSafeCall( func(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz, gpu_krnl.ptr<Npp8u>(), mask_sz, anc) );
    }
}


void cv::gpu::erode( const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor, int iterations)
{
    static npp_morf_func funcs[] = {0, nppiErode_8u_C1R, 0, 0, nppiErode_8u_C4R };

    morphoogy_caller(funcs[src.channels()], src, dst, kernel, anchor, iterations);    
}

void cv::gpu::dilate( const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor, int iterations)
{
    static npp_morf_func funcs[] = {0, nppiDilate_8u_C1R, 0, 0, nppiDilate_8u_C4R };
    morphoogy_caller(funcs[src.channels()], src, dst, kernel, anchor, iterations);
}

void cv::gpu::morphologyEx( const GpuMat& src, GpuMat& dst, int op, const Mat& kernel, Point anchor, int iterations)
{
    GpuMat temp;
    switch( op )
    {
    case MORPH_ERODE:   erode( src, dst, kernel, anchor, iterations); break;    
    case MORPH_DILATE: dilate( src, dst, kernel, anchor, iterations); break;    
    case MORPH_OPEN:
         erode( src, dst, kernel, anchor, iterations);
        dilate( dst, dst, kernel, anchor, iterations);
        break;
    case CV_MOP_CLOSE:
        dilate( src, dst, kernel, anchor, iterations);
         erode( dst, dst, kernel, anchor, iterations);
        break;
    case CV_MOP_GRADIENT:
         erode( src, temp, kernel, anchor, iterations);
        dilate( src, dst, kernel, anchor, iterations);        
        subtract(dst, temp, dst);
        break;
    case CV_MOP_TOPHAT:
        if( src.data != dst.data )
            temp = dst;
        erode( src, temp, kernel, anchor, iterations);
        dilate( temp, temp, kernel, anchor, iterations);        
        subtract(src, temp, dst);
        break;
    case CV_MOP_BLACKHAT:
        if( src.data != dst.data )
            temp = dst;
        dilate( src, temp, kernel, anchor, iterations);
        erode( temp, temp, kernel, anchor, iterations);
        subtract(temp, src, dst);
        break;
    default:
        CV_Error( CV_StsBadArg, "unknown morphological operation" );
    }
}

#endif
