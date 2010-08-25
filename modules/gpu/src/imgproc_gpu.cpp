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

void cv::gpu::remap(const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::meanShiftFiltering_GPU(const GpuMat&, GpuMat&, int, int, TermCriteria ) { throw_nogpu(); }
void cv::gpu::drawColorDisp(const GpuMat&, GpuMat&, int) { throw_nogpu(); }
void cv::gpu::drawColorDisp(const GpuMat&, GpuMat&, int, const Stream&) { throw_nogpu(); }
void cv::gpu::reprojectImageTo3D_GPU(const GpuMat&, GpuMat&, const Mat&) { throw_nogpu(); }
void cv::gpu::reprojectImageTo3D_GPU(const GpuMat&, GpuMat&, const Mat&, const Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu 
{ 
    namespace impl 
    {
        extern "C" void remap_gpu(const DevMem2D& src, const DevMem2D_<float>& xmap, const DevMem2D_<float>& ymap, DevMem2D dst);

        extern "C" void meanShiftFiltering_gpu(const DevMem2D& src, DevMem2D dst, int sp, int sr, int maxIter, float eps);

        void drawColorDisp_gpu(const DevMem2D& src, const DevMem2D& dst, int ndisp, const cudaStream_t& stream);
        void drawColorDisp_gpu(const DevMem2D_<short>& src, const DevMem2D& dst, int ndisp, const cudaStream_t& stream);

        void reprojectImageTo3D_gpu(const DevMem2D& disp, const DevMem2Df& xyzw, const float* q, const cudaStream_t& stream);
        void reprojectImageTo3D_gpu(const DevMem2D_<short>& disp, const DevMem2Df& xyzw, const float* q, const cudaStream_t& stream);
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

namespace
{
    template <typename T>
    void drawColorDisp_caller(const GpuMat& src, GpuMat& dst, int ndisp, const cudaStream_t& stream)
    {        
        GpuMat out;
        if (&dst != &src)
            out = dst;
        out.create(src.size(), CV_8UC4);

        impl::drawColorDisp_gpu((DevMem2D_<T>)src, out, ndisp, stream);

        dst = out;
    }

    typedef void (*drawColorDisp_caller_t)(const GpuMat& src, GpuMat& dst, int ndisp, const cudaStream_t& stream);

    const drawColorDisp_caller_t drawColorDisp_callers[] = {drawColorDisp_caller<unsigned char>, 0, 0, drawColorDisp_caller<short>, 0, 0, 0, 0};
}

void cv::gpu::drawColorDisp(const GpuMat& src, GpuMat& dst, int ndisp)
{
    CV_Assert(src.type() == CV_8U || src.type() == CV_16S);
    
    drawColorDisp_callers[src.type()](src, dst, ndisp, 0);
}

void cv::gpu::drawColorDisp(const GpuMat& src, GpuMat& dst, int ndisp, const Stream& stream)
{
    CV_Assert(src.type() == CV_8U || src.type() == CV_16S);
    
    drawColorDisp_callers[src.type()](src, dst, ndisp, StreamAccessor::getStream(stream));
}

namespace
{
    template <typename T>
    void reprojectImageTo3D_caller(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, const cudaStream_t& stream)
    {        
        xyzw.create(disp.rows, disp.cols, CV_32FC4);
        impl::reprojectImageTo3D_gpu((DevMem2D_<T>)disp, xyzw, Q.ptr<float>(), stream);
    }
    
    typedef void (*reprojectImageTo3D_caller_t)(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, const cudaStream_t& stream);
    
    const reprojectImageTo3D_caller_t reprojectImageTo3D_callers[] = {reprojectImageTo3D_caller<unsigned char>, 0, 0, reprojectImageTo3D_caller<short>, 0, 0, 0, 0};
}

void cv::gpu::reprojectImageTo3D_GPU(const GpuMat& disp, GpuMat& xyzw, const Mat& Q)
{
    CV_Assert((disp.type() == CV_8U || disp.type() == CV_16S) && Q.type() == CV_32F && Q.rows == 4 && Q.cols == 4);
    
    reprojectImageTo3D_callers[disp.type()](disp, xyzw, Q, 0);
}

void cv::gpu::reprojectImageTo3D_GPU(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, const Stream& stream)
{
    CV_Assert((disp.type() == CV_8U || disp.type() == CV_16S) && Q.type() == CV_32F && Q.rows == 4 && Q.cols == 4);
    
    reprojectImageTo3D_callers[disp.type()](disp, xyzw, Q, StreamAccessor::getStream(stream));
}

#endif /* !defined (HAVE_CUDA) */