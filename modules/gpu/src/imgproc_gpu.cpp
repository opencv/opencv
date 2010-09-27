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

void cv::gpu::remap(const GpuMat&, GpuMat&, const GpuMat&, const GpuMat&){ throw_nogpu(); }
void cv::gpu::meanShiftFiltering(const GpuMat&, GpuMat&, int, int, TermCriteria) { throw_nogpu(); }
void cv::gpu::drawColorDisp(const GpuMat&, GpuMat&, int) { throw_nogpu(); }
void cv::gpu::drawColorDisp(const GpuMat&, GpuMat&, int, const Stream&) { throw_nogpu(); }
void cv::gpu::reprojectImageTo3D(const GpuMat&, GpuMat&, const Mat&) { throw_nogpu(); }
void cv::gpu::reprojectImageTo3D(const GpuMat&, GpuMat&, const Mat&, const Stream&) { throw_nogpu(); }
void cv::gpu::cvtColor(const GpuMat&, GpuMat&, int, int) { throw_nogpu(); }
void cv::gpu::cvtColor(const GpuMat&, GpuMat&, int, int, const Stream&) { throw_nogpu(); }
double cv::gpu::threshold(const GpuMat&, GpuMat&, double) { throw_nogpu(); return 0.0; }
void cv::gpu::resize(const GpuMat&, GpuMat&, Size, double, double, int) { throw_nogpu(); }
void cv::gpu::copyMakeBorder(const GpuMat&, GpuMat&, int, int, int, int, const Scalar&) { throw_nogpu(); }
void cv::gpu::warpAffine(const GpuMat&, GpuMat&, const Mat&, Size, int) { throw_nogpu(); }
void cv::gpu::warpPerspective(const GpuMat&, GpuMat&, const Mat&, Size, int) { throw_nogpu(); }
void cv::gpu::rotate(const GpuMat&, GpuMat&, Size, double, double, double, int) { throw_nogpu(); }
void cv::gpu::integral(GpuMat&, GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::boxFilter(const GpuMat&, GpuMat&, Size, Point) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu 
{ 
    namespace improc 
    {
        void remap_gpu_1c(const DevMem2D& src, const DevMem2Df& xmap, const DevMem2Df& ymap, DevMem2D dst);
        void remap_gpu_3c(const DevMem2D& src, const DevMem2Df& xmap, const DevMem2Df& ymap, DevMem2D dst);

        extern "C" void meanShiftFiltering_gpu(const DevMem2D& src, DevMem2D dst, int sp, int sr, int maxIter, float eps);

        void drawColorDisp_gpu(const DevMem2D& src, const DevMem2D& dst, int ndisp, const cudaStream_t& stream);
        void drawColorDisp_gpu(const DevMem2D_<short>& src, const DevMem2D& dst, int ndisp, const cudaStream_t& stream);

        void reprojectImageTo3D_gpu(const DevMem2D& disp, const DevMem2Df& xyzw, const float* q, const cudaStream_t& stream);
        void reprojectImageTo3D_gpu(const DevMem2D_<short>& disp, const DevMem2Df& xyzw, const float* q, const cudaStream_t& stream);

        void swapChannels_gpu_8u(const DevMem2D& src, const DevMem2D& dst, int cn, const int* coeffs, cudaStream_t stream);
        void swapChannels_gpu_16u(const DevMem2D& src, const DevMem2D& dst, int cn, const int* coeffs, cudaStream_t stream);
        void swapChannels_gpu_32f(const DevMem2D& src, const DevMem2D& dst, int cn, const int* coeffs, cudaStream_t stream);

        void RGB2RGB_gpu_8u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, cudaStream_t stream);
        void RGB2RGB_gpu_16u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, cudaStream_t stream);
        void RGB2RGB_gpu_32f(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, cudaStream_t stream);

        void RGB5x52RGB_gpu(const DevMem2D& src, int green_bits, const DevMem2D& dst, int dstcn, int bidx, cudaStream_t stream);
        void RGB2RGB5x5_gpu(const DevMem2D& src, int srccn, const DevMem2D& dst, int green_bits, int bidx, cudaStream_t stream);

        void Gray2RGB_gpu(const DevMem2D& src, const DevMem2D& dst, int dstcn, cudaStream_t stream);
        void Gray2RGB_gpu(const DevMem2D_<ushort>& src, const DevMem2D_<ushort>& dst, int dstcn, cudaStream_t stream);
        void Gray2RGB_gpu(const DevMem2Df& src, const DevMem2Df& dst, int dstcn, cudaStream_t stream);

        void RGB2Gray_gpu(const DevMem2D& src, int srccn, const DevMem2D& dst, int bidx, cudaStream_t stream);
        void RGB2Gray_gpu(const DevMem2D_<ushort>& src, int srccn, const DevMem2D_<ushort>& dst, int bidx, cudaStream_t stream);
        void RGB2Gray_gpu(const DevMem2Df& src, int srccn, const DevMem2Df& dst, int bidx, cudaStream_t stream);
    }
}}

////////////////////////////////////////////////////////////////////////
// remap

void cv::gpu::remap(const GpuMat& src, GpuMat& dst, const GpuMat& xmap, const GpuMat& ymap)
{
    typedef void (*remap_gpu_t)(const DevMem2D& src, const DevMem2Df& xmap, const DevMem2Df& ymap, DevMem2D dst);
    static const remap_gpu_t callers[] = {improc::remap_gpu_1c, 0, improc::remap_gpu_3c};

    CV_Assert((src.type() == CV_8U || src.type() == CV_8UC3) && xmap.type() == CV_32F && ymap.type() == CV_32F);

    GpuMat out;
    if (dst.data != src.data)
        out = dst;

    out.create(xmap.size(), src.type());
    
    callers[src.channels() - 1](src, xmap, ymap, out);
    
    dst = out;
}

////////////////////////////////////////////////////////////////////////
// meanShiftFiltering_GPU

void cv::gpu::meanShiftFiltering(const GpuMat& src, GpuMat& dst, int sp, int sr, TermCriteria criteria)
{       
    if( src.empty() )
        CV_Error( CV_StsBadArg, "The input image is empty" );

    if( src.depth() != CV_8U || src.channels() != 4 )
        CV_Error( CV_StsUnsupportedFormat, "Only 8-bit, 4-channel images are supported" );

    dst.create( src.size(), CV_8UC4 );
    
    if( !(criteria.type & TermCriteria::MAX_ITER) )
        criteria.maxCount = 5;
    
    int maxIter = std::min(std::max(criteria.maxCount, 1), 100);
    
    float eps;
    if( !(criteria.type & TermCriteria::EPS) )
        eps = 1.f;
    eps = (float)std::max(criteria.epsilon, 0.0);        

    improc::meanShiftFiltering_gpu(src, dst, sp, sr, maxIter, eps);    
}

////////////////////////////////////////////////////////////////////////
// drawColorDisp

namespace
{
    template <typename T>
    void drawColorDisp_caller(const GpuMat& src, GpuMat& dst, int ndisp, const cudaStream_t& stream)
    {        
        GpuMat out;
        if (dst.data != src.data)
            out = dst;
        out.create(src.size(), CV_8UC4);

        improc::drawColorDisp_gpu((DevMem2D_<T>)src, out, ndisp, stream);

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

////////////////////////////////////////////////////////////////////////
// reprojectImageTo3D

namespace
{
    template <typename T>
    void reprojectImageTo3D_caller(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, const cudaStream_t& stream)
    {        
        xyzw.create(disp.rows, disp.cols, CV_32FC4);
        improc::reprojectImageTo3D_gpu((DevMem2D_<T>)disp, xyzw, Q.ptr<float>(), stream);
    }
    
    typedef void (*reprojectImageTo3D_caller_t)(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, const cudaStream_t& stream);
    
    const reprojectImageTo3D_caller_t reprojectImageTo3D_callers[] = {reprojectImageTo3D_caller<unsigned char>, 0, 0, reprojectImageTo3D_caller<short>, 0, 0, 0, 0};
}

void cv::gpu::reprojectImageTo3D(const GpuMat& disp, GpuMat& xyzw, const Mat& Q)
{
    CV_Assert((disp.type() == CV_8U || disp.type() == CV_16S) && Q.type() == CV_32F && Q.rows == 4 && Q.cols == 4);
    
    reprojectImageTo3D_callers[disp.type()](disp, xyzw, Q, 0);
}

void cv::gpu::reprojectImageTo3D(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, const Stream& stream)
{
    CV_Assert((disp.type() == CV_8U || disp.type() == CV_16S) && Q.type() == CV_32F && Q.rows == 4 && Q.cols == 4);
    
    reprojectImageTo3D_callers[disp.type()](disp, xyzw, Q, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// cvtColor

namespace
{
    void cvtColor_caller(const GpuMat& src, GpuMat& dst, int code, int dcn, const cudaStream_t& stream) 
    {
        Size sz = src.size();
        int scn = src.channels(), depth = src.depth(), bidx;
        
        CV_Assert(depth == CV_8U || depth == CV_16U || depth == CV_32F);

        GpuMat out;
        if (dst.data != src.data)
            out = dst;

        NppiSize nppsz;
        nppsz.height = src.rows;
        nppsz.width = src.cols;

        switch (code)
        {
            case CV_BGR2BGRA: case CV_RGB2BGRA: case CV_BGRA2BGR:
            case CV_RGBA2BGR: case CV_RGB2BGR: case CV_BGRA2RGBA:
                CV_Assert(scn == 3 || scn == 4);

                dcn = code == CV_BGR2BGRA || code == CV_RGB2BGRA || code == CV_BGRA2RGBA ? 4 : 3;
                bidx = code == CV_BGR2BGRA || code == CV_BGRA2BGR ? 0 : 2;
                
                out.create(sz, CV_MAKETYPE(depth, dcn));
                if( depth == CV_8U )
                    improc::RGB2RGB_gpu_8u(src, scn, out, dcn, bidx, stream);
                else if( depth == CV_16U )
                    improc::RGB2RGB_gpu_16u(src, scn, out, dcn, bidx, stream);
                else
                    improc::RGB2RGB_gpu_32f(src, scn, out, dcn, bidx, stream);
                break;
                
            case CV_BGR2BGR565: case CV_BGR2BGR555: case CV_RGB2BGR565: case CV_RGB2BGR555:
            case CV_BGRA2BGR565: case CV_BGRA2BGR555: case CV_RGBA2BGR565: case CV_RGBA2BGR555:
                CV_Assert( (scn == 3 || scn == 4) && depth == CV_8U );
                out.create(sz, CV_8UC2);

                improc::RGB2RGB5x5_gpu(src, scn, out, code == CV_BGR2BGR565 || code == CV_RGB2BGR565 ||
                          code == CV_BGRA2BGR565 || code == CV_RGBA2BGR565 ? 6 : 5,
                          code == CV_BGR2BGR565 || code == CV_BGR2BGR555 ||
                          code == CV_BGRA2BGR565 || code == CV_BGRA2BGR555 ? 0 : 2,
                          stream);
                break;
            
            //case CV_BGR5652BGR: case CV_BGR5552BGR: case CV_BGR5652RGB: case CV_BGR5552RGB:
            //case CV_BGR5652BGRA: case CV_BGR5552BGRA: case CV_BGR5652RGBA: case CV_BGR5552RGBA:
            //    if(dcn <= 0) dcn = 3;
            //    CV_Assert( (dcn == 3 || dcn == 4) && scn == 2 && depth == CV_8U );
            //    out.create(sz, CV_MAKETYPE(depth, dcn));

            //    improc::RGB5x52RGB_gpu(src, code == CV_BGR2BGR565 || code == CV_RGB2BGR565 ||
            //              code == CV_BGRA2BGR565 || code == CV_RGBA2BGR565 ? 6 : 5, out, dcn,
            //              code == CV_BGR2BGR565 || code == CV_BGR2BGR555 ||
            //              code == CV_BGRA2BGR565 || code == CV_BGRA2BGR555 ? 0 : 2,
            //              stream);
            //    break;
                        
            case CV_BGR2GRAY: case CV_BGRA2GRAY: case CV_RGB2GRAY: case CV_RGBA2GRAY:
                CV_Assert(scn == 3 || scn == 4);

                out.create(sz, CV_MAKETYPE(depth, 1));
                bidx = code == CV_BGR2GRAY || code == CV_BGRA2GRAY ? 0 : 2;
                
                if( depth == CV_8U )
                    improc::RGB2Gray_gpu((DevMem2D)src, scn, (DevMem2D)out, bidx, stream);
                else if( depth == CV_16U )
                    improc::RGB2Gray_gpu((DevMem2D_<unsigned short>)src, scn, (DevMem2D_<unsigned short>)out, bidx, stream);
                else
                    improc::RGB2Gray_gpu((DevMem2Df)src, scn, (DevMem2Df)out, bidx, stream);
                break;
            
            //case CV_BGR5652GRAY: case CV_BGR5552GRAY:
            //    CV_Assert( scn == 2 && depth == CV_8U );
            //    dst.create(sz, CV_8UC1);
            //    CvtColorLoop(src, dst, RGB5x52Gray(code == CV_BGR5652GRAY ? 6 : 5));
            //    break;
            
            case CV_GRAY2BGR: case CV_GRAY2BGRA:
                if (dcn <= 0) 
                    dcn = 3;
                CV_Assert(scn == 1 && (dcn == 3 || dcn == 4));

                out.create(sz, CV_MAKETYPE(depth, dcn));
                
                if( depth == CV_8U )
                    improc::Gray2RGB_gpu((DevMem2D)src, (DevMem2D)out, dcn, stream);
                else if( depth == CV_16U )
                    improc::Gray2RGB_gpu((DevMem2D_<unsigned short>)src, (DevMem2D_<unsigned short>)out, dcn, stream);
                else
                    improc::Gray2RGB_gpu((DevMem2Df)src, (DevMem2Df)out, dcn, stream);
                break;
                
            //case CV_GRAY2BGR565: case CV_GRAY2BGR555:
            //    CV_Assert( scn == 1 && depth == CV_8U );
            //    dst.create(sz, CV_8UC2);
            //    
            //    CvtColorLoop(src, dst, Gray2RGB5x5(code == CV_GRAY2BGR565 ? 6 : 5));
            //    break;
                
            case CV_RGB2YCrCb:
                CV_Assert(scn == 3 && depth == CV_8U);
                
                out.create(sz, CV_MAKETYPE(depth, 3));

                nppSafeCall( nppiRGBToYCbCr_8u_C3R(src.ptr<Npp8u>(), src.step, out.ptr<Npp8u>(), out.step, nppsz) );
                {
                    static int coeffs[] = {0, 2, 1};
                    improc::swapChannels_gpu_8u(out, out, 3, coeffs, 0);
                }
                break;

            case CV_YCrCb2RGB:
                CV_Assert(scn == 3 && depth == CV_8U);
                
                out.create(sz, CV_MAKETYPE(depth, 3));

                {
                    static int coeffs[] = {0, 2, 1};
                    GpuMat src1(src.size(), src.type());
                    improc::swapChannels_gpu_8u(src, src1, 3, coeffs, 0);
                    nppSafeCall( nppiYCbCrToRGB_8u_C3R(src1.ptr<Npp8u>(), src1.step, out.ptr<Npp8u>(), out.step, nppsz) );   
                }             
                break;

            //case CV_BGR2YCrCb: case CV_RGB2YCrCb:
            //case CV_BGR2YUV: case CV_RGB2YUV:
            //    {
            //    CV_Assert( scn == 3 || scn == 4 );
            //    bidx = code == CV_BGR2YCrCb || code == CV_RGB2YUV ? 0 : 2;
            //    static const float yuv_f[] = { 0.114f, 0.587f, 0.299f, 0.492f, 0.877f };
            //    static const int yuv_i[] = { B2Y, G2Y, R2Y, 8061, 14369 };
            //    const float* coeffs_f = code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? 0 : yuv_f;
            //    const int* coeffs_i = code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? 0 : yuv_i;
            //        
            //    dst.create(sz, CV_MAKETYPE(depth, 3));
            //    
            //    if( depth == CV_8U )
            //        CvtColorLoop(src, dst, RGB2YCrCb_i<uchar>(scn, bidx, coeffs_i));
            //    else if( depth == CV_16U )
            //        CvtColorLoop(src, dst, RGB2YCrCb_i<ushort>(scn, bidx, coeffs_i));
            //    else
            //        CvtColorLoop(src, dst, RGB2YCrCb_f<float>(scn, bidx, coeffs_f));
            //    }
            //    break;
                
            //case CV_YCrCb2BGR: case CV_YCrCb2RGB:
            //case CV_YUV2BGR: case CV_YUV2RGB:
            //    {
            //    if( dcn <= 0 ) dcn = 3;
            //    CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) );
            //    bidx = code == CV_YCrCb2BGR || code == CV_YUV2RGB ? 0 : 2;
            //    static const float yuv_f[] = { 2.032f, -0.395f, -0.581f, 1.140f };
            //    static const int yuv_i[] = { 33292, -6472, -9519, 18678 }; 
            //    const float* coeffs_f = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? 0 : yuv_f;
            //    const int* coeffs_i = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? 0 : yuv_i;
            //    
            //    dst.create(sz, CV_MAKETYPE(depth, dcn));
            //    
            //    if( depth == CV_8U )
            //        CvtColorLoop(src, dst, YCrCb2RGB_i<uchar>(dcn, bidx, coeffs_i));
            //    else if( depth == CV_16U )
            //        CvtColorLoop(src, dst, YCrCb2RGB_i<ushort>(dcn, bidx, coeffs_i));
            //    else
            //        CvtColorLoop(src, dst, YCrCb2RGB_f<float>(dcn, bidx, coeffs_f));
            //    }
            //    break;
            
            //case CV_BGR2XYZ: case CV_RGB2XYZ:
            //    CV_Assert( scn == 3 || scn == 4 );
            //    bidx = code == CV_BGR2XYZ ? 0 : 2;
            //    
            //    dst.create(sz, CV_MAKETYPE(depth, 3));
            //    
            //    if( depth == CV_8U )
            //        CvtColorLoop(src, dst, RGB2XYZ_i<uchar>(scn, bidx, 0));
            //    else if( depth == CV_16U )
            //        CvtColorLoop(src, dst, RGB2XYZ_i<ushort>(scn, bidx, 0));
            //    else
            //        CvtColorLoop(src, dst, RGB2XYZ_f<float>(scn, bidx, 0));
            //    break;
            
            //case CV_XYZ2BGR: case CV_XYZ2RGB:
            //    if( dcn <= 0 ) dcn = 3;
            //    CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) );
            //    bidx = code == CV_XYZ2BGR ? 0 : 2;
            //    
            //    dst.create(sz, CV_MAKETYPE(depth, dcn));
            //    
            //    if( depth == CV_8U )
            //        CvtColorLoop(src, dst, XYZ2RGB_i<uchar>(dcn, bidx, 0));
            //    else if( depth == CV_16U )
            //        CvtColorLoop(src, dst, XYZ2RGB_i<ushort>(dcn, bidx, 0));
            //    else
            //        CvtColorLoop(src, dst, XYZ2RGB_f<float>(dcn, bidx, 0));
            //    break;
                
            //case CV_BGR2HSV: case CV_RGB2HSV: case CV_BGR2HSV_FULL: case CV_RGB2HSV_FULL:
            //case CV_BGR2HLS: case CV_RGB2HLS: case CV_BGR2HLS_FULL: case CV_RGB2HLS_FULL:
            //    {
            //    CV_Assert( (scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F) );
            //    bidx = code == CV_BGR2HSV || code == CV_BGR2HLS ||
            //        code == CV_BGR2HSV_FULL || code == CV_BGR2HLS_FULL ? 0 : 2;
            //    int hrange = depth == CV_32F ? 360 : code == CV_BGR2HSV || code == CV_RGB2HSV ||
            //        code == CV_BGR2HLS || code == CV_RGB2HLS ? 180 : 255;
            //    
            //    dst.create(sz, CV_MAKETYPE(depth, 3));
            //    
            //    if( code == CV_BGR2HSV || code == CV_RGB2HSV ||
            //        code == CV_BGR2HSV_FULL || code == CV_RGB2HSV_FULL )
            //    {
            //        if( depth == CV_8U )
            //            CvtColorLoop(src, dst, RGB2HSV_b(scn, bidx, hrange));
            //        else
            //            CvtColorLoop(src, dst, RGB2HSV_f(scn, bidx, (float)hrange));
            //    }
            //    else
            //    {
            //        if( depth == CV_8U )
            //            CvtColorLoop(src, dst, RGB2HLS_b(scn, bidx, hrange));
            //        else
            //            CvtColorLoop(src, dst, RGB2HLS_f(scn, bidx, (float)hrange));
            //    }
            //    }
            //    break;
            
            //case CV_HSV2BGR: case CV_HSV2RGB: case CV_HSV2BGR_FULL: case CV_HSV2RGB_FULL:
            //case CV_HLS2BGR: case CV_HLS2RGB: case CV_HLS2BGR_FULL: case CV_HLS2RGB_FULL:
            //    {
            //    if( dcn <= 0 ) dcn = 3;
            //    CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F) );
            //    bidx = code == CV_HSV2BGR || code == CV_HLS2BGR ||
            //        code == CV_HSV2BGR_FULL || code == CV_HLS2BGR_FULL ? 0 : 2;
            //    int hrange = depth == CV_32F ? 360 : code == CV_HSV2BGR || code == CV_HSV2RGB ||
            //        code == CV_HLS2BGR || code == CV_HLS2RGB ? 180 : 255;
            //    
            //    dst.create(sz, CV_MAKETYPE(depth, dcn));
            //    
            //    if( code == CV_HSV2BGR || code == CV_HSV2RGB ||
            //        code == CV_HSV2BGR_FULL || code == CV_HSV2RGB_FULL )
            //    {
            //        if( depth == CV_8U )
            //            CvtColorLoop(src, dst, HSV2RGB_b(dcn, bidx, hrange));
            //        else
            //            CvtColorLoop(src, dst, HSV2RGB_f(dcn, bidx, (float)hrange));
            //    }
            //    else
            //    {
            //        if( depth == CV_8U )
            //            CvtColorLoop(src, dst, HLS2RGB_b(dcn, bidx, hrange));
            //        else
            //            CvtColorLoop(src, dst, HLS2RGB_f(dcn, bidx, (float)hrange));
            //    }
            //    }
            //    break;
                
            //case CV_BGR2Lab: case CV_RGB2Lab: case CV_LBGR2Lab: case CV_LRGB2Lab:
            //case CV_BGR2Luv: case CV_RGB2Luv: case CV_LBGR2Luv: case CV_LRGB2Luv:
            //    {
            //    CV_Assert( (scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F) );
            //    bidx = code == CV_BGR2Lab || code == CV_BGR2Luv ||
            //           code == CV_LBGR2Lab || code == CV_LBGR2Luv ? 0 : 2;
            //    bool srgb = code == CV_BGR2Lab || code == CV_RGB2Lab ||
            //                code == CV_BGR2Luv || code == CV_RGB2Luv;
            //    
            //    dst.create(sz, CV_MAKETYPE(depth, 3));
            //    
            //    if( code == CV_BGR2Lab || code == CV_RGB2Lab ||
            //        code == CV_LBGR2Lab || code == CV_LRGB2Lab )
            //    {
            //        if( depth == CV_8U )
            //            CvtColorLoop(src, dst, RGB2Lab_b(scn, bidx, 0, 0, srgb));
            //        else
            //            CvtColorLoop(src, dst, RGB2Lab_f(scn, bidx, 0, 0, srgb));
            //    }
            //    else
            //    {
            //        if( depth == CV_8U )
            //            CvtColorLoop(src, dst, RGB2Luv_b(scn, bidx, 0, 0, srgb));
            //        else
            //            CvtColorLoop(src, dst, RGB2Luv_f(scn, bidx, 0, 0, srgb));
            //    }
            //    }
            //    break;
            
            //case CV_Lab2BGR: case CV_Lab2RGB: case CV_Lab2LBGR: case CV_Lab2LRGB:
            //case CV_Luv2BGR: case CV_Luv2RGB: case CV_Luv2LBGR: case CV_Luv2LRGB:
            //    {
            //    if( dcn <= 0 ) dcn = 3;
            //    CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F) );
            //    bidx = code == CV_Lab2BGR || code == CV_Luv2BGR ||
            //           code == CV_Lab2LBGR || code == CV_Luv2LBGR ? 0 : 2;
            //    bool srgb = code == CV_Lab2BGR || code == CV_Lab2RGB ||
            //            code == CV_Luv2BGR || code == CV_Luv2RGB;
            //    
            //    dst.create(sz, CV_MAKETYPE(depth, dcn));
            //    
            //    if( code == CV_Lab2BGR || code == CV_Lab2RGB ||
            //        code == CV_Lab2LBGR || code == CV_Lab2LRGB )
            //    {
            //        if( depth == CV_8U )
            //            CvtColorLoop(src, dst, Lab2RGB_b(dcn, bidx, 0, 0, srgb));
            //        else
            //            CvtColorLoop(src, dst, Lab2RGB_f(dcn, bidx, 0, 0, srgb));
            //    }
            //    else
            //    {
            //        if( depth == CV_8U )
            //            CvtColorLoop(src, dst, Luv2RGB_b(dcn, bidx, 0, 0, srgb));
            //        else
            //            CvtColorLoop(src, dst, Luv2RGB_f(dcn, bidx, 0, 0, srgb));
            //    }
            //    }
            //    break;
                
            //case CV_BayerBG2BGR: case CV_BayerGB2BGR: case CV_BayerRG2BGR: case CV_BayerGR2BGR:
            //case CV_BayerBG2BGR_VNG: case CV_BayerGB2BGR_VNG: case CV_BayerRG2BGR_VNG: case CV_BayerGR2BGR_VNG:
            //    if(dcn <= 0) dcn = 3;
            //    CV_Assert( scn == 1 && dcn == 3 && depth == CV_8U );
            //    dst.create(sz, CV_8UC3);
            //    
            //    if( code == CV_BayerBG2BGR || code == CV_BayerGB2BGR ||
            //        code == CV_BayerRG2BGR || code == CV_BayerGR2BGR )
            //        Bayer2RGB_8u(src, dst, code);
            //    else
            //        Bayer2RGB_VNG_8u(src, dst, code);
            //    break;

            default:
                CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );
        }

        dst = out;
    }
}

void cv::gpu::cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn)
{
    cvtColor_caller(src, dst, code, dcn, 0);
}

void cv::gpu::cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn, const Stream& stream)
{
    cvtColor_caller(src, dst, code, dcn, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// threshold

double cv::gpu::threshold(const GpuMat& src, GpuMat& dst, double thresh) 
{ 
    CV_Assert(src.type() == CV_32FC1)

    dst.create( src.size(), src.type() );

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiThreshold_32f_C1R(src.ptr<Npp32f>(), src.step, 
        dst.ptr<Npp32f>(), dst.step, sz, static_cast<Npp32f>(thresh), NPP_CMP_GREATER) );

    return thresh;
}

////////////////////////////////////////////////////////////////////////
// resize

void cv::gpu::resize(const GpuMat& src, GpuMat& dst, Size dsize, double fx, double fy, int interpolation)
{
    static const int npp_inter[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC, 0, NPPI_INTER_LANCZOS};

    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);
    CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC || interpolation == INTER_LANCZOS4);

    CV_Assert( src.size().area() > 0 );
    CV_Assert( !(dsize == Size()) || (fx > 0 && fy > 0) );

    if( dsize == Size() )
    {
        dsize = Size(saturate_cast<int>(src.cols * fx), saturate_cast<int>(src.rows * fy));
    }
    else
    {
        fx = (double)dsize.width / src.cols;
        fy = (double)dsize.height / src.rows;
    }

    dst.create(dsize, src.type());

    NppiSize srcsz;
    srcsz.width  = src.cols;
    srcsz.height = src.rows;
    NppiRect srcrect;
    srcrect.x = srcrect.y = 0;
    srcrect.width  = src.cols;
    srcrect.height = src.rows;
    NppiSize dstsz;
    dstsz.width  = dst.cols;
    dstsz.height = dst.rows;

    if (src.type() == CV_8UC1)
    {
        nppSafeCall( nppiResize_8u_C1R(src.ptr<Npp8u>(), srcsz, src.step, srcrect,
            dst.ptr<Npp8u>(), dst.step, dstsz, fx, fy, npp_inter[interpolation]) );
    }
    else
    {
        nppSafeCall( nppiResize_8u_C4R(src.ptr<Npp8u>(), srcsz, src.step, srcrect,
            dst.ptr<Npp8u>(), dst.step, dstsz, fx, fy, npp_inter[interpolation]) );
    }
}

////////////////////////////////////////////////////////////////////////
// copyMakeBorder

void cv::gpu::copyMakeBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, const Scalar& value) 
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4 || src.type() == CV_32SC1);

    dst.create(src.rows + top + bottom, src.cols + left + right, src.type());

	NppiSize srcsz;
	srcsz.width  = src.cols;
	srcsz.height = src.rows;
    NppiSize dstsz;
	dstsz.width  = dst.cols;
	dstsz.height = dst.rows;

    switch (src.type())
    {
    case CV_8UC1:
		{
            Npp8u nVal = static_cast<Npp8u>(value[0]);
            nppSafeCall( nppiCopyConstBorder_8u_C1R(src.ptr<Npp8u>(), src.step, srcsz, 
                dst.ptr<Npp8u>(), dst.step, dstsz, top, left, nVal) );
            break;
		}
    case CV_8UC4:
		{
            Npp8u nVal[] = {static_cast<Npp8u>(value[0]), static_cast<Npp8u>(value[1]), static_cast<Npp8u>(value[2]), static_cast<Npp8u>(value[3])};
            nppSafeCall( nppiCopyConstBorder_8u_C4R(src.ptr<Npp8u>(), src.step, srcsz, 
                dst.ptr<Npp8u>(), dst.step, dstsz, top, left, nVal) );
            break;
		}
    case CV_32SC1:
	    {
            Npp32s nVal = static_cast<Npp32s>(value[0]);
            nppSafeCall( nppiCopyConstBorder_32s_C1R(src.ptr<Npp32s>(), src.step, srcsz, 
                dst.ptr<Npp32s>(), dst.step, dstsz, top, left, nVal) );
            break;
	    }
    default:
        CV_Assert(!"Unsupported source type");
    }
}

////////////////////////////////////////////////////////////////////////
// warp

namespace
{    
    typedef NppStatus (*npp_warp_8u_t)(const Npp8u* pSrc, NppiSize srcSize, int srcStep, NppiRect srcRoi, Npp8u* pDst, 
                                       int dstStep, NppiRect dstRoi, const double coeffs[][3], 
                                       int interpolation);
    typedef NppStatus (*npp_warp_16u_t)(const Npp16u* pSrc, NppiSize srcSize, int srcStep, NppiRect srcRoi, Npp16u* pDst, 
                                       int dstStep, NppiRect dstRoi, const double coeffs[][3], 
                                       int interpolation);
    typedef NppStatus (*npp_warp_32s_t)(const Npp32s* pSrc, NppiSize srcSize, int srcStep, NppiRect srcRoi, Npp32s* pDst, 
                                       int dstStep, NppiRect dstRoi, const double coeffs[][3], 
                                       int interpolation);
    typedef NppStatus (*npp_warp_32f_t)(const Npp32f* pSrc, NppiSize srcSize, int srcStep, NppiRect srcRoi, Npp32f* pDst, 
                                       int dstStep, NppiRect dstRoi, const double coeffs[][3], 
                                       int interpolation);

    void nppWarpCaller(const GpuMat& src, GpuMat& dst, double coeffs[][3], const Size& dsize, int flags, 
                       npp_warp_8u_t npp_warp_8u[][2], npp_warp_16u_t npp_warp_16u[][2], 
                       npp_warp_32s_t npp_warp_32s[][2], npp_warp_32f_t npp_warp_32f[][2]) 
    {
        static const int npp_inter[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
    
        int interpolation = flags & INTER_MAX;

        CV_Assert((src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32S || src.depth() == CV_32F) && src.channels() != 2);
        CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC);

        dst.create(dsize, src.type());

        NppiSize srcsz;
        srcsz.height = src.rows;
        srcsz.width = src.cols;
        NppiRect srcroi;
        srcroi.x = srcroi.y = 0;
        srcroi.height = src.rows;
        srcroi.width = src.cols;
        NppiRect dstroi;
        dstroi.x = dstroi.y = 0;
        dstroi.height = dst.rows;
        dstroi.width = dst.cols;

        int warpInd = (flags & WARP_INVERSE_MAP) >> 4;

        switch (src.depth())
        {
        case CV_8U:
            nppSafeCall( npp_warp_8u[src.channels()][warpInd](src.ptr<Npp8u>(), srcsz, src.step, srcroi, 
                dst.ptr<Npp8u>(), dst.step, dstroi, coeffs, npp_inter[interpolation]) );
            break;
        case CV_16U:
            nppSafeCall( npp_warp_16u[src.channels()][warpInd](src.ptr<Npp16u>(), srcsz, src.step, srcroi, 
                dst.ptr<Npp16u>(), dst.step, dstroi, coeffs, npp_inter[interpolation]) );
            break;
        case CV_32S:
            nppSafeCall( npp_warp_32s[src.channels()][warpInd](src.ptr<Npp32s>(), srcsz, src.step, srcroi, 
                dst.ptr<Npp32s>(), dst.step, dstroi, coeffs, npp_inter[interpolation]) );
            break;
        case CV_32F:
            nppSafeCall( npp_warp_32f[src.channels()][warpInd](src.ptr<Npp32f>(), srcsz, src.step, srcroi, 
                dst.ptr<Npp32f>(), dst.step, dstroi, coeffs, npp_inter[interpolation]) );
            break;
        default:
            CV_Assert(!"Unsupported source type");
        }
    }
}

void cv::gpu::warpAffine(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags) 
{
    static npp_warp_8u_t npp_warpAffine_8u[][2] = 
        {
            {0, 0}, 
            {nppiWarpAffine_8u_C1R, nppiWarpAffineBack_8u_C1R}, 
            {0, 0}, 
            {nppiWarpAffine_8u_C3R, nppiWarpAffineBack_8u_C3R}, 
            {nppiWarpAffine_8u_C4R, nppiWarpAffineBack_8u_C4R}
        };
    static npp_warp_16u_t npp_warpAffine_16u[][2] = 
        {
            {0, 0}, 
            {nppiWarpAffine_16u_C1R, nppiWarpAffineBack_16u_C1R}, 
            {0, 0}, 
            {nppiWarpAffine_16u_C3R, nppiWarpAffineBack_16u_C3R}, 
            {nppiWarpAffine_16u_C4R, nppiWarpAffineBack_16u_C4R}
        };
    static npp_warp_32s_t npp_warpAffine_32s[][2] = 
        {
            {0, 0}, 
            {nppiWarpAffine_32s_C1R, nppiWarpAffineBack_32s_C1R}, 
            {0, 0}, 
            {nppiWarpAffine_32s_C3R, nppiWarpAffineBack_32s_C3R}, 
            {nppiWarpAffine_32s_C4R, nppiWarpAffineBack_32s_C4R}
        };
    static npp_warp_32f_t npp_warpAffine_32f[][2] = 
        {
            {0, 0}, 
            {nppiWarpAffine_32f_C1R, nppiWarpAffineBack_32f_C1R}, 
            {0, 0}, 
            {nppiWarpAffine_32f_C3R, nppiWarpAffineBack_32f_C3R}, 
            {nppiWarpAffine_32f_C4R, nppiWarpAffineBack_32f_C4R}
        };

    CV_Assert(M.rows == 2 && M.cols == 3);

    double coeffs[2][3];
    Mat coeffsMat(2, 3, CV_64F, (void*)coeffs);
    M.convertTo(coeffsMat, coeffsMat.type());

    nppWarpCaller(src, dst, coeffs, dsize, flags, npp_warpAffine_8u, npp_warpAffine_16u, npp_warpAffine_32s, npp_warpAffine_32f);
}

void cv::gpu::warpPerspective(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags)
{
    static npp_warp_8u_t npp_warpPerspective_8u[][2] = 
        {
            {0, 0}, 
            {nppiWarpPerspective_8u_C1R, nppiWarpPerspectiveBack_8u_C1R}, 
            {0, 0}, 
            {nppiWarpPerspective_8u_C3R, nppiWarpPerspectiveBack_8u_C3R}, 
            {nppiWarpPerspective_8u_C4R, nppiWarpPerspectiveBack_8u_C4R}
        };
    static npp_warp_16u_t npp_warpPerspective_16u[][2] = 
        {
            {0, 0}, 
            {nppiWarpPerspective_16u_C1R, nppiWarpPerspectiveBack_16u_C1R}, 
            {0, 0}, 
            {nppiWarpPerspective_16u_C3R, nppiWarpPerspectiveBack_16u_C3R}, 
            {nppiWarpPerspective_16u_C4R, nppiWarpPerspectiveBack_16u_C4R}
        };
    static npp_warp_32s_t npp_warpPerspective_32s[][2] = 
        {
            {0, 0}, 
            {nppiWarpPerspective_32s_C1R, nppiWarpPerspectiveBack_32s_C1R}, 
            {0, 0}, 
            {nppiWarpPerspective_32s_C3R, nppiWarpPerspectiveBack_32s_C3R}, 
            {nppiWarpPerspective_32s_C4R, nppiWarpPerspectiveBack_32s_C4R}
        };
    static npp_warp_32f_t npp_warpPerspective_32f[][2] = 
        {
            {0, 0}, 
            {nppiWarpPerspective_32f_C1R, nppiWarpPerspectiveBack_32f_C1R}, 
            {0, 0}, 
            {nppiWarpPerspective_32f_C3R, nppiWarpPerspectiveBack_32f_C3R}, 
            {nppiWarpPerspective_32f_C4R, nppiWarpPerspectiveBack_32f_C4R}
        };

    CV_Assert(M.rows == 3 && M.cols == 3);

    double coeffs[3][3];
    Mat coeffsMat(3, 3, CV_64F, (void*)coeffs);
    M.convertTo(coeffsMat, coeffsMat.type());

    nppWarpCaller(src, dst, coeffs, dsize, flags, npp_warpPerspective_8u, npp_warpPerspective_16u, npp_warpPerspective_32s, npp_warpPerspective_32f);
}

////////////////////////////////////////////////////////////////////////
// rotate

void cv::gpu::rotate(const GpuMat& src, GpuMat& dst, Size dsize, double angle, double xShift, double yShift, int interpolation)
{
    static const int npp_inter[] = {NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC};
    
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);
    CV_Assert(interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC);

    dst.create(dsize, src.type());

    NppiSize srcsz;
    srcsz.height = src.rows;
    srcsz.width = src.cols;
    NppiRect srcroi;
    srcroi.x = srcroi.y = 0;
    srcroi.height = src.rows;
    srcroi.width = src.cols;
    NppiRect dstroi;
    dstroi.x = dstroi.y = 0;
    dstroi.height = dst.rows;
    dstroi.width = dst.cols;

    if (src.type() == CV_8UC1)
    {
        nppSafeCall( nppiRotate_8u_C1R(src.ptr<Npp8u>(), srcsz, src.step, srcroi, 
            dst.ptr<Npp8u>(), dst.step, dstroi, angle, xShift, yShift, npp_inter[interpolation]) );
    }
    else
    {
        nppSafeCall( nppiRotate_8u_C4R(src.ptr<Npp8u>(), srcsz, src.step, srcroi, 
            dst.ptr<Npp8u>(), dst.step, dstroi, angle, xShift, yShift, npp_inter[interpolation]) );
    }
}

////////////////////////////////////////////////////////////////////////
// integral

void cv::gpu::integral(GpuMat& src, GpuMat& sum, GpuMat& sqsum)
{
    CV_Assert(src.type() == CV_8UC1);
    
    int w = src.cols + 1, h = src.rows + 1;

    sum.create(h, w, CV_32S);
    sqsum.create(h, w, CV_32F);

    NppiSize sz;
    sz.width = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiSqrIntegral_8u32s32f_C1R(src.ptr<Npp8u>(), src.step, sum.ptr<Npp32s>(), 
        sum.step, sqsum.ptr<Npp32f>(), sqsum.step, sz, 0, 0.0f, h) );
}

////////////////////////////////////////////////////////////////////////
// boxFilter

void cv::gpu::boxFilter(const GpuMat& src, GpuMat& dst, Size ksize, Point anchor)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);
    CV_Assert(ksize.height == 3 || ksize.height == 5 || ksize.height == 7);
    CV_Assert(ksize.height == ksize.width);

    if (anchor.x == -1)
        anchor.x = 0;
    if (anchor.y == -1)
        anchor.y = 0;

    CV_Assert(anchor.x == 0 && anchor.y == 0);

    dst.create(src.size(), src.type());

    NppiSize srcsz;
    srcsz.height = src.rows;
    srcsz.width = src.cols;
    NppiSize masksz;
    masksz.height = ksize.height;
    masksz.width = ksize.width;
    NppiPoint anc;
    anc.x = anchor.x;
    anc.y = anchor.y;

    if (src.type() == CV_8UC1)
    {
        nppSafeCall( nppiFilterBox_8u_C1R(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, srcsz, masksz, anc) );
    }
    else
    {
        nppSafeCall( nppiFilterBox_8u_C4R(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, srcsz, masksz, anc) );
    }
}

#endif /* !defined (HAVE_CUDA) */
