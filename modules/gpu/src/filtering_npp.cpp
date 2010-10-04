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
void cv::gpu::boxFilter(const GpuMat&, GpuMat&, Size, Point) { throw_nogpu(); }
void cv::gpu::sumWindowColumn(const GpuMat&, GpuMat&, int, int) { throw_nogpu(); }
void cv::gpu::sumWindowRow(const GpuMat&, GpuMat&, int, int) { throw_nogpu(); }
void cv::gpu::Sobel(const GpuMat&, GpuMat&, int, int, int, int, double) { throw_nogpu(); }
void cv::gpu::GaussianBlur(const GpuMat&, GpuMat&, Size, double, double) { throw_nogpu(); }

#else

namespace 
{
    typedef NppStatus (*npp_morf_func)(const Npp8u*, Npp32s, Npp8u*, Npp32s, NppiSize, const Npp8u*, NppiSize, NppiPoint);


    void morphoogy_caller(npp_morf_func func, const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor, int iterations)
    {
        CV_Assert(src.type() == CV_8U || src.type() == CV_8UC4);        
        CV_Assert(kernel.type() == CV_8U && (kernel.cols & 1) != 0 && (kernel.rows & 1) != 0);

        if( anchor.x == -1 )
            anchor.x = kernel.cols / 2;
        if( anchor.y == -1 )
            anchor.y = kernel.rows / 2;

        // in NPP for Cuda 3.1 only such anchor is supported.
        CV_Assert(anchor.x == 0 && anchor.y == 0);

        if (iterations == 0)
        {
            src.copyTo(dst);
            return;
        }

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
        GpuMat dstBuf;
        if (iterations > 1)
            dstBuf.create(src.size(), src.type());

        nppSafeCall( func(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz, gpu_krnl.ptr<Npp8u>(), mask_sz, anc) );
        for(int i = 1; i < iterations; ++i)
        {
            dst.swap(dstBuf);
            nppSafeCall( func(dstBuf.ptr<Npp8u>(), dstBuf.step, dst.ptr<Npp8u>(), dst.step, sz, gpu_krnl.ptr<Npp8u>(), mask_sz, anc) );
        }
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

////////////////////////////////////////////////////////////////////////
// sumWindow Filter

namespace
{
    typedef NppStatus (*nppSumWindow_t)(const Npp8u * pSrc, Npp32s nSrcStep, 
                                        Npp32f * pDst, Npp32s nDstStep, NppiSize oROI, 
                                        Npp32s nMaskSize, Npp32s nAnchor);

    inline void sumWindowCaller(nppSumWindow_t func, const GpuMat& src, GpuMat& dst, int ksize, int anchor)
    {
        CV_Assert(src.type() == CV_8UC1);
        
        if (anchor == -1)
            anchor = ksize / 2;

        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        dst.create(src.size(), CV_32FC1);

        nppSafeCall( func(src.ptr<Npp8u>(), src.step, dst.ptr<Npp32f>(), dst.step, sz, ksize, anchor) );
    }
}

void cv::gpu::sumWindowColumn(const GpuMat& src, GpuMat& dst, int ksize, int anchor)
{
    sumWindowCaller(nppiSumWindowColumn_8u32f_C1R, src, dst, ksize, anchor);
}

void cv::gpu::sumWindowRow(const GpuMat& src, GpuMat& dst, int ksize, int anchor)
{
    sumWindowCaller(nppiSumWindowRow_8u32f_C1R, src, dst, ksize, anchor);
}

////////////////////////////////////////////////////////////////////////
// Filter Engine

namespace
{
    typedef NppStatus (*nppFilter1D_t)(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
                     const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);
    typedef NppStatus (*nppFilter2D_t)(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);

    void applyRowFilter(const GpuMat& src, GpuMat& dst, const GpuMat& rowKernel, Npp32s anchor = -1, Npp32s nDivisor = 1)
    {
        static const nppFilter1D_t nppFilter1D_callers[] = {nppiFilterRow_8u_C1R, nppiFilterRow_8u_C4R};

        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);

        int kRowSize = rowKernel.cols;

        dst.create(src.size(), src.type());
        dst = Scalar();

        NppiSize oROI;
        oROI.width = src.cols - kRowSize + 1;
        oROI.height = src.rows;

        if (anchor < 0)
            anchor = kRowSize >> 1;

        GpuMat srcROI = src.colRange(kRowSize-1, oROI.width);
        GpuMat dstROI = dst.colRange(kRowSize-1, oROI.width);

        nppFilter1D_callers[src.channels() >> 2](srcROI.ptr<Npp8u>(), srcROI.step, dstROI.ptr<Npp8u>(), dstROI.step, oROI, 
                rowKernel.ptr<Npp32s>(), kRowSize, anchor, nDivisor);
    }

    void applyColumnFilter(const GpuMat& src, GpuMat& dst, const GpuMat& columnKernel, Npp32s anchor = -1, Npp32s nDivisor = 1)
    {
        static const nppFilter1D_t nppFilter1D_callers[] = {nppiFilterColumn_8u_C1R, nppiFilterColumn_8u_C4R};

        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);

        int kColSize = columnKernel.cols;

        dst.create(src.size(), src.type());
        dst = Scalar();

        NppiSize oROI;
        oROI.width = src.cols;
        oROI.height = src.rows - kColSize + 1;

        if (anchor < 0)
            anchor = kColSize >> 1;

        GpuMat srcROI = src.rowRange(kColSize-1, oROI.height);
        GpuMat dstROI = dst.rowRange(kColSize-1, oROI.height);
        
        nppFilter1D_callers[src.channels() >> 2](srcROI.ptr<Npp8u>(), srcROI.step, dstROI.ptr<Npp8u>(), dstROI.step, oROI, 
                columnKernel.ptr<Npp32s>(), kColSize, anchor, nDivisor);
    }

    inline void applySeparableFilter(const GpuMat& src, GpuMat& dst, const GpuMat& rowKernel, const GpuMat& columnKernel, 
        const cv::Point& anchor = cv::Point(-1, -1), Npp32s nDivisor = 1)
    {
        GpuMat dstBuf;
        applyRowFilter(src, dstBuf, rowKernel, anchor.x, nDivisor);
        applyColumnFilter(dstBuf, dst, columnKernel, anchor.y, nDivisor);
    }

    void makeNppKernel(Mat kernel, GpuMat& dst)
    {
        kernel.convertTo(kernel, CV_32S); 
        kernel = kernel.t();
        int ksize = kernel.cols;
        for (int i = 0; i < ksize / 2; ++i)
        {
            std::swap(kernel.at<int>(0, i), kernel.at<int>(0, ksize - 1 - i));
        }
        dst.upload(kernel);
    }

    void applyFilter2D(const GpuMat& src, GpuMat& dst, const GpuMat& kernel, cv::Point anchor = cv::Point(-1, -1), Npp32s nDivisor = 1)
    {
        static const nppFilter2D_t nppFilter2D_callers[] = {nppiFilter_8u_C1R, nppiFilter_8u_C4R};        

        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);

        dst.create(src.size(), src.type());
        dst = Scalar();

        NppiSize oROI;
        oROI.width = src.cols - kernel.cols + 1;
        oROI.height = src.rows - kernel.rows + 1;

        if (anchor.x < 0)
            anchor.x = kernel.cols >> 1;
        if (anchor.y < 0)
            anchor.y = kernel.rows >> 1;

        GpuMat srcROI = src(Range(kernel.rows-1, oROI.height), Range(kernel.cols-1, oROI.width));
        GpuMat dstROI = dst(Range(kernel.rows-1, oROI.height), Range(kernel.cols-1, oROI.width));

        NppiSize oKernelSize;
        oKernelSize.height = kernel.rows;
        oKernelSize.width = kernel.cols;
        NppiPoint oAnchor;
        oAnchor.x = anchor.x;
        oAnchor.y = anchor.y;
        
        nppFilter2D_callers[src.channels() >> 2](srcROI.ptr<Npp8u>(), srcROI.step, dstROI.ptr<Npp8u>(), dstROI.step, oROI, 
                kernel.ptr<Npp32s>(), oKernelSize, oAnchor, nDivisor);
    }
}

////////////////////////////////////////////////////////////////////////
// Sobel

void cv::gpu::Sobel(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, int ksize, double scale)
{
    Mat kx, ky;
    getDerivKernels(kx, ky, dx, dy, ksize, false, CV_32F);

    if (scale != 1)
    {
        // usually the smoothing part is the slowest to compute,
        // so try to scale it instead of the faster differenciating part
        if (dx == 0)
            kx *= scale;
        else
            ky *= scale;
    }
    
    GpuMat rowKernel; makeNppKernel(kx, rowKernel);
    GpuMat columnKernel; makeNppKernel(ky, columnKernel);

    applySeparableFilter(src, dst, rowKernel, columnKernel);
}

////////////////////////////////////////////////////////////////////////
// GaussianBlur

void cv::gpu::GaussianBlur(const GpuMat& src, GpuMat& dst, Size ksize, double sigma1, double sigma2)
{
    if (ksize.width == 1 && ksize.height == 1)
    {
        src.copyTo(dst);
        return;
    }

    int depth = src.depth();
    if (sigma2 <= 0)
        sigma2 = sigma1;

    // automatic detection of kernel size from sigma
    if (ksize.width <= 0 && sigma1 > 0)
        ksize.width = cvRound(sigma1 * (depth == CV_8U ? 3 : 4) * 2 + 1) | 1;
    if (ksize.height <= 0 && sigma2 > 0)
        ksize.height = cvRound(sigma2 * (depth == CV_8U ? 3 : 4) * 2 + 1) | 1;

    CV_Assert(ksize.width > 0 && ksize.width % 2 == 1 && ksize.height > 0 && ksize.height % 2 == 1);

    sigma1 = std::max(sigma1, 0.0);
    sigma2 = std::max(sigma2, 0.0);
    
    const int scaleFactor = 256;

    Mat kx = getGaussianKernel(ksize.width, sigma1, std::max(depth, CV_32F));
    kx.convertTo(kx, kx.depth(), scaleFactor);
    Mat ky;
    if (ksize.height == ksize.width && std::abs(sigma1 - sigma2) < DBL_EPSILON)
        ky = kx;
    else
    {
        ky = getGaussianKernel(ksize.height, sigma2, std::max(depth, CV_32F));        
        ky.convertTo(ky, ky.depth(), scaleFactor);
    }

    GpuMat rowKernel; makeNppKernel(kx, rowKernel);
    GpuMat columnKernel; makeNppKernel(ky, columnKernel);

    applySeparableFilter(src, dst, rowKernel, columnKernel, cv::Point(-1, -1), scaleFactor);
}

#endif
