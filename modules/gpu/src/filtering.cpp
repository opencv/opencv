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

Ptr<FilterEngine_GPU> cv::gpu::createFilter2D_GPU(const Ptr<BaseFilter_GPU>) { throw_nogpu(); return Ptr<FilterEngine_GPU>(0); }
Ptr<FilterEngine_GPU> cv::gpu::createSeparableFilter_GPU(const Ptr<BaseRowFilter_GPU>&, const Ptr<BaseColumnFilter_GPU>&) { throw_nogpu(); return Ptr<FilterEngine_GPU>(0); }
Ptr<BaseRowFilter_GPU> cv::gpu::getRowSumFilter_GPU(int, int, int, int) { throw_nogpu(); return Ptr<BaseRowFilter_GPU>(0); }
Ptr<BaseColumnFilter_GPU> cv::gpu::getColumnSumFilter_GPU(int, int, int, int) { throw_nogpu(); return Ptr<BaseColumnFilter_GPU>(0); }
Ptr<BaseFilter_GPU> cv::gpu::getBoxFilter_GPU(int, int, const Size&, Point) { throw_nogpu(); return Ptr<BaseFilter_GPU>(0); }
Ptr<FilterEngine_GPU> cv::gpu::createBoxFilter_GPU(int, int, const Size&, const Point&) { throw_nogpu(); return Ptr<FilterEngine_GPU>(0); }
Ptr<BaseFilter_GPU> cv::gpu::getMorphologyFilter_GPU(int, int, const Mat&, const Size&, Point) { throw_nogpu(); return Ptr<BaseFilter_GPU>(0); }
Ptr<FilterEngine_GPU> cv::gpu::createMorphologyFilter_GPU(int, int, const Mat&, const Point&, int) { throw_nogpu(); return Ptr<FilterEngine_GPU>(0); }
Ptr<BaseFilter_GPU> cv::gpu::getLinearFilter_GPU(int, int, const Mat&, const Size&, Point) { throw_nogpu(); return Ptr<BaseFilter_GPU>(0); }
Ptr<FilterEngine_GPU> cv::gpu::createLinearFilter_GPU(int, int, const Mat&, const Point&) { throw_nogpu(); return Ptr<FilterEngine_GPU>(0); }
Ptr<BaseRowFilter_GPU> cv::gpu::getLinearRowFilter_GPU(int, int, const Mat&, int) { throw_nogpu(); return Ptr<BaseRowFilter_GPU>(0); }
Ptr<BaseColumnFilter_GPU> cv::gpu::getLinearColumnFilter_GPU(int, int, const Mat&, int) { throw_nogpu(); return Ptr<BaseColumnFilter_GPU>(0); }
Ptr<FilterEngine_GPU> cv::gpu::createSeparableLinearFilter_GPU(int, int, const Mat&, const Mat&, const Point&) { throw_nogpu(); return Ptr<FilterEngine_GPU>(0); }
Ptr<FilterEngine_GPU> cv::gpu::createDerivFilter_GPU(int, int, int, int, int) { throw_nogpu(); return Ptr<FilterEngine_GPU>(0); }
Ptr<FilterEngine_GPU> cv::gpu::createGaussianFilter_GPU(int, Size, double, double) { throw_nogpu(); return Ptr<FilterEngine_GPU>(0); }
Ptr<BaseFilter_GPU> cv::gpu::getMaxFilter_GPU(int, int, const Size&, Point) { throw_nogpu(); return Ptr<BaseFilter_GPU>(0); }
Ptr<BaseFilter_GPU> cv::gpu::getMinFilter_GPU(int, int, const Size&, Point) { throw_nogpu(); return Ptr<BaseFilter_GPU>(0); }

void cv::gpu::boxFilter(const GpuMat&, GpuMat&, int, Size, Point) { throw_nogpu(); }
void cv::gpu::erode( const GpuMat&, GpuMat&, const Mat&, Point, int) { throw_nogpu(); }
void cv::gpu::dilate( const GpuMat&, GpuMat&, const Mat&, Point, int) { throw_nogpu(); }
void cv::gpu::morphologyEx( const GpuMat&, GpuMat&, int, const Mat&, Point, int) { throw_nogpu(); }
void cv::gpu::filter2D(const GpuMat&, GpuMat&, int, const Mat&, Point) { throw_nogpu(); }
void cv::gpu::sepFilter2D(const GpuMat&, GpuMat&, int, const Mat&, const Mat&, Point) { throw_nogpu(); }
void cv::gpu::Sobel(const GpuMat&, GpuMat&, int, int, int, int, double) { throw_nogpu(); }
void cv::gpu::Scharr(const GpuMat&, GpuMat&, int, int, int, double) { throw_nogpu(); }
void cv::gpu::GaussianBlur(const GpuMat&, GpuMat&, Size, double, double) { throw_nogpu(); }
void cv::gpu::Laplacian(const GpuMat&, GpuMat&, int, int, double) { throw_nogpu(); }

#else

namespace
{
    inline void normalizeAnchor(int& anchor, int ksize)
    {
        if (anchor < 0)
            anchor = ksize >> 1;

        CV_Assert(0 <= anchor && anchor < ksize);
    }

    inline void normalizeAnchor(Point& anchor, const Size& ksize)
    {
        normalizeAnchor(anchor.x, ksize.width);
        normalizeAnchor(anchor.y, ksize.height);
    }

    inline void normalizeROI(Rect& roi, const Size& ksize, const Point& anchor, const Size& src_size)
    {
        if (roi == Rect(0,0,-1,-1))
            roi = Rect(anchor.x, anchor.y, src_size.width - ksize.width, src_size.height - ksize.height);

        CV_Assert(roi.x >= 0 && roi.y >= 0 && roi.width <= src_size.width && roi.height <= src_size.height);
    }

    inline void normalizeKernel(const Mat& kernel, GpuMat& gpu_krnl, int type = CV_8U, int* nDivisor = 0, bool reverse = false)
    {
        int scale = nDivisor && (kernel.depth() == CV_32F || kernel.depth() == CV_64F) ? 256 : 1;
        if (nDivisor) *nDivisor = scale;
        
        Mat temp(kernel.size(), type);
        kernel.convertTo(temp, type, scale);
        Mat cont_krnl = temp.reshape(1, 1);

        if (reverse)
        {
            int count = cont_krnl.cols >> 1;
            for (int i = 0; i < count; ++i)
            {
                std::swap(cont_krnl.at<int>(0, i), cont_krnl.at<int>(0, cont_krnl.cols - 1 - i));
            }
        }

        gpu_krnl.upload(cont_krnl);
    } 
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Filter2D

namespace
{
    class Filter2DEngine_GPU : public FilterEngine_GPU
    {
    public:
        Filter2DEngine_GPU(const Ptr<BaseFilter_GPU>& filter2D_) : filter2D(filter2D_) {}

        virtual void apply(const GpuMat& src, GpuMat& dst, Rect roi = Rect(0,0,-1,-1))
        {
            Size src_size = src.size();

            dst.create(src_size, src.type());
            dst = Scalar(0.0);

            normalizeROI(roi, filter2D->ksize, filter2D->anchor, src_size);

            GpuMat srcROI = src(roi);
            GpuMat dstROI = dst(roi);

            (*filter2D)(srcROI, dstROI);
        }

        Ptr<BaseFilter_GPU> filter2D;
    };
}

Ptr<FilterEngine_GPU> cv::gpu::createFilter2D_GPU(const Ptr<BaseFilter_GPU> filter2D)
{
    return Ptr<FilterEngine_GPU>(new Filter2DEngine_GPU(filter2D));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// SeparableFilter

namespace
{
    class SeparableFilterEngine_GPU : public FilterEngine_GPU
    {
    public:
        SeparableFilterEngine_GPU(const Ptr<BaseRowFilter_GPU>& rowFilter_, 
                                       const Ptr<BaseColumnFilter_GPU>& columnFilter_) :
            rowFilter(rowFilter_), columnFilter(columnFilter_)
        {
            ksize = Size(rowFilter->ksize, columnFilter->ksize);
            anchor = Point(rowFilter->anchor, columnFilter->anchor);
        }

        virtual void apply(const GpuMat& src, GpuMat& dst, Rect roi = Rect(0,0,-1,-1))
        {
            Size src_size = src.size();
            int src_type = src.type();

            dst.create(src_size, src_type);
            dst = Scalar(0.0);
            dstBuf.create(src_size, src_type);
            dstBuf = Scalar(0.0);

            normalizeROI(roi, ksize, anchor, src_size);

            srcROI = src(roi);
            dstROI = dst(roi);
            dstBufROI = dstBuf(roi);
            
            (*rowFilter)(srcROI, dstBufROI);
            (*columnFilter)(dstBufROI, dstROI);
        }

        Ptr<BaseRowFilter_GPU> rowFilter;
        Ptr<BaseColumnFilter_GPU> columnFilter;
        Size ksize;
        Point anchor;
        GpuMat dstBuf;
        GpuMat srcROI;
        GpuMat dstROI;
        GpuMat dstBufROI;
    };
}

Ptr<FilterEngine_GPU> cv::gpu::createSeparableFilter_GPU(const Ptr<BaseRowFilter_GPU>& rowFilter, 
    const Ptr<BaseColumnFilter_GPU>& columnFilter)
{
    return Ptr<FilterEngine_GPU>(new SeparableFilterEngine_GPU(rowFilter, columnFilter));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 1D Sum Filter

namespace
{
    class NppRowSumFilter : public BaseRowFilter_GPU
    {
    public:
        NppRowSumFilter(int ksize_, int anchor_) : BaseRowFilter_GPU(ksize_, anchor_) {}

        virtual void operator()(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            nppSafeCall( nppiSumWindowRow_8u32f_C1R(src.ptr<Npp8u>(), src.step, dst.ptr<Npp32f>(), dst.step, sz, ksize, anchor) );
        }
    };
}

Ptr<BaseRowFilter_GPU> cv::gpu::getRowSumFilter_GPU(int srcType, int sumType, int ksize, int anchor)
{
    CV_Assert(srcType == CV_8UC1 && sumType == CV_32FC1);

    normalizeAnchor(anchor, ksize);

    return Ptr<BaseRowFilter_GPU>(new NppRowSumFilter(ksize, anchor));
}

namespace
{
    class NppColumnSumFilter : public BaseColumnFilter_GPU
    {
    public:
        NppColumnSumFilter(int ksize_, int anchor_) : BaseColumnFilter_GPU(ksize_, anchor_) {}

        virtual void operator()(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            nppSafeCall( nppiSumWindowColumn_8u32f_C1R(src.ptr<Npp8u>(), src.step, dst.ptr<Npp32f>(), dst.step, sz, ksize, anchor) );
        }
    };
}

Ptr<BaseColumnFilter_GPU> cv::gpu::getColumnSumFilter_GPU(int sumType, int dstType, int ksize, int anchor)
{
    CV_Assert(sumType == CV_8UC1 && dstType == CV_32FC1);

    normalizeAnchor(anchor, ksize);

    return Ptr<BaseColumnFilter_GPU>(new NppColumnSumFilter(ksize, anchor));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Box Filter

namespace
{
    typedef NppStatus (*nppFilterBox_t)(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
        NppiSize oMaskSize, NppiPoint oAnchor);

    class NPPBoxFilter : public BaseFilter_GPU
    {
    public:
        NPPBoxFilter(const Size& ksize_, const Point& anchor_, nppFilterBox_t func_) : BaseFilter_GPU(ksize_, anchor_), func(func_) {}

        virtual void operator()(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;
            NppiSize oKernelSize;
            oKernelSize.height = ksize.height;
            oKernelSize.width = ksize.width;
            NppiPoint oAnchor;
            oAnchor.x = anchor.x;
            oAnchor.y = anchor.y;
            
            nppSafeCall( func(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz, oKernelSize, oAnchor) );
        }

        nppFilterBox_t func;
    };
}

Ptr<BaseFilter_GPU> cv::gpu::getBoxFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor)
{
    static const nppFilterBox_t nppFilterBox_callers[] = {0, nppiFilterBox_8u_C1R, 0, 0, nppiFilterBox_8u_C4R};

    CV_Assert((srcType == CV_8UC1 || srcType == CV_8UC4) && dstType == srcType); 

    normalizeAnchor(anchor, ksize);

    return Ptr<BaseFilter_GPU>(new NPPBoxFilter(ksize, anchor, nppFilterBox_callers[CV_MAT_CN(srcType)]));
}

Ptr<FilterEngine_GPU> cv::gpu::createBoxFilter_GPU(int srcType, int dstType, const Size& ksize, const Point& anchor)
{
    Ptr<BaseFilter_GPU> boxFilter = getBoxFilter_GPU(srcType, dstType, ksize, anchor);
    return createFilter2D_GPU(boxFilter);
}

void cv::gpu::boxFilter(const GpuMat& src, GpuMat& dst, int ddepth, Size ksize, Point anchor)
{
    int sdepth = src.depth(), cn = src.channels();
    if( ddepth < 0 )
        ddepth = sdepth;

    dst.create(src.size(), CV_MAKETYPE(ddepth, cn));

    Ptr<FilterEngine_GPU> f = createBoxFilter_GPU(src.type(), dst.type(), ksize, anchor);
    f->apply(src, dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Morphology Filter

namespace
{
    typedef NppStatus (*nppMorfFilter_t)(const Npp8u*, Npp32s, Npp8u*, Npp32s, NppiSize, const Npp8u*, NppiSize, NppiPoint);

    class NPPMorphFilter : public BaseFilter_GPU
    {
    public:
        NPPMorphFilter(const Size& ksize_, const Point& anchor_, const GpuMat& kernel_, nppMorfFilter_t func_) : 
            BaseFilter_GPU(ksize_, anchor_), kernel(kernel_), func(func_) {}

        virtual void operator()(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;
            NppiSize oKernelSize;
            oKernelSize.height = ksize.height;
            oKernelSize.width = ksize.width;
            NppiPoint oAnchor;
            oAnchor.x = anchor.x;
            oAnchor.y = anchor.y;

            nppSafeCall( func(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz, kernel.ptr<Npp8u>(), oKernelSize, oAnchor) );
        }

        GpuMat kernel;
        nppMorfFilter_t func;
    };
}

Ptr<BaseFilter_GPU> cv::gpu::getMorphologyFilter_GPU(int op, int type, const Mat& kernel, const Size& ksize, Point anchor)
{
    static const nppMorfFilter_t nppMorfFilter_callers[2][5] = 
    {
        {0, nppiErode_8u_C1R, 0, 0, nppiErode_8u_C4R },
        {0, nppiDilate_8u_C1R, 0, 0, nppiDilate_8u_C4R }
    };
 
    CV_Assert(op == MORPH_ERODE || op == MORPH_DILATE);   
    CV_Assert(type == CV_8UC1 || type == CV_8UC4); 
        
    GpuMat gpu_krnl;
    normalizeKernel(kernel, gpu_krnl);
    normalizeAnchor(anchor, ksize);
    
    return Ptr<BaseFilter_GPU>(new NPPMorphFilter(ksize, anchor, gpu_krnl, nppMorfFilter_callers[op][CV_MAT_CN(type)]));
}

namespace
{
    class MorphologyFilterEngine_GPU : public Filter2DEngine_GPU
    {
    public:
        MorphologyFilterEngine_GPU(const Ptr<BaseFilter_GPU>& filter2D_, int iters_) : 
          Filter2DEngine_GPU(filter2D_), iters(iters_) {}

        virtual void apply(const GpuMat& src, GpuMat& dst, Rect roi = Rect(0,0,-1,-1))
        {
            if (iters > 1)
                morfBuf.create(src.size(), src.type());

            Filter2DEngine_GPU::apply(src, dst);
            for(int i = 1; i < iters; ++i)
            {
                dst.swap(morfBuf);
                Filter2DEngine_GPU::apply(morfBuf, dst);
            }
        }

        int iters;
        GpuMat morfBuf;
    };
}

Ptr<FilterEngine_GPU> cv::gpu::createMorphologyFilter_GPU(int op, int type, const Mat& kernel, const Point& anchor, int iterations)
{
    CV_Assert(iterations > 0);

    Size ksize = kernel.size();

    Ptr<BaseFilter_GPU> filter2D = getMorphologyFilter_GPU(op, type, kernel, ksize, anchor);

    return Ptr<FilterEngine_GPU>(new MorphologyFilterEngine_GPU(filter2D, iterations));
}

namespace
{
    void morphOp(int op, const GpuMat& src, GpuMat& dst, const Mat& _kernel, Point anchor, int iterations)
    {
        Mat kernel;
        Size ksize = _kernel.data ? _kernel.size() : Size(3, 3);

        normalizeAnchor(anchor, ksize);

        if (iterations == 0 || _kernel.rows * _kernel.cols == 1)
        {
            src.copyTo(dst);
            return;
        }

        dst.create(src.size(), src.type());

        if (!_kernel.data)
        {
            kernel = getStructuringElement(MORPH_RECT, Size(1 + iterations * 2, 1 + iterations * 2));
            anchor = Point(iterations, iterations);
            iterations = 1;
        }
        else if (iterations > 1 && countNonZero(_kernel) == _kernel.rows * _kernel.cols)
        {
            anchor = Point(anchor.x * iterations, anchor.y * iterations);
            kernel = getStructuringElement(MORPH_RECT, Size(ksize.width + iterations * (ksize.width - 1), 
                ksize.height + iterations * (ksize.height - 1)), anchor);
            iterations = 1;
        }
        else
            kernel = _kernel;

        Ptr<FilterEngine_GPU> f = createMorphologyFilter_GPU(op, src.type(), kernel, anchor, iterations);

        f->apply(src, dst);
    }
}

void cv::gpu::erode( const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor, int iterations)
{
    morphOp(MORPH_ERODE, src, dst, kernel, anchor, iterations);
}

void cv::gpu::dilate( const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor, int iterations)
{
    morphOp(MORPH_DILATE, src, dst, kernel, anchor, iterations);
}

void cv::gpu::morphologyEx( const GpuMat& src, GpuMat& dst, int op, const Mat& kernel, Point anchor, int iterations)
{
    GpuMat temp;
    switch( op )
    {
    case MORPH_ERODE:   erode( src, dst, kernel, anchor, iterations); break;    
    case MORPH_DILATE: dilate( src, dst, kernel, anchor, iterations); break;    
    case MORPH_OPEN:
        erode( src, temp, kernel, anchor, iterations);
        dilate( temp, dst, kernel, anchor, iterations);
        break;
    case CV_MOP_CLOSE:
        dilate( src, temp, kernel, anchor, iterations);
         erode( temp, dst, kernel, anchor, iterations);
        break;
    case CV_MOP_GRADIENT:
         erode( src, temp, kernel, anchor, iterations);
        dilate( src, dst, kernel, anchor, iterations);        
        subtract(dst, temp, dst);
        break;
    case CV_MOP_TOPHAT:
        erode( src, dst, kernel, anchor, iterations);
        dilate( dst, temp, kernel, anchor, iterations);        
        subtract(src, temp, dst);
        break;
    case CV_MOP_BLACKHAT:
        dilate( src, dst, kernel, anchor, iterations);
        erode( dst, temp, kernel, anchor, iterations);
        subtract(temp, src, dst);
        break;
    default:
        CV_Error( CV_StsBadArg, "unknown morphological operation" );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Linear Filter

namespace
{
    typedef NppStatus (*nppFilter2D_t)(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
        const Npp32s * pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor);

    class NPPLinearFilter : public BaseFilter_GPU
    {
    public:
        NPPLinearFilter(const Size& ksize_, const Point& anchor_, const GpuMat& kernel_, Npp32s nDivisor_, nppFilter2D_t func_) : 
            BaseFilter_GPU(ksize_, anchor_), kernel(kernel_), nDivisor(nDivisor_), func(func_) {}

        virtual void operator()(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;
            NppiSize oKernelSize;
            oKernelSize.height = ksize.height;
            oKernelSize.width = ksize.width;
            NppiPoint oAnchor;
            oAnchor.x = anchor.x;
            oAnchor.y = anchor.y;
                                  
            nppSafeCall( func(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz, 
                kernel.ptr<Npp32s>(), oKernelSize, oAnchor, nDivisor) );
        }

        GpuMat kernel;
        Npp32s nDivisor;
        nppFilter2D_t func;
    };
}

Ptr<BaseFilter_GPU> cv::gpu::getLinearFilter_GPU(int srcType, int dstType, const Mat& kernel, const Size& ksize, Point anchor)
{
    static const nppFilter2D_t cppFilter2D_callers[] = {0, nppiFilter_8u_C1R, 0, 0, nppiFilter_8u_C4R};

    CV_Assert((srcType == CV_8UC1 || srcType == CV_8UC4) && dstType == srcType);
    
    GpuMat gpu_krnl;
    int nDivisor;
    normalizeKernel(kernel, gpu_krnl, CV_32S, &nDivisor, true);
    normalizeAnchor(anchor, ksize);

    return Ptr<BaseFilter_GPU>(new NPPLinearFilter(ksize, anchor, gpu_krnl, nDivisor, cppFilter2D_callers[CV_MAT_CN(srcType)]));
}    

Ptr<FilterEngine_GPU> cv::gpu::createLinearFilter_GPU(int srcType, int dstType, const Mat& kernel, const Point& anchor)
{
    Size ksize = kernel.size();

    Ptr<BaseFilter_GPU> linearFilter = getLinearFilter_GPU(srcType, dstType, kernel, ksize, anchor);

    return createFilter2D_GPU(linearFilter);
}

void cv::gpu::filter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernel, Point anchor)
{
    if( ddepth < 0 )
        ddepth = src.depth();

    dst.create(src.size(), CV_MAKETYPE(ddepth, src.channels()));

    Ptr<FilterEngine_GPU> f = createLinearFilter_GPU(src.type(), dst.type(), kernel, anchor);
    f->apply(src, dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Separable Linear Filter

namespace cv { namespace gpu { namespace filters
{
    void linearRowFilter_gpu_8u_8u_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearRowFilter_gpu_8u_8s_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearRowFilter_gpu_8s_8u_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearRowFilter_gpu_8s_8s_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearRowFilter_gpu_16u_16u_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearRowFilter_gpu_16u_16s_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearRowFilter_gpu_16s_16u_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearRowFilter_gpu_16s_16s_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearRowFilter_gpu_32s_32s_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearRowFilter_gpu_32s_32f_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearRowFilter_gpu_32f_32s_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearRowFilter_gpu_32f_32f_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);

    void linearColumnFilter_gpu_8u_8u_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearColumnFilter_gpu_8u_8s_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearColumnFilter_gpu_8s_8u_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearColumnFilter_gpu_8s_8s_c4(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearColumnFilter_gpu_16u_16u_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearColumnFilter_gpu_16u_16s_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearColumnFilter_gpu_16s_16u_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearColumnFilter_gpu_16s_16s_c2(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearColumnFilter_gpu_32s_32s_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearColumnFilter_gpu_32s_32f_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearColumnFilter_gpu_32f_32s_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
    void linearColumnFilter_gpu_32f_32f_c1(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);
}}}

namespace
{
    typedef NppStatus (*nppFilter1D_t)(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oROI, 
        const Npp32s * pKernel, Npp32s nMaskSize, Npp32s nAnchor, Npp32s nDivisor);

    typedef void (*gpuFilter1D_t)(const DevMem2D& src, const DevMem2D& dst, const float kernel[], int ksize, int anchor);

    class NppLinearRowFilter : public BaseRowFilter_GPU
    {
    public:
        NppLinearRowFilter(int ksize_, int anchor_, const GpuMat& kernel_, Npp32s nDivisor_, nppFilter1D_t func_) : 
            BaseRowFilter_GPU(ksize_, anchor_), kernel(kernel_), nDivisor(nDivisor_), func(func_) {}

        virtual void operator()(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            nppSafeCall( func(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz, kernel.ptr<Npp32s>(), ksize, anchor, nDivisor) );
        }

        GpuMat kernel;
        Npp32s nDivisor;
        nppFilter1D_t func;
    };

    class GpuLinearRowFilter : public BaseRowFilter_GPU
    {
    public:
        GpuLinearRowFilter(int ksize_, int anchor_, const Mat& kernel_, gpuFilter1D_t func_) : 
            BaseRowFilter_GPU(ksize_, anchor_), kernel(kernel_), func(func_) {}

        virtual void operator()(const GpuMat& src, GpuMat& dst)
        {
            func(src, dst, kernel.ptr<float>(), ksize, anchor);
        }

        Mat kernel;
        gpuFilter1D_t func;
    };
}

Ptr<BaseRowFilter_GPU> cv::gpu::getLinearRowFilter_GPU(int srcType, int bufType, const Mat& rowKernel, int anchor)
{
    using namespace cv::gpu::filters;
    static const nppFilter1D_t nppFilter1D_callers[] = {0, nppiFilterRow_8u_C1R, 0, 0, nppiFilterRow_8u_C4R};
    static const gpuFilter1D_t gpuFilter1D_callers[6][6] =
    {
        {linearRowFilter_gpu_8u_8u_c4,linearRowFilter_gpu_8u_8s_c4,0,0,0,0},
        {linearRowFilter_gpu_8s_8u_c4,linearRowFilter_gpu_8s_8s_c4,0,0,0,0},
        {0,0,linearRowFilter_gpu_16u_16u_c2,linearRowFilter_gpu_16u_16s_c2,0,0},
        {0,0,linearRowFilter_gpu_16s_16u_c2,linearRowFilter_gpu_16s_16s_c2,0,0},
        {0,0,0,0,linearRowFilter_gpu_32s_32s_c1, linearRowFilter_gpu_32s_32f_c1},
        {0,0,0,0,linearRowFilter_gpu_32f_32s_c1, linearRowFilter_gpu_32f_32f_c1}
    };
    
    if ((bufType == srcType) && (srcType == CV_8UC1 || srcType == CV_8UC4))
    {
        GpuMat gpu_row_krnl;
        int nDivisor;
        normalizeKernel(rowKernel, gpu_row_krnl, CV_32S, &nDivisor, true);

        int ksize = gpu_row_krnl.cols;
        normalizeAnchor(anchor, ksize);

        return Ptr<BaseRowFilter_GPU>(new NppLinearRowFilter(ksize, anchor, gpu_row_krnl, nDivisor,
            nppFilter1D_callers[CV_MAT_CN(srcType)]));
    }

    CV_Assert(srcType == CV_8UC4 || srcType == CV_8SC4 || srcType == CV_16UC2 || srcType == CV_16SC2 || srcType == CV_32SC1 || srcType == CV_32FC1);
    CV_Assert(bufType == CV_8UC4 || bufType == CV_8SC4 || bufType == CV_16UC2 || bufType == CV_16SC2 || bufType == CV_32SC1 || bufType == CV_32FC1);

    Mat temp(rowKernel.size(), CV_32FC1);
    rowKernel.convertTo(temp, CV_32FC1);
    Mat cont_krnl = temp.reshape(1, 1);

    int ksize = cont_krnl.cols;
    normalizeAnchor(anchor, ksize);

    return Ptr<BaseRowFilter_GPU>(new GpuLinearRowFilter(ksize, anchor, cont_krnl, 
        gpuFilter1D_callers[CV_MAT_DEPTH(srcType)][CV_MAT_DEPTH(bufType)]));
}

namespace
{
    class NppLinearColumnFilter : public BaseColumnFilter_GPU
    {
    public:
        NppLinearColumnFilter(int ksize_, int anchor_, const GpuMat& kernel_, Npp32s nDivisor_, nppFilter1D_t func_) : 
            BaseColumnFilter_GPU(ksize_, anchor_), kernel(kernel_), nDivisor(nDivisor_), func(func_) {}

        virtual void operator()(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            nppSafeCall( func(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz, kernel.ptr<Npp32s>(), ksize, anchor, nDivisor) );
        }

        GpuMat kernel;
        Npp32s nDivisor;
        nppFilter1D_t func;
    };

    class GpuLinearColumnFilter : public BaseColumnFilter_GPU
    {
    public:
        GpuLinearColumnFilter(int ksize_, int anchor_, const Mat& kernel_, gpuFilter1D_t func_) : 
            BaseColumnFilter_GPU(ksize_, anchor_), kernel(kernel_), func(func_) {}

        virtual void operator()(const GpuMat& src, GpuMat& dst)
        {
            func(src, dst, kernel.ptr<float>(), ksize, anchor);
        }

        Mat kernel;
        gpuFilter1D_t func;
    };
}

Ptr<BaseColumnFilter_GPU> cv::gpu::getLinearColumnFilter_GPU(int bufType, int dstType, const Mat& columnKernel, int anchor)
{
    using namespace cv::gpu::filters;
    static const nppFilter1D_t nppFilter1D_callers[] = {0, nppiFilterColumn_8u_C1R, 0, 0, nppiFilterColumn_8u_C4R};
    static const gpuFilter1D_t gpuFilter1D_callers[6][6] =
    {
        {linearColumnFilter_gpu_8u_8u_c4,linearColumnFilter_gpu_8u_8s_c4,0,0,0,0},
        {linearColumnFilter_gpu_8s_8u_c4,linearColumnFilter_gpu_8s_8s_c4,0,0,0,0},
        {0,0,linearColumnFilter_gpu_16u_16u_c2,linearColumnFilter_gpu_16u_16s_c2,0,0},
        {0,0,linearColumnFilter_gpu_16s_16u_c2,linearColumnFilter_gpu_16s_16s_c2,0,0},
        {0,0,0,0,linearColumnFilter_gpu_32s_32s_c1, linearColumnFilter_gpu_32s_32f_c1},
        {0,0,0,0,linearColumnFilter_gpu_32f_32s_c1, linearColumnFilter_gpu_32f_32f_c1}
    };
    
    double kernelMin;
    minMaxLoc(columnKernel, &kernelMin);
    
    if ((bufType == dstType) && (bufType == CV_8UC1 || bufType == CV_8UC4))
    {
        GpuMat gpu_col_krnl;
        int nDivisor;
        normalizeKernel(columnKernel, gpu_col_krnl, CV_32S, &nDivisor, true);

        int ksize = gpu_col_krnl.cols;
        normalizeAnchor(anchor, ksize);

        return Ptr<BaseColumnFilter_GPU>(new NppLinearColumnFilter(ksize, anchor, gpu_col_krnl, nDivisor, 
            nppFilter1D_callers[CV_MAT_CN(bufType)]));
    }

    CV_Assert(dstType == CV_8UC4 || dstType == CV_8SC4 || dstType == CV_16UC2 || dstType == CV_16SC2 || dstType == CV_32SC1 || dstType == CV_32FC1);
    CV_Assert(bufType == CV_8UC4 || bufType == CV_8SC4 || bufType == CV_16UC2 || bufType == CV_16SC2 || bufType == CV_32SC1 || bufType == CV_32FC1);

    Mat temp(columnKernel.size(), CV_32FC1);
    columnKernel.convertTo(temp, CV_32FC1);
    Mat cont_krnl = temp.reshape(1, 1);

    int ksize = cont_krnl.cols;
    normalizeAnchor(anchor, ksize);

    return Ptr<BaseColumnFilter_GPU>(new GpuLinearColumnFilter(ksize, anchor, cont_krnl, 
        gpuFilter1D_callers[CV_MAT_DEPTH(bufType)][CV_MAT_DEPTH(dstType)]));
}

Ptr<FilterEngine_GPU> cv::gpu::createSeparableLinearFilter_GPU(int srcType, int dstType, const Mat& rowKernel, const Mat& columnKernel, 
    const Point& anchor)
{
    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(dstType);
    int cn = CV_MAT_CN(srcType);
    int bdepth = std::max(sdepth, ddepth);
    int bufType = CV_MAKETYPE(bdepth, cn);

    Ptr<BaseRowFilter_GPU> rowFilter = getLinearRowFilter_GPU(srcType, bufType, rowKernel, anchor.x);
    Ptr<BaseColumnFilter_GPU> columnFilter = getLinearColumnFilter_GPU(bufType, dstType, columnKernel, anchor.y);

    return createSeparableFilter_GPU(rowFilter, columnFilter);
}

void cv::gpu::sepFilter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernelX, const Mat& kernelY, Point anchor)
{
    if( ddepth < 0 )
        ddepth = src.depth();

    dst.create(src.size(), CV_MAKETYPE(ddepth, src.channels()));

    Ptr<FilterEngine_GPU> f = createSeparableLinearFilter_GPU(src.type(), dst.type(), kernelX, kernelY, anchor);
    f->apply(src, dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Deriv Filter

Ptr<FilterEngine_GPU> cv::gpu::createDerivFilter_GPU(int srcType, int dstType, int dx, int dy, int ksize)
{
    Mat kx, ky;
    getDerivKernels(kx, ky, dx, dy, ksize, false, CV_32F);
    return createSeparableLinearFilter_GPU(srcType, dstType, kx, ky, Point(-1,-1));
}

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
    
    sepFilter2D(src, dst, ddepth, kx, ky, Point(-1,-1));
}

void cv::gpu::Scharr(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, double scale)
{
    Mat kx, ky;
    getDerivKernels(kx, ky, dx, dy, -1, false, CV_32F);

    if( scale != 1 )
    {
        // usually the smoothing part is the slowest to compute,
        // so try to scale it instead of the faster differenciating part
        if( dx == 0 )
            kx *= scale;
        else
            ky *= scale;
    }

    sepFilter2D(src, dst, ddepth, kx, ky, Point(-1,-1));
}

void cv::gpu::Laplacian(const GpuMat& src, GpuMat& dst, int ddepth, int ksize, double scale)
{
    CV_Assert(ksize == 1 || ksize == 3);

    static const int K[2][9] =
    {
        {0, 1, 0, 1, -4, 1, 0, 1, 0},
        {2, 0, 2, 0, -8, 0, 2, 0, 2}
    };
    Mat kernel(3, 3, CV_32S, (void*)K[ksize == 3]);
    if (scale != 1)
        kernel *= scale;
    
    filter2D(src, dst, ddepth, kernel, Point(-1,-1));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Gaussian Filter

Ptr<FilterEngine_GPU> cv::gpu::createGaussianFilter_GPU(int type, Size ksize, double sigma1, double sigma2)
{        
    int depth = CV_MAT_DEPTH(type);

    if (sigma2 <= 0)
        sigma2 = sigma1;

    // automatic detection of kernel size from sigma
    if (ksize.width <= 0 && sigma1 > 0)
        ksize.width = cvRound(sigma1 * (depth == CV_8U ? 3 : 4)*2 + 1) | 1;
    if (ksize.height <= 0 && sigma2 > 0)
        ksize.height = cvRound(sigma2 * (depth == CV_8U ? 3 : 4)*2 + 1) | 1;

    CV_Assert( ksize.width > 0 && ksize.width % 2 == 1 && ksize.height > 0 && ksize.height % 2 == 1 );

    sigma1 = std::max(sigma1, 0.0);
    sigma2 = std::max(sigma2, 0.0);

    Mat kx = getGaussianKernel( ksize.width, sigma1, std::max(depth, CV_32F) );
    Mat ky;
    if( ksize.height == ksize.width && std::abs(sigma1 - sigma2) < DBL_EPSILON )
        ky = kx;
    else
        ky = getGaussianKernel( ksize.height, sigma2, std::max(depth, CV_32F) );

    return createSeparableLinearFilter_GPU(type, type, kx, ky);
}

void cv::gpu::GaussianBlur(const GpuMat& src, GpuMat& dst, Size ksize, double sigma1, double sigma2)
{
    if (ksize.width == 1 && ksize.height == 1)
    {
        src.copyTo(dst);
        return;
    }

    dst.create(src.size(), src.type());
    
    Ptr<FilterEngine_GPU> f = createGaussianFilter_GPU(src.type(), ksize, sigma1, sigma2);
    f->apply(src, dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Image Rank Filter

namespace
{
    typedef NppStatus (*nppFilterRank_t)(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
        NppiSize oMaskSize, NppiPoint oAnchor);

    class NPPRankFilter : public BaseFilter_GPU
    {
    public:
        NPPRankFilter(const Size& ksize_, const Point& anchor_, nppFilterRank_t func_) : BaseFilter_GPU(ksize_, anchor_), func(func_) {}

        virtual void operator()(const GpuMat& src, GpuMat& dst)
        {
            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;
            NppiSize oKernelSize;
            oKernelSize.height = ksize.height;
            oKernelSize.width = ksize.width;
            NppiPoint oAnchor;
            oAnchor.x = anchor.x;
            oAnchor.y = anchor.y;
            
            nppSafeCall( func(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz, oKernelSize, oAnchor) );
        }

        nppFilterRank_t func;
    };
}

Ptr<BaseFilter_GPU> cv::gpu::getMaxFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor)
{
    static const nppFilterRank_t nppFilterRank_callers[] = {0, nppiFilterMax_8u_C1R, 0, 0, nppiFilterMax_8u_C4R};

    CV_Assert((srcType == CV_8UC1 || srcType == CV_8UC4) && dstType == srcType); 

    normalizeAnchor(anchor, ksize);

    return Ptr<BaseFilter_GPU>(new NPPRankFilter(ksize, anchor, nppFilterRank_callers[CV_MAT_CN(srcType)]));
}

Ptr<BaseFilter_GPU> cv::gpu::getMinFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor)
{
    static const nppFilterRank_t nppFilterRank_callers[] = {0, nppiFilterMin_8u_C1R, 0, 0, nppiFilterMin_8u_C4R};

    CV_Assert((srcType == CV_8UC1 || srcType == CV_8UC4) && dstType == srcType); 

    normalizeAnchor(anchor, ksize);

    return Ptr<BaseFilter_GPU>(new NPPRankFilter(ksize, anchor, nppFilterRank_callers[CV_MAT_CN(srcType)]));
}

#endif
