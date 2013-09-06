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
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

Ptr<Filter> cv::cuda::createBoxFilter(int, int, Size, Point, int, Scalar) { throw_no_cuda(); return Ptr<Filter>(); }

Ptr<Filter> cv::cuda::createLinearFilter(int, int, InputArray, Point, int, Scalar) { throw_no_cuda(); return Ptr<Filter>(); }

Ptr<Filter> cv::cuda::createLaplacianFilter(int, int, int, double, int, Scalar) { throw_no_cuda(); return Ptr<Filter>(); }

Ptr<Filter> cv::cuda::createSeparableLinearFilter(int, int, InputArray, InputArray, Point, int, int) { throw_no_cuda(); return Ptr<Filter>(); }

Ptr<Filter> cv::cuda::createDerivFilter(int, int, int, int, int, bool, double, int, int) { throw_no_cuda(); return Ptr<Filter>(); }
Ptr<Filter> cv::cuda::createSobelFilter(int, int, int, int, int, double, int, int) { throw_no_cuda(); return Ptr<Filter>(); }
Ptr<Filter> cv::cuda::createScharrFilter(int, int, int, int, double, int, int) { throw_no_cuda(); return Ptr<Filter>(); }

Ptr<Filter> cv::cuda::createGaussianFilter(int, int, Size, double, double, int, int) { throw_no_cuda(); return Ptr<Filter>(); }

Ptr<Filter> cv::cuda::createMorphologyFilter(int, int, InputArray, Point, int) { throw_no_cuda(); return Ptr<Filter>(); }

Ptr<Filter> cv::cuda::createBoxMaxFilter(int, Size, Point, int, Scalar) { throw_no_cuda(); return Ptr<Filter>(); }
Ptr<Filter> cv::cuda::createBoxMinFilter(int, Size, Point, int, Scalar) { throw_no_cuda(); return Ptr<Filter>(); }

Ptr<Filter> cv::cuda::createRowSumFilter(int, int, int, int, int, Scalar) { throw_no_cuda(); return Ptr<Filter>(); }
Ptr<Filter> cv::cuda::createColumnSumFilter(int, int, int, int, int, Scalar) { throw_no_cuda(); return Ptr<Filter>(); }

#else

namespace
{
    void normalizeAnchor(int& anchor, int ksize)
    {
        if (anchor < 0)
            anchor = ksize >> 1;

        CV_Assert( 0 <= anchor && anchor < ksize );
    }

    void normalizeAnchor(Point& anchor, Size ksize)
    {
        normalizeAnchor(anchor.x, ksize.width);
        normalizeAnchor(anchor.y, ksize.height);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Box Filter

namespace
{
    class NPPBoxFilter : public Filter
    {
    public:
        NPPBoxFilter(int srcType, int dstType, Size ksize, Point anchor, int borderMode, Scalar borderVal);

        void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

    private:
        typedef NppStatus (*nppFilterBox_t)(const Npp8u* pSrc, Npp32s nSrcStep, Npp8u* pDst, Npp32s nDstStep,
                                            NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor);

        Size ksize_;
        Point anchor_;
        int type_;
        nppFilterBox_t func_;
        int borderMode_;
        Scalar borderVal_;
        GpuMat srcBorder_;
    };

    NPPBoxFilter::NPPBoxFilter(int srcType, int dstType, Size ksize, Point anchor, int borderMode, Scalar borderVal) :
        ksize_(ksize), anchor_(anchor), type_(srcType), borderMode_(borderMode), borderVal_(borderVal)
    {
        static const nppFilterBox_t funcs[] = {0, nppiFilterBox_8u_C1R, 0, 0, nppiFilterBox_8u_C4R};

        CV_Assert( srcType == CV_8UC1 || srcType == CV_8UC4 );
        CV_Assert( dstType == srcType );

        normalizeAnchor(anchor_, ksize);

        func_ = funcs[CV_MAT_CN(srcType)];
    }

    void NPPBoxFilter::apply(InputArray _src, OutputArray _dst, Stream& _stream)
    {
        GpuMat src = _src.getGpuMat();
        CV_Assert( src.type() == type_ );

        cuda::copyMakeBorder(src, srcBorder_, ksize_.height, ksize_.height, ksize_.width, ksize_.width, borderMode_, borderVal_, _stream);

        _dst.create(src.size(), src.type());
        GpuMat dst = _dst.getGpuMat();

        GpuMat srcRoi = srcBorder_(Rect(ksize_.width, ksize_.height, src.cols, src.rows));

        cudaStream_t stream = StreamAccessor::getStream(_stream);
        NppStreamHandler h(stream);

        NppiSize oSizeROI;
        oSizeROI.width = src.cols;
        oSizeROI.height = src.rows;

        NppiSize oMaskSize;
        oMaskSize.height = ksize_.height;
        oMaskSize.width = ksize_.width;

        NppiPoint oAnchor;
        oAnchor.x = anchor_.x;
        oAnchor.y = anchor_.y;

        nppSafeCall( func_(srcRoi.ptr<Npp8u>(), static_cast<int>(srcRoi.step),
                           dst.ptr<Npp8u>(), static_cast<int>(dst.step),
                           oSizeROI, oMaskSize, oAnchor) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

Ptr<Filter> cv::cuda::createBoxFilter(int srcType, int dstType, Size ksize, Point anchor, int borderMode, Scalar borderVal)
{
    if (dstType < 0)
        dstType = srcType;

    dstType = CV_MAKE_TYPE(CV_MAT_DEPTH(dstType), CV_MAT_CN(srcType));

    return makePtr<NPPBoxFilter>(srcType, dstType, ksize, anchor, borderMode, borderVal);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Linear Filter

namespace cv { namespace cuda { namespace device
{
    template <typename T, typename D>
    void filter2D(PtrStepSzb srcWhole, int ofsX, int ofsY, PtrStepSzb dst, const float* kernel,
                  int kWidth, int kHeight, int anchorX, int anchorY,
                  int borderMode, const float* borderValue, cudaStream_t stream);
}}}

namespace
{
    class LinearFilter : public Filter
    {
    public:
        LinearFilter(int srcType, int dstType, InputArray kernel, Point anchor, int borderMode, Scalar borderVal);

        void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

    private:
        typedef void (*filter2D_t)(PtrStepSzb srcWhole, int ofsX, int ofsY, PtrStepSzb dst, const float* kernel,
                                   int kWidth, int kHeight, int anchorX, int anchorY,
                                   int borderMode, const float* borderValue, cudaStream_t stream);

        GpuMat kernel_;
        Point anchor_;
        int type_;
        filter2D_t func_;
        int borderMode_;
        Scalar_<float> borderVal_;
    };

    LinearFilter::LinearFilter(int srcType, int dstType, InputArray _kernel, Point anchor, int borderMode, Scalar borderVal) :
        anchor_(anchor), type_(srcType), borderMode_(borderMode), borderVal_(borderVal)
    {
        const int sdepth = CV_MAT_DEPTH(srcType);
        const int scn = CV_MAT_CN(srcType);

        Mat kernel = _kernel.getMat();

        CV_Assert( sdepth == CV_8U || sdepth == CV_16U || sdepth == CV_32F );
        CV_Assert( scn == 1 || scn == 4 );
        CV_Assert( dstType == srcType );
        CV_Assert( kernel.channels() == 1 );
        CV_Assert( borderMode == BORDER_REFLECT101 || borderMode == BORDER_REPLICATE || borderMode == BORDER_CONSTANT || borderMode == BORDER_REFLECT || borderMode == BORDER_WRAP );

        Mat kernel32F;
        kernel.convertTo(kernel32F, CV_32F);

        kernel_ = cuda::createContinuous(kernel.size(), CV_32FC1);
        kernel_.upload(kernel32F);

        normalizeAnchor(anchor_, kernel.size());

        switch (srcType)
        {
        case CV_8UC1:
            func_ = cv::cuda::device::filter2D<uchar, uchar>;
            break;
        case CV_8UC4:
            func_ = cv::cuda::device::filter2D<uchar4, uchar4>;
            break;
        case CV_16UC1:
            func_ = cv::cuda::device::filter2D<ushort, ushort>;
            break;
        case CV_16UC4:
            func_ = cv::cuda::device::filter2D<ushort4, ushort4>;
            break;
        case CV_32FC1:
            func_ = cv::cuda::device::filter2D<float, float>;
            break;
        case CV_32FC4:
            func_ = cv::cuda::device::filter2D<float4, float4>;
            break;
        }
    }

    void LinearFilter::apply(InputArray _src, OutputArray _dst, Stream& _stream)
    {
        GpuMat src = _src.getGpuMat();
        CV_Assert( src.type() == type_ );

        _dst.create(src.size(), src.type());
        GpuMat dst = _dst.getGpuMat();

        Point ofs;
        Size wholeSize;
        src.locateROI(wholeSize, ofs);

        GpuMat srcWhole(wholeSize, src.type(), src.datastart);

        func_(srcWhole, ofs.x, ofs.y, dst, kernel_.ptr<float>(),
              kernel_.cols, kernel_.rows, anchor_.x, anchor_.y,
              borderMode_, borderVal_.val, StreamAccessor::getStream(_stream));
    }
}

Ptr<Filter> cv::cuda::createLinearFilter(int srcType, int dstType, InputArray kernel, Point anchor, int borderMode, Scalar borderVal)
{
    if (dstType < 0)
        dstType = srcType;

    dstType = CV_MAKE_TYPE(CV_MAT_DEPTH(dstType), CV_MAT_CN(srcType));

    return makePtr<LinearFilter>(srcType, dstType, kernel, anchor, borderMode, borderVal);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Laplacian Filter

Ptr<Filter> cv::cuda::createLaplacianFilter(int srcType, int dstType, int ksize, double scale, int borderMode, Scalar borderVal)
{
    CV_Assert( ksize == 1 || ksize == 3 );

    static const float K[2][9] =
    {
        {0.0f, 1.0f, 0.0f, 1.0f, -4.0f, 1.0f, 0.0f, 1.0f, 0.0f},
        {2.0f, 0.0f, 2.0f, 0.0f, -8.0f, 0.0f, 2.0f, 0.0f, 2.0f}
    };

    Mat kernel(3, 3, CV_32FC1, (void*)K[ksize == 3]);
    if (scale != 1)
        kernel *= scale;

    return cuda::createLinearFilter(srcType, dstType, kernel, Point(-1,-1), borderMode, borderVal);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Separable Linear Filter

namespace filter
{
    template <typename T, typename D>
    void linearRow(PtrStepSzb src, PtrStepSzb dst, const float* kernel, int ksize, int anchor, int brd_type, int cc, cudaStream_t stream);

    template <typename T, typename D>
    void linearColumn(PtrStepSzb src, PtrStepSzb dst, const float* kernel, int ksize, int anchor, int brd_type, int cc, cudaStream_t stream);
}

namespace
{
    class SeparableLinearFilter : public Filter
    {
    public:
        SeparableLinearFilter(int srcType, int dstType,
                              InputArray rowKernel, InputArray columnKernel,
                              Point anchor, int rowBorderMode, int columnBorderMode);

        void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

    private:
        typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, const float* kernel, int ksize, int anchor, int brd_type, int cc, cudaStream_t stream);

        int srcType_, bufType_, dstType_;
        GpuMat rowKernel_, columnKernel_;
        func_t rowFilter_, columnFilter_;
        Point anchor_;
        int rowBorderMode_, columnBorderMode_;

        GpuMat buf_;
    };

    SeparableLinearFilter::SeparableLinearFilter(int srcType, int dstType,
                                                 InputArray _rowKernel, InputArray _columnKernel,
                                                 Point anchor, int rowBorderMode, int columnBorderMode) :
        srcType_(srcType), dstType_(dstType), anchor_(anchor), rowBorderMode_(rowBorderMode), columnBorderMode_(columnBorderMode)
    {
        static const func_t rowFilterFuncs[7][4] =
        {
            {filter::linearRow<uchar, float>, 0, filter::linearRow<uchar3, float3>, filter::linearRow<uchar4, float4>},
            {0, 0, 0, 0},
            {filter::linearRow<ushort, float>, 0, filter::linearRow<ushort3, float3>, filter::linearRow<ushort4, float4>},
            {filter::linearRow<short, float>, 0, filter::linearRow<short3, float3>, filter::linearRow<short4, float4>},
            {filter::linearRow<int, float>, 0, filter::linearRow<int3, float3>, filter::linearRow<int4, float4>},
            {filter::linearRow<float, float>, 0, filter::linearRow<float3, float3>, filter::linearRow<float4, float4>},
            {0, 0, 0, 0}
        };

        static const func_t columnFilterFuncs[7][4] =
        {
            {filter::linearColumn<float, uchar>, 0, filter::linearColumn<float3, uchar3>, filter::linearColumn<float4, uchar4>},
            {0, 0, 0, 0},
            {filter::linearColumn<float, ushort>, 0, filter::linearColumn<float3, ushort3>, filter::linearColumn<float4, ushort4>},
            {filter::linearColumn<float, short>, 0, filter::linearColumn<float3, short3>, filter::linearColumn<float4, short4>},
            {filter::linearColumn<float, int>, 0, filter::linearColumn<float3, int3>, filter::linearColumn<float4, int4>},
            {filter::linearColumn<float, float>, 0, filter::linearColumn<float3, float3>, filter::linearColumn<float4, float4>},
            {0, 0, 0, 0}
        };

        const int sdepth = CV_MAT_DEPTH(srcType);
        const int cn = CV_MAT_CN(srcType);
        const int ddepth = CV_MAT_DEPTH(dstType);

        Mat rowKernel = _rowKernel.getMat();
        Mat columnKernel = _columnKernel.getMat();

        CV_Assert( sdepth <= CV_64F && cn <= 4 );
        CV_Assert( rowKernel.channels() == 1 );
        CV_Assert( columnKernel.channels() == 1 );
        CV_Assert( rowBorderMode == BORDER_REFLECT101 || rowBorderMode == BORDER_REPLICATE || rowBorderMode == BORDER_CONSTANT || rowBorderMode == BORDER_REFLECT || rowBorderMode == BORDER_WRAP );
        CV_Assert( columnBorderMode == BORDER_REFLECT101 || columnBorderMode == BORDER_REPLICATE || columnBorderMode == BORDER_CONSTANT || columnBorderMode == BORDER_REFLECT || columnBorderMode == BORDER_WRAP );

        Mat kernel32F;

        rowKernel.convertTo(kernel32F, CV_32F);
        rowKernel_.upload(kernel32F.reshape(1, 1));

        columnKernel.convertTo(kernel32F, CV_32F);
        columnKernel_.upload(kernel32F.reshape(1, 1));

        CV_Assert( rowKernel_.cols > 0 && rowKernel_.cols <= 32 );
        CV_Assert( columnKernel_.cols > 0 && columnKernel_.cols <= 32 );

        normalizeAnchor(anchor_.x, rowKernel_.cols);
        normalizeAnchor(anchor_.y, columnKernel_.cols);

        bufType_ = CV_MAKE_TYPE(CV_32F, cn);

        rowFilter_ = rowFilterFuncs[sdepth][cn - 1];
        CV_Assert( rowFilter_ != 0 );

        columnFilter_ = columnFilterFuncs[ddepth][cn - 1];
        CV_Assert( columnFilter_ != 0 );
    }

    void SeparableLinearFilter::apply(InputArray _src, OutputArray _dst, Stream& _stream)
    {
        GpuMat src = _src.getGpuMat();
        CV_Assert( src.type() == srcType_ );

        _dst.create(src.size(), dstType_);
        GpuMat dst = _dst.getGpuMat();

        ensureSizeIsEnough(src.size(), bufType_, buf_);

        DeviceInfo devInfo;
        const int cc = devInfo.majorVersion() * 10 + devInfo.minorVersion();

        cudaStream_t stream = StreamAccessor::getStream(_stream);

        rowFilter_(src, buf_, rowKernel_.ptr<float>(), rowKernel_.cols, anchor_.x, rowBorderMode_, cc, stream);
        columnFilter_(buf_, dst, columnKernel_.ptr<float>(), columnKernel_.cols, anchor_.y, columnBorderMode_, cc, stream);
    }
}

Ptr<Filter> cv::cuda::createSeparableLinearFilter(int srcType, int dstType, InputArray rowKernel, InputArray columnKernel, Point anchor, int rowBorderMode, int columnBorderMode)
{
    if (dstType < 0)
        dstType = srcType;

    dstType = CV_MAKE_TYPE(CV_MAT_DEPTH(dstType), CV_MAT_CN(srcType));

    if (columnBorderMode < 0)
        columnBorderMode = rowBorderMode;

    return makePtr<SeparableLinearFilter>(srcType, dstType, rowKernel, columnKernel, anchor, rowBorderMode, columnBorderMode);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Deriv Filter

Ptr<Filter> cv::cuda::createDerivFilter(int srcType, int dstType, int dx, int dy, int ksize, bool normalize, double scale, int rowBorderMode, int columnBorderMode)
{
    Mat kx, ky;
    getDerivKernels(kx, ky, dx, dy, ksize, normalize, CV_32F);

    if (scale != 1)
    {
        // usually the smoothing part is the slowest to compute,
        // so try to scale it instead of the faster differenciating part
        if (dx == 0)
            kx *= scale;
        else
            ky *= scale;
    }

    return cuda::createSeparableLinearFilter(srcType, dstType, kx, ky, Point(-1, -1), rowBorderMode, columnBorderMode);
}

Ptr<Filter> cv::cuda::createSobelFilter(int srcType, int dstType, int dx, int dy, int ksize, double scale, int rowBorderMode, int columnBorderMode)
{
    return cuda::createDerivFilter(srcType, dstType, dx, dy, ksize, false, scale, rowBorderMode, columnBorderMode);
}

Ptr<Filter> cv::cuda::createScharrFilter(int srcType, int dstType, int dx, int dy, double scale, int rowBorderMode, int columnBorderMode)
{
    return cuda::createDerivFilter(srcType, dstType, dx, dy, -1, false, scale, rowBorderMode, columnBorderMode);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Gaussian Filter

Ptr<Filter> cv::cuda::createGaussianFilter(int srcType, int dstType, Size ksize, double sigma1, double sigma2, int rowBorderMode, int columnBorderMode)
{
    const int depth = CV_MAT_DEPTH(srcType);

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

    Mat kx = getGaussianKernel(ksize.width, sigma1, CV_32F);
    Mat ky;
    if (ksize.height == ksize.width && std::abs(sigma1 - sigma2) < DBL_EPSILON)
        ky = kx;
    else
        ky = getGaussianKernel(ksize.height, sigma2, CV_32F);

    return createSeparableLinearFilter(srcType, dstType, kx, ky, Point(-1,-1), rowBorderMode, columnBorderMode);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Morphology Filter

namespace
{
    class MorphologyFilter : public Filter
    {
    public:
        MorphologyFilter(int op, int srcType, InputArray kernel, Point anchor, int iterations);

        void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

    private:
        typedef NppStatus (*nppMorfFilter_t)(const Npp8u* pSrc, Npp32s nSrcStep, Npp8u* pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                             const Npp8u* pMask, NppiSize oMaskSize, NppiPoint oAnchor);

        int type_;
        GpuMat kernel_;
        Point anchor_;
        int iters_;
        nppMorfFilter_t func_;

        GpuMat srcBorder_;
        GpuMat buf_;
    };

    MorphologyFilter::MorphologyFilter(int op, int srcType, InputArray _kernel, Point anchor, int iterations) :
        type_(srcType), anchor_(anchor), iters_(iterations)
    {
        static const nppMorfFilter_t funcs[2][5] =
        {
            {0, nppiErode_8u_C1R, 0, 0, nppiErode_8u_C4R },
            {0, nppiDilate_8u_C1R, 0, 0, nppiDilate_8u_C4R }
        };

        CV_Assert( op == MORPH_ERODE || op == MORPH_DILATE );
        CV_Assert( srcType == CV_8UC1 || srcType == CV_8UC4 );

        Mat kernel = _kernel.getMat();
        Size ksize = !kernel.empty() ? _kernel.size() : Size(3, 3);

        normalizeAnchor(anchor_, ksize);

        if (kernel.empty())
        {
            kernel = getStructuringElement(MORPH_RECT, Size(1 + iters_ * 2, 1 + iters_ * 2));
            anchor_ = Point(iters_, iters_);
            iters_ = 1;
        }
        else if (iters_ > 1 && countNonZero(kernel) == (int) kernel.total())
        {
            anchor_ = Point(anchor_.x * iters_, anchor_.y * iters_);
            kernel = getStructuringElement(MORPH_RECT,
                                           Size(ksize.width + (iters_ - 1) * (ksize.width - 1),
                                                ksize.height + (iters_ - 1) * (ksize.height - 1)),
                                           anchor_);
            iters_ = 1;
        }

        CV_Assert( kernel.channels() == 1 );

        Mat kernel8U;
        kernel.convertTo(kernel8U, CV_8U);

        kernel_ = cuda::createContinuous(kernel.size(), CV_8UC1);
        kernel_.upload(kernel8U);

        func_ = funcs[op][CV_MAT_CN(srcType)];
    }

    void MorphologyFilter::apply(InputArray _src, OutputArray _dst, Stream& _stream)
    {
        GpuMat src = _src.getGpuMat();
        CV_Assert( src.type() == type_ );

        Size ksize = kernel_.size();
        cuda::copyMakeBorder(src, srcBorder_, ksize.height, ksize.height, ksize.width, ksize.width, BORDER_DEFAULT, Scalar(), _stream);

        GpuMat srcRoi = srcBorder_(Rect(ksize.width, ksize.height, src.cols, src.rows));

        GpuMat bufRoi;
        if (iters_ > 1)
        {
            ensureSizeIsEnough(srcBorder_.size(), type_, buf_);
            buf_.setTo(Scalar::all(0), _stream);
            bufRoi = buf_(Rect(ksize.width, ksize.height, src.cols, src.rows));
        }

        _dst.create(src.size(), src.type());
        GpuMat dst = _dst.getGpuMat();

        cudaStream_t stream = StreamAccessor::getStream(_stream);
        NppStreamHandler h(stream);

        NppiSize oSizeROI;
        oSizeROI.width = src.cols;
        oSizeROI.height = src.rows;

        NppiSize oMaskSize;
        oMaskSize.height = ksize.height;
        oMaskSize.width = ksize.width;

        NppiPoint oAnchor;
        oAnchor.x = anchor_.x;
        oAnchor.y = anchor_.y;

        nppSafeCall( func_(srcRoi.ptr<Npp8u>(), static_cast<int>(srcRoi.step), dst.ptr<Npp8u>(), static_cast<int>(dst.step),
                           oSizeROI, kernel_.ptr<Npp8u>(), oMaskSize, oAnchor) );

        for(int i = 1; i < iters_; ++i)
        {
            dst.copyTo(bufRoi, _stream);

            nppSafeCall( func_(bufRoi.ptr<Npp8u>(), static_cast<int>(bufRoi.step), dst.ptr<Npp8u>(), static_cast<int>(dst.step),
                               oSizeROI, kernel_.ptr<Npp8u>(), oMaskSize, oAnchor) );
        }

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

namespace
{
    class MorphologyExFilter : public Filter
    {
    public:
        MorphologyExFilter(int srcType, InputArray kernel, Point anchor, int iterations);

    protected:
        Ptr<cuda::Filter> erodeFilter_, dilateFilter_;
        GpuMat buf_;
    };

    MorphologyExFilter::MorphologyExFilter(int srcType, InputArray kernel, Point anchor, int iterations)
    {
        erodeFilter_ = cuda::createMorphologyFilter(MORPH_ERODE, srcType, kernel, anchor, iterations);
        dilateFilter_ = cuda::createMorphologyFilter(MORPH_DILATE, srcType, kernel, anchor, iterations);
    }

    // MORPH_OPEN

    class MorphologyOpenFilter : public MorphologyExFilter
    {
    public:
        MorphologyOpenFilter(int srcType, InputArray kernel, Point anchor, int iterations);

        void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null());
    };

    MorphologyOpenFilter::MorphologyOpenFilter(int srcType, InputArray kernel, Point anchor, int iterations) :
        MorphologyExFilter(srcType, kernel, anchor, iterations)
    {
    }

    void MorphologyOpenFilter::apply(InputArray src, OutputArray dst, Stream& stream)
    {
        erodeFilter_->apply(src, buf_, stream);
        dilateFilter_->apply(buf_, dst, stream);
    }

    // MORPH_CLOSE

    class MorphologyCloseFilter : public MorphologyExFilter
    {
    public:
        MorphologyCloseFilter(int srcType, InputArray kernel, Point anchor, int iterations);

        void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null());
    };

    MorphologyCloseFilter::MorphologyCloseFilter(int srcType, InputArray kernel, Point anchor, int iterations) :
        MorphologyExFilter(srcType, kernel, anchor, iterations)
    {
    }

    void MorphologyCloseFilter::apply(InputArray src, OutputArray dst, Stream& stream)
    {
        dilateFilter_->apply(src, buf_, stream);
        erodeFilter_->apply(buf_, dst, stream);
    }

    // MORPH_GRADIENT

    class MorphologyGradientFilter : public MorphologyExFilter
    {
    public:
        MorphologyGradientFilter(int srcType, InputArray kernel, Point anchor, int iterations);

        void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null());
    };

    MorphologyGradientFilter::MorphologyGradientFilter(int srcType, InputArray kernel, Point anchor, int iterations) :
        MorphologyExFilter(srcType, kernel, anchor, iterations)
    {
    }

    void MorphologyGradientFilter::apply(InputArray src, OutputArray dst, Stream& stream)
    {
        erodeFilter_->apply(src, buf_, stream);
        dilateFilter_->apply(src, dst, stream);
        cuda::subtract(dst, buf_, dst, noArray(), -1, stream);
    }

    // MORPH_TOPHAT

    class MorphologyTophatFilter : public MorphologyExFilter
    {
    public:
        MorphologyTophatFilter(int srcType, InputArray kernel, Point anchor, int iterations);

        void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null());
    };

    MorphologyTophatFilter::MorphologyTophatFilter(int srcType, InputArray kernel, Point anchor, int iterations) :
        MorphologyExFilter(srcType, kernel, anchor, iterations)
    {
    }

    void MorphologyTophatFilter::apply(InputArray src, OutputArray dst, Stream& stream)
    {
        erodeFilter_->apply(src, dst, stream);
        dilateFilter_->apply(dst, buf_, stream);
        cuda::subtract(src, buf_, dst, noArray(), -1, stream);
    }

    // MORPH_BLACKHAT

    class MorphologyBlackhatFilter : public MorphologyExFilter
    {
    public:
        MorphologyBlackhatFilter(int srcType, InputArray kernel, Point anchor, int iterations);

        void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null());
    };

    MorphologyBlackhatFilter::MorphologyBlackhatFilter(int srcType, InputArray kernel, Point anchor, int iterations) :
        MorphologyExFilter(srcType, kernel, anchor, iterations)
    {
    }

    void MorphologyBlackhatFilter::apply(InputArray src, OutputArray dst, Stream& stream)
    {
        dilateFilter_->apply(src, dst, stream);
        erodeFilter_->apply(dst, buf_, stream);
        cuda::subtract(buf_, src, dst, noArray(), -1, stream);
    }
}

Ptr<Filter> cv::cuda::createMorphologyFilter(int op, int srcType, InputArray kernel, Point anchor, int iterations)
{
    switch( op )
    {
    case MORPH_ERODE:
    case MORPH_DILATE:
        return makePtr<MorphologyFilter>(op, srcType, kernel, anchor, iterations);
        break;

    case MORPH_OPEN:
        return makePtr<MorphologyOpenFilter>(srcType, kernel, anchor, iterations);
        break;

    case MORPH_CLOSE:
        return makePtr<MorphologyCloseFilter>(srcType, kernel, anchor, iterations);
        break;

    case MORPH_GRADIENT:
        return makePtr<MorphologyGradientFilter>(srcType, kernel, anchor, iterations);
        break;

    case MORPH_TOPHAT:
        return makePtr<MorphologyTophatFilter>(srcType, kernel, anchor, iterations);
        break;

    case MORPH_BLACKHAT:
        return makePtr<MorphologyBlackhatFilter>(srcType, kernel, anchor, iterations);
        break;

    default:
        CV_Error(Error::StsBadArg, "Unknown morphological operation");
        return Ptr<Filter>();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Image Rank Filter

namespace
{
    enum RankType
    {
        RANK_MAX,
        RANK_MIN
    };

    class NPPRankFilter : public Filter
    {
    public:
        NPPRankFilter(int op, int srcType, Size ksize, Point anchor, int borderMode, Scalar borderVal);

        void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

    private:
        typedef NppStatus (*nppFilterRank_t)(const Npp8u* pSrc, Npp32s nSrcStep, Npp8u* pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                             NppiSize oMaskSize, NppiPoint oAnchor);

        int type_;
        Size ksize_;
        Point anchor_;
        int borderMode_;
        Scalar borderVal_;
        nppFilterRank_t func_;

        GpuMat srcBorder_;
    };

    NPPRankFilter::NPPRankFilter(int op, int srcType, Size ksize, Point anchor, int borderMode, Scalar borderVal) :
        type_(srcType), ksize_(ksize), anchor_(anchor), borderMode_(borderMode), borderVal_(borderVal)
    {
        static const nppFilterRank_t maxFuncs[] = {0, nppiFilterMax_8u_C1R, 0, 0, nppiFilterMax_8u_C4R};
        static const nppFilterRank_t minFuncs[] = {0, nppiFilterMin_8u_C1R, 0, 0, nppiFilterMin_8u_C4R};

        CV_Assert( srcType == CV_8UC1 || srcType == CV_8UC4 );

        normalizeAnchor(anchor_, ksize_);

        if (op == RANK_MAX)
            func_ = maxFuncs[CV_MAT_CN(srcType)];
        else
            func_ = minFuncs[CV_MAT_CN(srcType)];
    }

    void NPPRankFilter::apply(InputArray _src, OutputArray _dst, Stream& _stream)
    {
        GpuMat src = _src.getGpuMat();
        CV_Assert( src.type() == type_ );

        cuda::copyMakeBorder(src, srcBorder_, ksize_.height, ksize_.height, ksize_.width, ksize_.width, borderMode_, borderVal_, _stream);

        _dst.create(src.size(), src.type());
        GpuMat dst = _dst.getGpuMat();

        GpuMat srcRoi = srcBorder_(Rect(ksize_.width, ksize_.height, src.cols, src.rows));

        cudaStream_t stream = StreamAccessor::getStream(_stream);
        NppStreamHandler h(stream);

        NppiSize oSizeROI;
        oSizeROI.width = src.cols;
        oSizeROI.height = src.rows;

        NppiSize oMaskSize;
        oMaskSize.height = ksize_.height;
        oMaskSize.width = ksize_.width;

        NppiPoint oAnchor;
        oAnchor.x = anchor_.x;
        oAnchor.y = anchor_.y;

        nppSafeCall( func_(srcRoi.ptr<Npp8u>(), static_cast<int>(srcRoi.step), dst.ptr<Npp8u>(), static_cast<int>(dst.step),
                           oSizeROI, oMaskSize, oAnchor) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

Ptr<Filter> cv::cuda::createBoxMaxFilter(int srcType, Size ksize, Point anchor, int borderMode, Scalar borderVal)
{
    return makePtr<NPPRankFilter>(RANK_MAX, srcType, ksize, anchor, borderMode, borderVal);
}

Ptr<Filter> cv::cuda::createBoxMinFilter(int srcType, Size ksize, Point anchor, int borderMode, Scalar borderVal)
{
    return makePtr<NPPRankFilter>(RANK_MIN, srcType, ksize, anchor, borderMode, borderVal);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 1D Sum Filter

namespace
{
    class NppRowSumFilter : public Filter
    {
    public:
        NppRowSumFilter(int srcType, int dstType, int ksize, int anchor, int borderMode, Scalar borderVal);

        void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

    private:
        int srcType_, dstType_;
        int ksize_;
        int anchor_;
        int borderMode_;
        Scalar borderVal_;

        GpuMat srcBorder_;
    };

    NppRowSumFilter::NppRowSumFilter(int srcType, int dstType, int ksize, int anchor, int borderMode, Scalar borderVal) :
        srcType_(srcType), dstType_(dstType), ksize_(ksize), anchor_(anchor), borderMode_(borderMode), borderVal_(borderVal)
    {
        CV_Assert( srcType_ == CV_8UC1 );
        CV_Assert( dstType_ == CV_32FC1 );

        normalizeAnchor(anchor_, ksize_);
    }

    void NppRowSumFilter::apply(InputArray _src, OutputArray _dst, Stream& _stream)
    {
        GpuMat src = _src.getGpuMat();
        CV_Assert( src.type() == srcType_ );

        cuda::copyMakeBorder(src, srcBorder_, 0, 0, ksize_, ksize_, borderMode_, borderVal_, _stream);

        _dst.create(src.size(), dstType_);
        GpuMat dst = _dst.getGpuMat();

        GpuMat srcRoi = srcBorder_(Rect(ksize_, 0, src.cols, src.rows));

        cudaStream_t stream = StreamAccessor::getStream(_stream);
        NppStreamHandler h(stream);

        NppiSize oSizeROI;
        oSizeROI.width = src.cols;
        oSizeROI.height = src.rows;

        nppSafeCall( nppiSumWindowRow_8u32f_C1R(srcRoi.ptr<Npp8u>(), static_cast<int>(srcRoi.step),
                                                dst.ptr<Npp32f>(), static_cast<int>(dst.step),
                                                oSizeROI, ksize_, anchor_) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

Ptr<Filter> cv::cuda::createRowSumFilter(int srcType, int dstType, int ksize, int anchor, int borderMode, Scalar borderVal)
{
    return makePtr<NppRowSumFilter>(srcType, dstType, ksize, anchor, borderMode, borderVal);
}

namespace
{
    class NppColumnSumFilter : public Filter
    {
    public:
        NppColumnSumFilter(int srcType, int dstType, int ksize, int anchor, int borderMode, Scalar borderVal);

        void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null());

    private:
        int srcType_, dstType_;
        int ksize_;
        int anchor_;
        int borderMode_;
        Scalar borderVal_;

        GpuMat srcBorder_;
    };

    NppColumnSumFilter::NppColumnSumFilter(int srcType, int dstType, int ksize, int anchor, int borderMode, Scalar borderVal) :
        srcType_(srcType), dstType_(dstType), ksize_(ksize), anchor_(anchor), borderMode_(borderMode), borderVal_(borderVal)
    {
        CV_Assert( srcType_ == CV_8UC1 );
        CV_Assert( dstType_ == CV_32FC1 );

        normalizeAnchor(anchor_, ksize_);
    }

    void NppColumnSumFilter::apply(InputArray _src, OutputArray _dst, Stream& _stream)
    {
        GpuMat src = _src.getGpuMat();
        CV_Assert( src.type() == srcType_ );

        cuda::copyMakeBorder(src, srcBorder_, ksize_, ksize_, 0, 0, borderMode_, borderVal_, _stream);

        _dst.create(src.size(), dstType_);
        GpuMat dst = _dst.getGpuMat();

        GpuMat srcRoi = srcBorder_(Rect(0, ksize_, src.cols, src.rows));

        cudaStream_t stream = StreamAccessor::getStream(_stream);
        NppStreamHandler h(stream);

        NppiSize oSizeROI;
        oSizeROI.width = src.cols;
        oSizeROI.height = src.rows;

        nppSafeCall( nppiSumWindowColumn_8u32f_C1R(srcRoi.ptr<Npp8u>(), static_cast<int>(srcRoi.step),
                                                   dst.ptr<Npp32f>(), static_cast<int>(dst.step),
                                                   oSizeROI, ksize_, anchor_) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

Ptr<Filter> cv::cuda::createColumnSumFilter(int srcType, int dstType, int ksize, int anchor, int borderMode, Scalar borderVal)
{
    return makePtr<NppColumnSumFilter>(srcType, dstType, ksize, anchor, borderMode, borderVal);
}

#endif
