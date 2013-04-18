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

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::gpu::evenLevels(GpuMat&, int, int, int) { throw_no_cuda(); }

void cv::gpu::histEven(const GpuMat&, GpuMat&, int, int, int, Stream&) { throw_no_cuda(); }
void cv::gpu::histEven(const GpuMat&, GpuMat&, GpuMat&, int, int, int, Stream&) { throw_no_cuda(); }
void cv::gpu::histEven(const GpuMat&, GpuMat*, int*, int*, int*, Stream&) { throw_no_cuda(); }
void cv::gpu::histEven(const GpuMat&, GpuMat*, GpuMat&, int*, int*, int*, Stream&) { throw_no_cuda(); }

void cv::gpu::histRange(const GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_no_cuda(); }
void cv::gpu::histRange(const GpuMat&, GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_no_cuda(); }
void cv::gpu::histRange(const GpuMat&, GpuMat*, const GpuMat*, Stream&) { throw_no_cuda(); }
void cv::gpu::histRange(const GpuMat&, GpuMat*, const GpuMat*, GpuMat&, Stream&) { throw_no_cuda(); }

void cv::gpu::calcHist(const GpuMat&, GpuMat&, Stream&) { throw_no_cuda(); }

void cv::gpu::equalizeHist(const GpuMat&, GpuMat&, Stream&) { throw_no_cuda(); }
void cv::gpu::equalizeHist(const GpuMat&, GpuMat&, GpuMat&, GpuMat&, Stream&) { throw_no_cuda(); }

cv::Ptr<cv::gpu::CLAHE> cv::gpu::createCLAHE(double, cv::Size) { throw_no_cuda(); return cv::Ptr<cv::gpu::CLAHE>(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// NPP Histogram

namespace
{
    typedef NppStatus (*get_buf_size_c1_t)(NppiSize oSizeROI, int nLevels, int* hpBufferSize);
    typedef NppStatus (*get_buf_size_c4_t)(NppiSize oSizeROI, int nLevels[], int* hpBufferSize);

    template<int SDEPTH> struct NppHistogramEvenFuncC1
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

    typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s * pHist,
            int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, Npp8u * pBuffer);
    };
    template<int SDEPTH> struct NppHistogramEvenFuncC4
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI,
            Npp32s * pHist[4], int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4], Npp8u * pBuffer);
    };

    template<int SDEPTH, typename NppHistogramEvenFuncC1<SDEPTH>::func_ptr func, get_buf_size_c1_t get_buf_size>
    struct NppHistogramEvenC1
    {
        typedef typename NppHistogramEvenFuncC1<SDEPTH>::src_t src_t;

        static void hist(const GpuMat& src, GpuMat& hist, GpuMat& buffer, int histSize, int lowerLevel, int upperLevel, cudaStream_t stream)
        {
            int levels = histSize + 1;
            hist.create(1, histSize, CV_32S);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            int buf_size;
            get_buf_size(sz, levels, &buf_size);

            ensureSizeIsEnough(1, buf_size, CV_8U, buffer);

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, hist.ptr<Npp32s>(), levels,
                lowerLevel, upperLevel, buffer.ptr<Npp8u>()) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppHistogramEvenFuncC4<SDEPTH>::func_ptr func, get_buf_size_c4_t get_buf_size>
    struct NppHistogramEvenC4
    {
        typedef typename NppHistogramEvenFuncC4<SDEPTH>::src_t src_t;

        static void hist(const GpuMat& src, GpuMat hist[4], GpuMat& buffer, int histSize[4], int lowerLevel[4], int upperLevel[4], cudaStream_t stream)
        {
            int levels[] = {histSize[0] + 1, histSize[1] + 1, histSize[2] + 1, histSize[3] + 1};
            hist[0].create(1, histSize[0], CV_32S);
            hist[1].create(1, histSize[1], CV_32S);
            hist[2].create(1, histSize[2], CV_32S);
            hist[3].create(1, histSize[3], CV_32S);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Npp32s* pHist[] = {hist[0].ptr<Npp32s>(), hist[1].ptr<Npp32s>(), hist[2].ptr<Npp32s>(), hist[3].ptr<Npp32s>()};

            int buf_size;
            get_buf_size(sz, levels, &buf_size);

            ensureSizeIsEnough(1, buf_size, CV_8U, buffer);

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, pHist, levels, lowerLevel, upperLevel, buffer.ptr<Npp8u>()) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };

    template<int SDEPTH> struct NppHistogramRangeFuncC1
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
        typedef Npp32s level_t;
        enum {LEVEL_TYPE_CODE=CV_32SC1};

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist,
            const Npp32s* pLevels, int nLevels, Npp8u* pBuffer);
    };
    template<> struct NppHistogramRangeFuncC1<CV_32F>
    {
        typedef Npp32f src_t;
        typedef Npp32f level_t;
        enum {LEVEL_TYPE_CODE=CV_32FC1};

        typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist,
            const Npp32f* pLevels, int nLevels, Npp8u* pBuffer);
    };
    template<int SDEPTH> struct NppHistogramRangeFuncC4
    {
        typedef typename NPPTypeTraits<SDEPTH>::npp_type src_t;
        typedef Npp32s level_t;
        enum {LEVEL_TYPE_CODE=CV_32SC1};

        typedef NppStatus (*func_ptr)(const src_t* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist[4],
            const Npp32s* pLevels[4], int nLevels[4], Npp8u* pBuffer);
    };
    template<> struct NppHistogramRangeFuncC4<CV_32F>
    {
        typedef Npp32f src_t;
        typedef Npp32f level_t;
        enum {LEVEL_TYPE_CODE=CV_32FC1};

        typedef NppStatus (*func_ptr)(const Npp32f* pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s* pHist[4],
            const Npp32f* pLevels[4], int nLevels[4], Npp8u* pBuffer);
    };

    template<int SDEPTH, typename NppHistogramRangeFuncC1<SDEPTH>::func_ptr func, get_buf_size_c1_t get_buf_size>
    struct NppHistogramRangeC1
    {
        typedef typename NppHistogramRangeFuncC1<SDEPTH>::src_t src_t;
        typedef typename NppHistogramRangeFuncC1<SDEPTH>::level_t level_t;
        enum {LEVEL_TYPE_CODE=NppHistogramRangeFuncC1<SDEPTH>::LEVEL_TYPE_CODE};

        static void hist(const GpuMat& src, GpuMat& hist, const GpuMat& levels, GpuMat& buffer, cudaStream_t stream)
        {
            CV_Assert(levels.type() == LEVEL_TYPE_CODE && levels.rows == 1);

            hist.create(1, levels.cols - 1, CV_32S);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            int buf_size;
            get_buf_size(sz, levels.cols, &buf_size);

            ensureSizeIsEnough(1, buf_size, CV_8U, buffer);

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, hist.ptr<Npp32s>(), levels.ptr<level_t>(), levels.cols, buffer.ptr<Npp8u>()) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppHistogramRangeFuncC4<SDEPTH>::func_ptr func, get_buf_size_c4_t get_buf_size>
    struct NppHistogramRangeC4
    {
        typedef typename NppHistogramRangeFuncC4<SDEPTH>::src_t src_t;
        typedef typename NppHistogramRangeFuncC1<SDEPTH>::level_t level_t;
        enum {LEVEL_TYPE_CODE=NppHistogramRangeFuncC1<SDEPTH>::LEVEL_TYPE_CODE};

        static void hist(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], GpuMat& buffer, cudaStream_t stream)
        {
            CV_Assert(levels[0].type() == LEVEL_TYPE_CODE && levels[0].rows == 1);
            CV_Assert(levels[1].type() == LEVEL_TYPE_CODE && levels[1].rows == 1);
            CV_Assert(levels[2].type() == LEVEL_TYPE_CODE && levels[2].rows == 1);
            CV_Assert(levels[3].type() == LEVEL_TYPE_CODE && levels[3].rows == 1);

            hist[0].create(1, levels[0].cols - 1, CV_32S);
            hist[1].create(1, levels[1].cols - 1, CV_32S);
            hist[2].create(1, levels[2].cols - 1, CV_32S);
            hist[3].create(1, levels[3].cols - 1, CV_32S);

            Npp32s* pHist[] = {hist[0].ptr<Npp32s>(), hist[1].ptr<Npp32s>(), hist[2].ptr<Npp32s>(), hist[3].ptr<Npp32s>()};
            int nLevels[] = {levels[0].cols, levels[1].cols, levels[2].cols, levels[3].cols};
            const level_t* pLevels[] = {levels[0].ptr<level_t>(), levels[1].ptr<level_t>(), levels[2].ptr<level_t>(), levels[3].ptr<level_t>()};

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            int buf_size;
            get_buf_size(sz, nLevels, &buf_size);

            ensureSizeIsEnough(1, buf_size, CV_8U, buffer);

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, pHist, pLevels, nLevels, buffer.ptr<Npp8u>()) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

void cv::gpu::evenLevels(GpuMat& levels, int nLevels, int lowerLevel, int upperLevel)
{
    Mat host_levels(1, nLevels, CV_32SC1);
    nppSafeCall( nppiEvenLevelsHost_32s(host_levels.ptr<Npp32s>(), nLevels, lowerLevel, upperLevel) );
    levels.upload(host_levels);
}

void cv::gpu::histEven(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel, Stream& stream)
{
    GpuMat buf;
    histEven(src, hist, buf, histSize, lowerLevel, upperLevel, stream);
}

void cv::gpu::histEven(const GpuMat& src, GpuMat& hist, GpuMat& buf, int histSize, int lowerLevel, int upperLevel, Stream& stream)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_16UC1 || src.type() == CV_16SC1 );

    typedef void (*hist_t)(const GpuMat& src, GpuMat& hist, GpuMat& buf, int levels, int lowerLevel, int upperLevel, cudaStream_t stream);
    static const hist_t hist_callers[] =
    {
        NppHistogramEvenC1<CV_8U , nppiHistogramEven_8u_C1R , nppiHistogramEvenGetBufferSize_8u_C1R >::hist,
        0,
        NppHistogramEvenC1<CV_16U, nppiHistogramEven_16u_C1R, nppiHistogramEvenGetBufferSize_16u_C1R>::hist,
        NppHistogramEvenC1<CV_16S, nppiHistogramEven_16s_C1R, nppiHistogramEvenGetBufferSize_16s_C1R>::hist
    };

    hist_callers[src.depth()](src, hist, buf, histSize, lowerLevel, upperLevel, StreamAccessor::getStream(stream));
}

void cv::gpu::histEven(const GpuMat& src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream)
{
    GpuMat buf;
    histEven(src, hist, buf, histSize, lowerLevel, upperLevel, stream);
}

void cv::gpu::histEven(const GpuMat& src, GpuMat hist[4], GpuMat& buf, int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream)
{
    CV_Assert(src.type() == CV_8UC4 || src.type() == CV_16UC4 || src.type() == CV_16SC4 );

    typedef void (*hist_t)(const GpuMat& src, GpuMat hist[4], GpuMat& buf, int levels[4], int lowerLevel[4], int upperLevel[4], cudaStream_t stream);
    static const hist_t hist_callers[] =
    {
        NppHistogramEvenC4<CV_8U , nppiHistogramEven_8u_C4R , nppiHistogramEvenGetBufferSize_8u_C4R >::hist,
        0,
        NppHistogramEvenC4<CV_16U, nppiHistogramEven_16u_C4R, nppiHistogramEvenGetBufferSize_16u_C4R>::hist,
        NppHistogramEvenC4<CV_16S, nppiHistogramEven_16s_C4R, nppiHistogramEvenGetBufferSize_16s_C4R>::hist
    };

    hist_callers[src.depth()](src, hist, buf, histSize, lowerLevel, upperLevel, StreamAccessor::getStream(stream));
}

void cv::gpu::histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, Stream& stream)
{
    GpuMat buf;
    histRange(src, hist, levels, buf, stream);
}

void cv::gpu::histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, GpuMat& buf, Stream& stream)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_16UC1 || src.type() == CV_16SC1 || src.type() == CV_32FC1);

    typedef void (*hist_t)(const GpuMat& src, GpuMat& hist, const GpuMat& levels, GpuMat& buf, cudaStream_t stream);
    static const hist_t hist_callers[] =
    {
        NppHistogramRangeC1<CV_8U , nppiHistogramRange_8u_C1R , nppiHistogramRangeGetBufferSize_8u_C1R >::hist,
        0,
        NppHistogramRangeC1<CV_16U, nppiHistogramRange_16u_C1R, nppiHistogramRangeGetBufferSize_16u_C1R>::hist,
        NppHistogramRangeC1<CV_16S, nppiHistogramRange_16s_C1R, nppiHistogramRangeGetBufferSize_16s_C1R>::hist,
        0,
        NppHistogramRangeC1<CV_32F, nppiHistogramRange_32f_C1R, nppiHistogramRangeGetBufferSize_32f_C1R>::hist
    };

    hist_callers[src.depth()](src, hist, levels, buf, StreamAccessor::getStream(stream));
}

void cv::gpu::histRange(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], Stream& stream)
{
    GpuMat buf;
    histRange(src, hist, levels, buf, stream);
}

void cv::gpu::histRange(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], GpuMat& buf, Stream& stream)
{
    CV_Assert(src.type() == CV_8UC4 || src.type() == CV_16UC4 || src.type() == CV_16SC4 || src.type() == CV_32FC4);

    typedef void (*hist_t)(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], GpuMat& buf, cudaStream_t stream);
    static const hist_t hist_callers[] =
    {
        NppHistogramRangeC4<CV_8U , nppiHistogramRange_8u_C4R , nppiHistogramRangeGetBufferSize_8u_C4R >::hist,
        0,
        NppHistogramRangeC4<CV_16U, nppiHistogramRange_16u_C4R, nppiHistogramRangeGetBufferSize_16u_C4R>::hist,
        NppHistogramRangeC4<CV_16S, nppiHistogramRange_16s_C4R, nppiHistogramRangeGetBufferSize_16s_C4R>::hist,
        0,
        NppHistogramRangeC4<CV_32F, nppiHistogramRange_32f_C4R, nppiHistogramRangeGetBufferSize_32f_C4R>::hist
    };

    hist_callers[src.depth()](src, hist, levels, buf, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// calcHist

namespace hist
{
    void histogram256(PtrStepSzb src, int* hist, cudaStream_t stream);
}

void cv::gpu::calcHist(const GpuMat& src, GpuMat& hist, Stream& stream)
{
    CV_Assert(src.type() == CV_8UC1);

    hist.create(1, 256, CV_32SC1);
    hist.setTo(Scalar::all(0));

    hist::histogram256(src, hist.ptr<int>(), StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// equalizeHist

namespace hist
{
    void equalizeHist(PtrStepSzb src, PtrStepSzb dst, const int* lut, cudaStream_t stream);
}

void cv::gpu::equalizeHist(const GpuMat& src, GpuMat& dst, Stream& stream)
{
    GpuMat hist;
    GpuMat buf;
    equalizeHist(src, dst, hist, buf, stream);
}

void cv::gpu::equalizeHist(const GpuMat& src, GpuMat& dst, GpuMat& hist, GpuMat& buf, Stream& s)
{
    CV_Assert(src.type() == CV_8UC1);

    dst.create(src.size(), src.type());

    int intBufSize;
    nppSafeCall( nppsIntegralGetBufferSize_32s(256, &intBufSize) );

    ensureSizeIsEnough(1, intBufSize + 256 * sizeof(int), CV_8UC1, buf);

    GpuMat intBuf(1, intBufSize, CV_8UC1, buf.ptr());
    GpuMat lut(1, 256, CV_32S, buf.ptr() + intBufSize);

    calcHist(src, hist, s);

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStreamHandler h(stream);

    nppSafeCall( nppsIntegral_32s(hist.ptr<Npp32s>(), lut.ptr<Npp32s>(), 256, intBuf.ptr<Npp8u>()) );

    hist::equalizeHist(src, dst, lut.ptr<int>(), stream);
}

////////////////////////////////////////////////////////////////////////
// CLAHE

namespace clahe
{
    void calcLut(PtrStepSzb src, PtrStepb lut, int tilesX, int tilesY, int2 tileSize, int clipLimit, float lutScale, cudaStream_t stream);
    void transform(PtrStepSzb src, PtrStepSzb dst, PtrStepb lut, int tilesX, int tilesY, int2 tileSize, cudaStream_t stream);
}

namespace
{
    class CLAHE_Impl : public cv::gpu::CLAHE
    {
    public:
        CLAHE_Impl(double clipLimit = 40.0, int tilesX = 8, int tilesY = 8);

        cv::AlgorithmInfo* info() const;

        void apply(cv::InputArray src, cv::OutputArray dst);
        void apply(InputArray src, OutputArray dst, Stream& stream);

        void setClipLimit(double clipLimit);
        double getClipLimit() const;

        void setTilesGridSize(cv::Size tileGridSize);
        cv::Size getTilesGridSize() const;

        void collectGarbage();

    private:
        double clipLimit_;
        int tilesX_;
        int tilesY_;

        GpuMat srcExt_;
        GpuMat lut_;
    };

    CLAHE_Impl::CLAHE_Impl(double clipLimit, int tilesX, int tilesY) :
        clipLimit_(clipLimit), tilesX_(tilesX), tilesY_(tilesY)
    {
    }

    CV_INIT_ALGORITHM(CLAHE_Impl, "CLAHE_GPU",
        obj.info()->addParam(obj, "clipLimit", obj.clipLimit_);
        obj.info()->addParam(obj, "tilesX", obj.tilesX_);
        obj.info()->addParam(obj, "tilesY", obj.tilesY_))

    void CLAHE_Impl::apply(cv::InputArray _src, cv::OutputArray _dst)
    {
        apply(_src, _dst, Stream::Null());
    }

    void CLAHE_Impl::apply(InputArray _src, OutputArray _dst, Stream& s)
    {
        GpuMat src = _src.getGpuMat();

        CV_Assert( src.type() == CV_8UC1 );

        _dst.create( src.size(), src.type() );
        GpuMat dst = _dst.getGpuMat();

        const int histSize = 256;

        ensureSizeIsEnough(tilesX_ * tilesY_, histSize, CV_8UC1, lut_);

        cudaStream_t stream = StreamAccessor::getStream(s);

        cv::Size tileSize;
        GpuMat srcForLut;

        if (src.cols % tilesX_ == 0 && src.rows % tilesY_ == 0)
        {
            tileSize = cv::Size(src.cols / tilesX_, src.rows / tilesY_);
            srcForLut = src;
        }
        else
        {
#ifndef HAVE_OPENCV_GPUARITHM
            throw_no_cuda();
#else
            cv::gpu::copyMakeBorder(src, srcExt_, 0, tilesY_ - (src.rows % tilesY_), 0, tilesX_ - (src.cols % tilesX_), cv::BORDER_REFLECT_101, cv::Scalar(), s);
#endif

            tileSize = cv::Size(srcExt_.cols / tilesX_, srcExt_.rows / tilesY_);
            srcForLut = srcExt_;
        }

        const int tileSizeTotal = tileSize.area();
        const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;

        int clipLimit = 0;
        if (clipLimit_ > 0.0)
        {
            clipLimit = static_cast<int>(clipLimit_ * tileSizeTotal / histSize);
            clipLimit = std::max(clipLimit, 1);
        }

        clahe::calcLut(srcForLut, lut_, tilesX_, tilesY_, make_int2(tileSize.width, tileSize.height), clipLimit, lutScale, stream);

        clahe::transform(src, dst, lut_, tilesX_, tilesY_, make_int2(tileSize.width, tileSize.height), stream);
    }

    void CLAHE_Impl::setClipLimit(double clipLimit)
    {
        clipLimit_ = clipLimit;
    }

    double CLAHE_Impl::getClipLimit() const
    {
        return clipLimit_;
    }

    void CLAHE_Impl::setTilesGridSize(cv::Size tileGridSize)
    {
        tilesX_ = tileGridSize.width;
        tilesY_ = tileGridSize.height;
    }

    cv::Size CLAHE_Impl::getTilesGridSize() const
    {
        return cv::Size(tilesX_, tilesY_);
    }

    void CLAHE_Impl::collectGarbage()
    {
        srcExt_.release();
        lut_.release();
    }
}

cv::Ptr<cv::gpu::CLAHE> cv::gpu::createCLAHE(double clipLimit, cv::Size tileGridSize)
{
    return new CLAHE_Impl(clipLimit, tileGridSize.width, tileGridSize.height);
}

#endif /* !defined (HAVE_CUDA) */
