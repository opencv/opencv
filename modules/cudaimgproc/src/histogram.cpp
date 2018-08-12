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

void cv::cuda::calcHist(InputArray, OutputArray, Stream&) { throw_no_cuda(); }

void cv::cuda::equalizeHist(InputArray, OutputArray, Stream&) { throw_no_cuda(); }

cv::Ptr<cv::cuda::CLAHE> cv::cuda::createCLAHE(double, cv::Size) { throw_no_cuda(); return cv::Ptr<cv::cuda::CLAHE>(); }

void cv::cuda::evenLevels(OutputArray, int, int, int, Stream&) { throw_no_cuda(); }

void cv::cuda::histEven(InputArray, OutputArray, int, int, int, Stream&) { throw_no_cuda(); }
void cv::cuda::histEven(InputArray, GpuMat*, int*, int*, int*, Stream&) { throw_no_cuda(); }

void cv::cuda::histRange(InputArray, OutputArray, InputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::histRange(InputArray, GpuMat*, const GpuMat*, Stream&) { throw_no_cuda(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// calcHist

namespace hist
{
    void histogram256(PtrStepSzb src, int* hist, cudaStream_t stream);
    void histogram256(PtrStepSzb src, PtrStepSzb mask, int* hist, cudaStream_t stream);
}

void cv::cuda::calcHist(InputArray _src, OutputArray _hist, Stream& stream)
{
    calcHist(_src, cv::cuda::GpuMat(), _hist, stream);
}

void cv::cuda::calcHist(InputArray _src, InputArray _mask, OutputArray _hist, Stream& stream)
{
    GpuMat src = _src.getGpuMat();
    GpuMat mask = _mask.getGpuMat();

    CV_Assert( src.type() == CV_8UC1 );
    CV_Assert( mask.empty() || mask.type() == CV_8UC1 );
    CV_Assert( mask.empty() || mask.size() == src.size() );

    _hist.create(1, 256, CV_32SC1);
    GpuMat hist = _hist.getGpuMat();

    hist.setTo(Scalar::all(0), stream);

    if (mask.empty())
        hist::histogram256(src, hist.ptr<int>(), StreamAccessor::getStream(stream));
    else
        hist::histogram256(src, mask, hist.ptr<int>(), StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// equalizeHist

namespace hist
{
    void equalizeHist(PtrStepSzb src, PtrStepSzb dst, const int* lut, cudaStream_t stream);
}

void cv::cuda::equalizeHist(InputArray _src, OutputArray _dst, Stream& _stream)
{
    GpuMat src = getInputMat(_src, _stream);

    CV_Assert( src.type() == CV_8UC1 );

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    int intBufSize;
    nppSafeCall( nppsIntegralGetBufferSize_32s(256, &intBufSize) );

    size_t bufSize = intBufSize + 2 * 256 * sizeof(int);

    BufferPool pool(_stream);
    GpuMat buf = pool.getBuffer(1, static_cast<int>(bufSize), CV_8UC1);

    GpuMat hist(1, 256, CV_32SC1, buf.data);
    GpuMat lut(1, 256, CV_32SC1, buf.data + 256 * sizeof(int));
    GpuMat intBuf(1, intBufSize, CV_8UC1, buf.data + 2 * 256 * sizeof(int));

    cuda::calcHist(src, hist, _stream);

    cudaStream_t stream = StreamAccessor::getStream(_stream);
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
    class CLAHE_Impl : public cv::cuda::CLAHE
    {
    public:
        CLAHE_Impl(double clipLimit = 40.0, int tilesX = 8, int tilesY = 8);

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
#ifndef HAVE_OPENCV_CUDAARITHM
            throw_no_cuda();
#else
            cv::cuda::copyMakeBorder(src, srcExt_, 0, tilesY_ - (src.rows % tilesY_), 0, tilesX_ - (src.cols % tilesX_), cv::BORDER_REFLECT_101, cv::Scalar(), s);
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

cv::Ptr<cv::cuda::CLAHE> cv::cuda::createCLAHE(double clipLimit, cv::Size tileGridSize)
{
    return makePtr<CLAHE_Impl>(clipLimit, tileGridSize.width, tileGridSize.height);
}

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

        static void hist(const GpuMat& src, OutputArray _hist, int histSize, int lowerLevel, int upperLevel, Stream& stream)
        {
            const int levels = histSize + 1;

            _hist.create(1, histSize, CV_32S);
            GpuMat hist = _hist.getGpuMat();

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            int buf_size;
            get_buf_size(sz, levels, &buf_size);

            BufferPool pool(stream);
            GpuMat buf = pool.getBuffer(1, buf_size, CV_8UC1);

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, hist.ptr<Npp32s>(), levels,
                lowerLevel, upperLevel, buf.ptr<Npp8u>()) );

            if (!stream)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int SDEPTH, typename NppHistogramEvenFuncC4<SDEPTH>::func_ptr func, get_buf_size_c4_t get_buf_size>
    struct NppHistogramEvenC4
    {
        typedef typename NppHistogramEvenFuncC4<SDEPTH>::src_t src_t;

        static void hist(const GpuMat& src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream)
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

            BufferPool pool(stream);
            GpuMat buf = pool.getBuffer(1, buf_size, CV_8UC1);

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, pHist, levels, lowerLevel, upperLevel, buf.ptr<Npp8u>()) );

            if (!stream)
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

        static void hist(const GpuMat& src, OutputArray _hist, const GpuMat& levels, Stream& stream)
        {
            CV_Assert( levels.type() == LEVEL_TYPE_CODE && levels.rows == 1 );

            _hist.create(1, levels.cols - 1, CV_32S);
            GpuMat hist = _hist.getGpuMat();

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            int buf_size;
            get_buf_size(sz, levels.cols, &buf_size);

            BufferPool pool(stream);
            GpuMat buf = pool.getBuffer(1, buf_size, CV_8UC1);

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, hist.ptr<Npp32s>(), levels.ptr<level_t>(), levels.cols, buf.ptr<Npp8u>()) );

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

        static void hist(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], Stream& stream)
        {
            CV_Assert( levels[0].type() == LEVEL_TYPE_CODE && levels[0].rows == 1 );
            CV_Assert( levels[1].type() == LEVEL_TYPE_CODE && levels[1].rows == 1 );
            CV_Assert( levels[2].type() == LEVEL_TYPE_CODE && levels[2].rows == 1 );
            CV_Assert( levels[3].type() == LEVEL_TYPE_CODE && levels[3].rows == 1 );

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

            BufferPool pool(stream);
            GpuMat buf = pool.getBuffer(1, buf_size, CV_8UC1);

            NppStreamHandler h(stream);

            nppSafeCall( func(src.ptr<src_t>(), static_cast<int>(src.step), sz, pHist, pLevels, nLevels, buf.ptr<Npp8u>()) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

void cv::cuda::evenLevels(OutputArray _levels, int nLevels, int lowerLevel, int upperLevel, Stream& stream)
{
    const int kind = _levels.kind();

    _levels.create(1, nLevels, CV_32SC1);

    Mat host_levels;
    if (kind == _InputArray::CUDA_GPU_MAT)
        host_levels.create(1, nLevels, CV_32SC1);
    else
        host_levels = _levels.getMat();

    nppSafeCall( nppiEvenLevelsHost_32s(host_levels.ptr<Npp32s>(), nLevels, lowerLevel, upperLevel) );

    if (kind == _InputArray::CUDA_GPU_MAT)
        _levels.getGpuMatRef().upload(host_levels, stream);
}

namespace hist
{
    void histEven8u(PtrStepSzb src, int* hist, int binCount, int lowerLevel, int upperLevel, cudaStream_t stream);
}

namespace
{
    void histEven8u(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel, cudaStream_t stream)
    {
        hist.create(1, histSize, CV_32S);
        cudaSafeCall( cudaMemsetAsync(hist.data, 0, histSize * sizeof(int), stream) );
        hist::histEven8u(src, hist.ptr<int>(), histSize, lowerLevel, upperLevel, stream);
    }
}

void cv::cuda::histEven(InputArray _src, OutputArray hist, int histSize, int lowerLevel, int upperLevel, Stream& stream)
{
    typedef void (*hist_t)(const GpuMat& src, OutputArray hist, int levels, int lowerLevel, int upperLevel, Stream& stream);
    static const hist_t hist_callers[] =
    {
        NppHistogramEvenC1<CV_8U , nppiHistogramEven_8u_C1R , nppiHistogramEvenGetBufferSize_8u_C1R >::hist,
        0,
        NppHistogramEvenC1<CV_16U, nppiHistogramEven_16u_C1R, nppiHistogramEvenGetBufferSize_16u_C1R>::hist,
        NppHistogramEvenC1<CV_16S, nppiHistogramEven_16s_C1R, nppiHistogramEvenGetBufferSize_16s_C1R>::hist
    };

    GpuMat src = _src.getGpuMat();

    if (src.depth() == CV_8U && deviceSupports(FEATURE_SET_COMPUTE_30))
    {
        histEven8u(src, hist.getGpuMatRef(), histSize, lowerLevel, upperLevel, StreamAccessor::getStream(stream));
        return;
    }

    CV_Assert( src.type() == CV_8UC1 || src.type() == CV_16UC1 || src.type() == CV_16SC1 );

    hist_callers[src.depth()](src, hist, histSize, lowerLevel, upperLevel, stream);
}

void cv::cuda::histEven(InputArray _src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream)
{
    typedef void (*hist_t)(const GpuMat& src, GpuMat hist[4], int levels[4], int lowerLevel[4], int upperLevel[4], Stream& stream);
    static const hist_t hist_callers[] =
    {
        NppHistogramEvenC4<CV_8U , nppiHistogramEven_8u_C4R , nppiHistogramEvenGetBufferSize_8u_C4R >::hist,
        0,
        NppHistogramEvenC4<CV_16U, nppiHistogramEven_16u_C4R, nppiHistogramEvenGetBufferSize_16u_C4R>::hist,
        NppHistogramEvenC4<CV_16S, nppiHistogramEven_16s_C4R, nppiHistogramEvenGetBufferSize_16s_C4R>::hist
    };

    GpuMat src = _src.getGpuMat();

    CV_Assert( src.type() == CV_8UC4 || src.type() == CV_16UC4 || src.type() == CV_16SC4 );

    hist_callers[src.depth()](src, hist, histSize, lowerLevel, upperLevel, stream);
}

void cv::cuda::histRange(InputArray _src, OutputArray hist, InputArray _levels, Stream& stream)
{
    typedef void (*hist_t)(const GpuMat& src, OutputArray hist, const GpuMat& levels, Stream& stream);
    static const hist_t hist_callers[] =
    {
        NppHistogramRangeC1<CV_8U , nppiHistogramRange_8u_C1R , nppiHistogramRangeGetBufferSize_8u_C1R >::hist,
        0,
        NppHistogramRangeC1<CV_16U, nppiHistogramRange_16u_C1R, nppiHistogramRangeGetBufferSize_16u_C1R>::hist,
        NppHistogramRangeC1<CV_16S, nppiHistogramRange_16s_C1R, nppiHistogramRangeGetBufferSize_16s_C1R>::hist,
        0,
        NppHistogramRangeC1<CV_32F, nppiHistogramRange_32f_C1R, nppiHistogramRangeGetBufferSize_32f_C1R>::hist
    };

    GpuMat src = _src.getGpuMat();
    GpuMat levels = _levels.getGpuMat();

    CV_Assert( src.type() == CV_8UC1 || src.type() == CV_16UC1 || src.type() == CV_16SC1 || src.type() == CV_32FC1 );

    hist_callers[src.depth()](src, hist, levels, stream);
}

void cv::cuda::histRange(InputArray _src, GpuMat hist[4], const GpuMat levels[4], Stream& stream)
{
    typedef void (*hist_t)(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], Stream& stream);
    static const hist_t hist_callers[] =
    {
        NppHistogramRangeC4<CV_8U , nppiHistogramRange_8u_C4R , nppiHistogramRangeGetBufferSize_8u_C4R >::hist,
        0,
        NppHistogramRangeC4<CV_16U, nppiHistogramRange_16u_C4R, nppiHistogramRangeGetBufferSize_16u_C4R>::hist,
        NppHistogramRangeC4<CV_16S, nppiHistogramRange_16s_C4R, nppiHistogramRangeGetBufferSize_16s_C4R>::hist,
        0,
        NppHistogramRangeC4<CV_32F, nppiHistogramRange_32f_C4R, nppiHistogramRangeGetBufferSize_32f_C4R>::hist
    };

    GpuMat src = _src.getGpuMat();

    CV_Assert( src.type() == CV_8UC4 || src.type() == CV_16UC4 || src.type() == CV_16SC4 || src.type() == CV_32FC4 );

    hist_callers[src.depth()](src, hist, levels, stream);
}

#endif /* !defined (HAVE_CUDA) */
