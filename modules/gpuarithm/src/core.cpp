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

void cv::gpu::merge(const GpuMat* /*src*/, size_t /*count*/, GpuMat& /*dst*/, Stream& /*stream*/) { throw_no_cuda(); }
void cv::gpu::merge(const std::vector<GpuMat>& /*src*/, GpuMat& /*dst*/, Stream& /*stream*/) { throw_no_cuda(); }

void cv::gpu::split(const GpuMat& /*src*/, GpuMat* /*dst*/, Stream& /*stream*/) { throw_no_cuda(); }
void cv::gpu::split(const GpuMat& /*src*/, std::vector<GpuMat>& /*dst*/, Stream& /*stream*/) { throw_no_cuda(); }

void cv::gpu::transpose(const GpuMat&, GpuMat&, Stream&) { throw_no_cuda(); }

void cv::gpu::flip(const GpuMat&, GpuMat&, int, Stream&) { throw_no_cuda(); }

void cv::gpu::LUT(const GpuMat&, const Mat&, GpuMat&, Stream&) { throw_no_cuda(); }

void cv::gpu::copyMakeBorder(const GpuMat&, GpuMat&, int, int, int, int, int, const Scalar&, Stream&) { throw_no_cuda(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// merge/split

namespace cv { namespace gpu { namespace cudev
{
    namespace split_merge
    {
        void merge_caller(const PtrStepSzb* src, PtrStepSzb& dst, int total_channels, size_t elem_size, const cudaStream_t& stream);
        void split_caller(const PtrStepSzb& src, PtrStepSzb* dst, int num_channels, size_t elem_size1, const cudaStream_t& stream);
    }
}}}

namespace
{
    void merge(const GpuMat* src, size_t n, GpuMat& dst, const cudaStream_t& stream)
    {
        using namespace ::cv::gpu::cudev::split_merge;

        CV_Assert(src);
        CV_Assert(n > 0);

        int depth = src[0].depth();
        Size size = src[0].size();

        if (depth == CV_64F)
        {
            if (!deviceSupports(NATIVE_DOUBLE))
                CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
        }

        bool single_channel_only = true;
        int total_channels = 0;

        for (size_t i = 0; i < n; ++i)
        {
            CV_Assert(src[i].size() == size);
            CV_Assert(src[i].depth() == depth);
            single_channel_only = single_channel_only && src[i].channels() == 1;
            total_channels += src[i].channels();
        }

        CV_Assert(single_channel_only);
        CV_Assert(total_channels <= 4);

        if (total_channels == 1)
            src[0].copyTo(dst);
        else
        {
            dst.create(size, CV_MAKETYPE(depth, total_channels));

            PtrStepSzb src_as_devmem[4];
            for(size_t i = 0; i < n; ++i)
                src_as_devmem[i] = src[i];

            PtrStepSzb dst_as_devmem(dst);
            merge_caller(src_as_devmem, dst_as_devmem, total_channels, CV_ELEM_SIZE(depth), stream);
        }
    }

    void split(const GpuMat& src, GpuMat* dst, const cudaStream_t& stream)
    {
        using namespace ::cv::gpu::cudev::split_merge;

        CV_Assert(dst);

        int depth = src.depth();
        int num_channels = src.channels();

        if (depth == CV_64F)
        {
            if (!deviceSupports(NATIVE_DOUBLE))
                CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
        }

        if (num_channels == 1)
        {
            src.copyTo(dst[0]);
            return;
        }

        for (int i = 0; i < num_channels; ++i)
            dst[i].create(src.size(), depth);

        CV_Assert(num_channels <= 4);

        PtrStepSzb dst_as_devmem[4];
        for (int i = 0; i < num_channels; ++i)
            dst_as_devmem[i] = dst[i];

        PtrStepSzb src_as_devmem(src);
        split_caller(src_as_devmem, dst_as_devmem, num_channels, src.elemSize1(), stream);
    }
}

void cv::gpu::merge(const GpuMat* src, size_t n, GpuMat& dst, Stream& stream)
{
    ::merge(src, n, dst, StreamAccessor::getStream(stream));
}


void cv::gpu::merge(const std::vector<GpuMat>& src, GpuMat& dst, Stream& stream)
{
    ::merge(&src[0], src.size(), dst, StreamAccessor::getStream(stream));
}

void cv::gpu::split(const GpuMat& src, GpuMat* dst, Stream& stream)
{
    ::split(src, dst, StreamAccessor::getStream(stream));
}

void cv::gpu::split(const GpuMat& src, std::vector<GpuMat>& dst, Stream& stream)
{
    dst.resize(src.channels());
    if(src.channels() > 0)
        ::split(src, &dst[0], StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// transpose

namespace arithm
{
    template <typename T> void transpose(PtrStepSz<T> src, PtrStepSz<T> dst, cudaStream_t stream);
}

void cv::gpu::transpose(const GpuMat& src, GpuMat& dst, Stream& s)
{
    CV_Assert( src.elemSize() == 1 || src.elemSize() == 4 || src.elemSize() == 8 );

    dst.create( src.cols, src.rows, src.type() );

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (src.elemSize() == 1)
    {
        NppStreamHandler h(stream);

        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        nppSafeCall( nppiTranspose_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step),
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    else if (src.elemSize() == 4)
    {
        arithm::transpose<int>(src, dst, stream);
    }
    else // if (src.elemSize() == 8)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");

        arithm::transpose<double>(src, dst, stream);
    }
}

////////////////////////////////////////////////////////////////////////
// flip

namespace
{
    template<int DEPTH> struct NppTypeTraits;
    template<> struct NppTypeTraits<CV_8U>  { typedef Npp8u npp_t; };
    template<> struct NppTypeTraits<CV_8S>  { typedef Npp8s npp_t; };
    template<> struct NppTypeTraits<CV_16U> { typedef Npp16u npp_t; };
    template<> struct NppTypeTraits<CV_16S> { typedef Npp16s npp_t; };
    template<> struct NppTypeTraits<CV_32S> { typedef Npp32s npp_t; };
    template<> struct NppTypeTraits<CV_32F> { typedef Npp32f npp_t; };
    template<> struct NppTypeTraits<CV_64F> { typedef Npp64f npp_t; };

    template <int DEPTH> struct NppMirrorFunc
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        typedef NppStatus (*func_t)(const npp_t* pSrc, int nSrcStep, npp_t* pDst, int nDstStep, NppiSize oROI, NppiAxis flip);
    };

    template <int DEPTH, typename NppMirrorFunc<DEPTH>::func_t func> struct NppMirror
    {
        typedef typename NppMirrorFunc<DEPTH>::npp_t npp_t;

        static void call(const GpuMat& src, GpuMat& dst, int flipCode, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize sz;
            sz.width  = src.cols;
            sz.height = src.rows;

            nppSafeCall( func(src.ptr<npp_t>(), static_cast<int>(src.step),
                dst.ptr<npp_t>(), static_cast<int>(dst.step), sz,
                (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS))) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

void cv::gpu::flip(const GpuMat& src, GpuMat& dst, int flipCode, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, GpuMat& dst, int flipCode, cudaStream_t stream);
    static const func_t funcs[6][4] =
    {
        {NppMirror<CV_8U, nppiMirror_8u_C1R>::call, 0, NppMirror<CV_8U, nppiMirror_8u_C3R>::call, NppMirror<CV_8U, nppiMirror_8u_C4R>::call},
        {0,0,0,0},
        {NppMirror<CV_16U, nppiMirror_16u_C1R>::call, 0, NppMirror<CV_16U, nppiMirror_16u_C3R>::call, NppMirror<CV_16U, nppiMirror_16u_C4R>::call},
        {0,0,0,0},
        {NppMirror<CV_32S, nppiMirror_32s_C1R>::call, 0, NppMirror<CV_32S, nppiMirror_32s_C3R>::call, NppMirror<CV_32S, nppiMirror_32s_C4R>::call},
        {NppMirror<CV_32F, nppiMirror_32f_C1R>::call, 0, NppMirror<CV_32F, nppiMirror_32f_C3R>::call, NppMirror<CV_32F, nppiMirror_32f_C4R>::call}
    };

    CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32S || src.depth() == CV_32F);
    CV_Assert(src.channels() == 1 || src.channels() == 3 || src.channels() == 4);

    dst.create(src.size(), src.type());

    funcs[src.depth()][src.channels() - 1](src, dst, flipCode, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// LUT

void cv::gpu::LUT(const GpuMat& src, const Mat& lut, GpuMat& dst, Stream& s)
{
    const int cn = src.channels();

    CV_Assert( src.type() == CV_8UC1 || src.type() == CV_8UC3 );
    CV_Assert( lut.depth() == CV_8U );
    CV_Assert( lut.channels() == 1 || lut.channels() == cn );
    CV_Assert( lut.rows * lut.cols == 256 && lut.isContinuous() );

    dst.create(src.size(), CV_MAKE_TYPE(lut.depth(), cn));

    NppiSize sz;
    sz.height = src.rows;
    sz.width = src.cols;

    Mat nppLut;
    lut.convertTo(nppLut, CV_32S);

    int nValues3[] = {256, 256, 256};

    Npp32s pLevels[256];
    for (int i = 0; i < 256; ++i)
        pLevels[i] = i;

    const Npp32s* pLevels3[3];

#if (CUDA_VERSION <= 4020)
    pLevels3[0] = pLevels3[1] = pLevels3[2] = pLevels;
#else
    GpuMat d_pLevels;
    d_pLevels.upload(Mat(1, 256, CV_32S, pLevels));
    pLevels3[0] = pLevels3[1] = pLevels3[2] = d_pLevels.ptr<Npp32s>();
#endif

    cudaStream_t stream = StreamAccessor::getStream(s);
    NppStreamHandler h(stream);

    if (src.type() == CV_8UC1)
    {
#if (CUDA_VERSION <= 4020)
        nppSafeCall( nppiLUT_Linear_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step),
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz, nppLut.ptr<Npp32s>(), pLevels, 256) );
#else
        GpuMat d_nppLut(Mat(1, 256, CV_32S, nppLut.data));
        nppSafeCall( nppiLUT_Linear_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step),
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz, d_nppLut.ptr<Npp32s>(), d_pLevels.ptr<Npp32s>(), 256) );
#endif
    }
    else
    {
        const Npp32s* pValues3[3];

        Mat nppLut3[3];
        if (nppLut.channels() == 1)
        {
#if (CUDA_VERSION <= 4020)
            pValues3[0] = pValues3[1] = pValues3[2] = nppLut.ptr<Npp32s>();
#else
            GpuMat d_nppLut(Mat(1, 256, CV_32S, nppLut.data));
            pValues3[0] = pValues3[1] = pValues3[2] = d_nppLut.ptr<Npp32s>();
#endif
        }
        else
        {
            cv::split(nppLut, nppLut3);

#if (CUDA_VERSION <= 4020)
            pValues3[0] = nppLut3[0].ptr<Npp32s>();
            pValues3[1] = nppLut3[1].ptr<Npp32s>();
            pValues3[2] = nppLut3[2].ptr<Npp32s>();
#else
            GpuMat d_nppLut0(Mat(1, 256, CV_32S, nppLut3[0].data));
            GpuMat d_nppLut1(Mat(1, 256, CV_32S, nppLut3[1].data));
            GpuMat d_nppLut2(Mat(1, 256, CV_32S, nppLut3[2].data));

            pValues3[0] = d_nppLut0.ptr<Npp32s>();
            pValues3[1] = d_nppLut1.ptr<Npp32s>();
            pValues3[2] = d_nppLut2.ptr<Npp32s>();
#endif
        }

        nppSafeCall( nppiLUT_Linear_8u_C3R(src.ptr<Npp8u>(), static_cast<int>(src.step),
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz, pValues3, pLevels3, nValues3) );
    }

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// copyMakeBorder

namespace cv { namespace gpu { namespace cudev
{
    namespace imgproc
    {
        template <typename T, int cn> void copyMakeBorder_gpu(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderMode, const T* borderValue, cudaStream_t stream);
    }
}}}

namespace
{
    template <typename T, int cn> void copyMakeBorder_caller(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderType, const Scalar& value, cudaStream_t stream)
    {
        using namespace ::cv::gpu::cudev::imgproc;

        Scalar_<T> val(saturate_cast<T>(value[0]), saturate_cast<T>(value[1]), saturate_cast<T>(value[2]), saturate_cast<T>(value[3]));

        copyMakeBorder_gpu<T, cn>(src, dst, top, left, borderType, val.val, stream);
    }
}

#if defined __GNUC__ && __GNUC__ > 2 && __GNUC_MINOR__  > 4
typedef Npp32s __attribute__((__may_alias__)) Npp32s_a;
#else
typedef Npp32s Npp32s_a;
#endif

void cv::gpu::copyMakeBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, int borderType, const Scalar& value, Stream& s)
{
    CV_Assert(src.depth() <= CV_32F && src.channels() <= 4);
    CV_Assert(borderType == BORDER_REFLECT_101 || borderType == BORDER_REPLICATE || borderType == BORDER_CONSTANT || borderType == BORDER_REFLECT || borderType == BORDER_WRAP);

    dst.create(src.rows + top + bottom, src.cols + left + right, src.type());

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (borderType == BORDER_CONSTANT && (src.type() == CV_8UC1 || src.type() == CV_8UC4 || src.type() == CV_32SC1 || src.type() == CV_32FC1))
    {
        NppiSize srcsz;
        srcsz.width  = src.cols;
        srcsz.height = src.rows;

        NppiSize dstsz;
        dstsz.width  = dst.cols;
        dstsz.height = dst.rows;

        NppStreamHandler h(stream);

        switch (src.type())
        {
        case CV_8UC1:
            {
                Npp8u nVal = saturate_cast<Npp8u>(value[0]);
                nppSafeCall( nppiCopyConstBorder_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step), srcsz,
                    dst.ptr<Npp8u>(), static_cast<int>(dst.step), dstsz, top, left, nVal) );
                break;
            }
        case CV_8UC4:
            {
                Npp8u nVal[] = {saturate_cast<Npp8u>(value[0]), saturate_cast<Npp8u>(value[1]), saturate_cast<Npp8u>(value[2]), saturate_cast<Npp8u>(value[3])};
                nppSafeCall( nppiCopyConstBorder_8u_C4R(src.ptr<Npp8u>(), static_cast<int>(src.step), srcsz,
                    dst.ptr<Npp8u>(), static_cast<int>(dst.step), dstsz, top, left, nVal) );
                break;
            }
        case CV_32SC1:
            {
                Npp32s nVal = saturate_cast<Npp32s>(value[0]);
                nppSafeCall( nppiCopyConstBorder_32s_C1R(src.ptr<Npp32s>(), static_cast<int>(src.step), srcsz,
                    dst.ptr<Npp32s>(), static_cast<int>(dst.step), dstsz, top, left, nVal) );
                break;
            }
        case CV_32FC1:
            {
                Npp32f val = saturate_cast<Npp32f>(value[0]);
                Npp32s nVal = *(reinterpret_cast<Npp32s_a*>(&val));
                nppSafeCall( nppiCopyConstBorder_32s_C1R(src.ptr<Npp32s>(), static_cast<int>(src.step), srcsz,
                    dst.ptr<Npp32s>(), static_cast<int>(dst.step), dstsz, top, left, nVal) );
                break;
            }
        }

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    else
    {
        typedef void (*caller_t)(const PtrStepSzb& src, const PtrStepSzb& dst, int top, int left, int borderType, const Scalar& value, cudaStream_t stream);
        static const caller_t callers[6][4] =
        {
            {   copyMakeBorder_caller<uchar, 1>  ,    copyMakeBorder_caller<uchar, 2>   ,    copyMakeBorder_caller<uchar, 3>  ,    copyMakeBorder_caller<uchar, 4>},
            {0/*copyMakeBorder_caller<schar, 1>*/, 0/*copyMakeBorder_caller<schar, 2>*/ , 0/*copyMakeBorder_caller<schar, 3>*/, 0/*copyMakeBorder_caller<schar, 4>*/},
            {   copyMakeBorder_caller<ushort, 1> , 0/*copyMakeBorder_caller<ushort, 2>*/,    copyMakeBorder_caller<ushort, 3> ,    copyMakeBorder_caller<ushort, 4>},
            {   copyMakeBorder_caller<short, 1>  , 0/*copyMakeBorder_caller<short, 2>*/ ,    copyMakeBorder_caller<short, 3>  ,    copyMakeBorder_caller<short, 4>},
            {0/*copyMakeBorder_caller<int,   1>*/, 0/*copyMakeBorder_caller<int,   2>*/ , 0/*copyMakeBorder_caller<int,   3>*/, 0/*copyMakeBorder_caller<int  , 4>*/},
            {   copyMakeBorder_caller<float, 1>  , 0/*copyMakeBorder_caller<float, 2>*/ ,    copyMakeBorder_caller<float, 3>  ,    copyMakeBorder_caller<float ,4>}
        };

        caller_t func = callers[src.depth()][src.channels() - 1];
        CV_Assert(func != 0);

        func(src, dst, top, left, borderType, value, stream);
    }
}

#endif /* !defined (HAVE_CUDA) */
