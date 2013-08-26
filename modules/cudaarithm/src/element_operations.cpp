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

void cv::cuda::add(InputArray, InputArray, OutputArray, InputArray, int, Stream&) { throw_no_cuda(); }
void cv::cuda::subtract(InputArray, InputArray, OutputArray, InputArray, int, Stream&) { throw_no_cuda(); }
void cv::cuda::multiply(InputArray, InputArray, OutputArray, double, int, Stream&) { throw_no_cuda(); }
void cv::cuda::divide(InputArray, InputArray, OutputArray, double, int, Stream&) { throw_no_cuda(); }
void cv::cuda::absdiff(InputArray, InputArray, OutputArray, Stream&) { throw_no_cuda(); }

void cv::cuda::abs(InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::sqr(InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::sqrt(InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::exp(InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::log(InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::pow(InputArray, double, OutputArray, Stream&) { throw_no_cuda(); }

void cv::cuda::compare(InputArray, InputArray, OutputArray, int, Stream&) { throw_no_cuda(); }

void cv::cuda::bitwise_not(InputArray, OutputArray, InputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::bitwise_or(InputArray, InputArray, OutputArray, InputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::bitwise_and(InputArray, InputArray, OutputArray, InputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::bitwise_xor(InputArray, InputArray, OutputArray, InputArray, Stream&) { throw_no_cuda(); }

void cv::cuda::rshift(InputArray, Scalar_<int>, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::lshift(InputArray, Scalar_<int>, OutputArray, Stream&) { throw_no_cuda(); }

void cv::cuda::min(InputArray, InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::max(InputArray, InputArray, OutputArray, Stream&) { throw_no_cuda(); }

void cv::cuda::addWeighted(InputArray, double, InputArray, double, double, OutputArray, int, Stream&) { throw_no_cuda(); }

double cv::cuda::threshold(InputArray, OutputArray, double, double, int, Stream&) {throw_no_cuda(); return 0.0;}

void cv::cuda::magnitude(InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::magnitude(InputArray, InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::magnitudeSqr(InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::magnitudeSqr(InputArray, InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::phase(InputArray, InputArray, OutputArray, bool, Stream&) { throw_no_cuda(); }
void cv::cuda::cartToPolar(InputArray, InputArray, OutputArray, OutputArray, bool, Stream&) { throw_no_cuda(); }
void cv::cuda::polarToCart(InputArray, InputArray, OutputArray, OutputArray, bool, Stream&) { throw_no_cuda(); }

#else

////////////////////////////////////////////////////////////////////////
// arithm_op

namespace
{
    typedef void (*mat_mat_func_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, double scale, Stream& stream, int op);
    typedef void (*mat_scalar_func_t)(const GpuMat& src, Scalar val, bool inv, GpuMat& dst, const GpuMat& mask, double scale, Stream& stream, int op);

    void arithm_op(InputArray _src1, InputArray _src2, OutputArray _dst, InputArray _mask, double scale, int dtype, Stream& stream,
                   mat_mat_func_t mat_mat_func, mat_scalar_func_t mat_scalar_func, int op = 0)
    {
        const int kind1 = _src1.kind();
        const int kind2 = _src2.kind();

        const bool isScalar1 = (kind1 == _InputArray::MATX);
        const bool isScalar2 = (kind2 == _InputArray::MATX);
        CV_Assert( !isScalar1 || !isScalar2 );

        GpuMat src1;
        if (!isScalar1)
            src1 = _src1.getGpuMat();

        GpuMat src2;
        if (!isScalar2)
            src2 = _src2.getGpuMat();

        Mat scalar;
        if (isScalar1)
            scalar = _src1.getMat();
        else if (isScalar2)
            scalar = _src2.getMat();

        Scalar val;
        if (!scalar.empty())
        {
            CV_Assert( scalar.total() <= 4 );
            scalar.convertTo(Mat_<double>(scalar.rows, scalar.cols, &val[0]), CV_64F);
        }

        GpuMat mask = _mask.getGpuMat();

        const int sdepth = src1.empty() ? src2.depth() : src1.depth();
        const int cn = src1.empty() ? src2.channels() : src1.channels();
        const Size size = src1.empty() ? src2.size() : src1.size();

        if (dtype < 0)
            dtype = sdepth;

        const int ddepth = CV_MAT_DEPTH(dtype);

        CV_Assert( sdepth <= CV_64F && ddepth <= CV_64F );
        CV_Assert( !scalar.empty() || (src2.type() == src1.type() && src2.size() == src1.size()) );
        CV_Assert( mask.empty() || (cn == 1 && mask.size() == size && mask.type() == CV_8UC1) );

        if (sdepth == CV_64F || ddepth == CV_64F)
        {
            if (!deviceSupports(NATIVE_DOUBLE))
                CV_Error(Error::StsUnsupportedFormat, "The device doesn't support double");
        }

        _dst.create(size, CV_MAKE_TYPE(ddepth, cn));
        GpuMat dst = _dst.getGpuMat();

        if (isScalar1)
            mat_scalar_func(src2, val, true, dst, mask, scale, stream, op);
        else if (isScalar2)
            mat_scalar_func(src1, val, false, dst, mask, scale, stream, op);
        else
            mat_mat_func(src1, src2, dst, mask, scale, stream, op);
    }
}

////////////////////////////////////////////////////////////////////////
// add

void addMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, double, Stream& _stream, int);

void addScalar(const GpuMat& src, Scalar val, bool, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int);

void cv::cuda::add(InputArray src1, InputArray src2, OutputArray dst, InputArray mask, int dtype, Stream& stream)
{
    arithm_op(src1, src2, dst, mask, 1.0, dtype, stream, addMat, addScalar);
}

////////////////////////////////////////////////////////////////////////
// subtract

void subMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, double, Stream& _stream, int);

void subScalar(const GpuMat& src, Scalar val, bool inv, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int);

void cv::cuda::subtract(InputArray src1, InputArray src2, OutputArray dst, InputArray mask, int dtype, Stream& stream)
{
    arithm_op(src1, src2, dst, mask, 1.0, dtype, stream, subMat, subScalar);
}

////////////////////////////////////////////////////////////////////////
// multiply

void mulMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double scale, Stream& stream, int);
void mulMat_8uc4_32f(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream);
void mulMat_16sc4_32f(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream);

void mulScalar(const GpuMat& src, cv::Scalar val, bool, GpuMat& dst, const GpuMat& mask, double scale, Stream& stream, int);

void cv::cuda::multiply(InputArray _src1, InputArray _src2, OutputArray _dst, double scale, int dtype, Stream& stream)
{
    if (_src1.type() == CV_8UC4 && _src2.type() == CV_32FC1)
    {
        GpuMat src1 = _src1.getGpuMat();
        GpuMat src2 = _src2.getGpuMat();

        CV_Assert( src1.size() == src2.size() );

        _dst.create(src1.size(), src1.type());
        GpuMat dst = _dst.getGpuMat();

        mulMat_8uc4_32f(src1, src2, dst, stream);
    }
    else if (_src1.type() == CV_16SC4 && _src2.type() == CV_32FC1)
    {
        GpuMat src1 = _src1.getGpuMat();
        GpuMat src2 = _src2.getGpuMat();

        CV_Assert( src1.size() == src2.size() );

        _dst.create(src1.size(), src1.type());
        GpuMat dst = _dst.getGpuMat();

        mulMat_16sc4_32f(src1, src2, dst, stream);
    }
    else
    {
        arithm_op(_src1, _src2, _dst, GpuMat(), scale, dtype, stream, mulMat, mulScalar);
    }
}

////////////////////////////////////////////////////////////////////////
// divide

void divMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double scale, Stream& stream, int);
void divMat_8uc4_32f(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream);
void divMat_16sc4_32f(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream);

void divScalar(const GpuMat& src, cv::Scalar val, bool inv, GpuMat& dst, const GpuMat& mask, double scale, Stream& stream, int);

void cv::cuda::divide(InputArray _src1, InputArray _src2, OutputArray _dst, double scale, int dtype, Stream& stream)
{
    if (_src1.type() == CV_8UC4 && _src2.type() == CV_32FC1)
    {
        GpuMat src1 = _src1.getGpuMat();
        GpuMat src2 = _src2.getGpuMat();

        CV_Assert( src1.size() == src2.size() );

        _dst.create(src1.size(), src1.type());
        GpuMat dst = _dst.getGpuMat();

        divMat_8uc4_32f(src1, src2, dst, stream);
    }
    else if (_src1.type() == CV_16SC4 && _src2.type() == CV_32FC1)
    {
        GpuMat src1 = _src1.getGpuMat();
        GpuMat src2 = _src2.getGpuMat();

        CV_Assert( src1.size() == src2.size() );

        _dst.create(src1.size(), src1.type());
        GpuMat dst = _dst.getGpuMat();

        divMat_16sc4_32f(src1, src2, dst, stream);
    }
    else
    {
        arithm_op(_src1, _src2, _dst, GpuMat(), scale, dtype, stream, divMat, divScalar);
    }
}

//////////////////////////////////////////////////////////////////////////////
// absdiff

void absDiffMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double, Stream& stream, int);

void absDiffScalar(const GpuMat& src, cv::Scalar val, bool, GpuMat& dst, const GpuMat&, double, Stream& stream, int);

void cv::cuda::absdiff(InputArray src1, InputArray src2, OutputArray dst, Stream& stream)
{
    arithm_op(src1, src2, dst, noArray(), 1.0, -1, stream, absDiffMat, absDiffScalar);
}

//////////////////////////////////////////////////////////////////////////////
// compare

void cmpMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double, Stream& stream, int cmpop);

void cmpScalar(const GpuMat& src, Scalar val, bool inv, GpuMat& dst, const GpuMat&, double, Stream& stream, int cmpop);

void cv::cuda::compare(InputArray src1, InputArray src2, OutputArray dst, int cmpop, Stream& stream)
{
    arithm_op(src1, src2, dst, noArray(), 1.0, CV_8U, stream, cmpMat, cmpScalar, cmpop);
}

//////////////////////////////////////////////////////////////////////////////
// Binary bitwise logical operations

namespace
{
    enum
    {
        BIT_OP_AND,
        BIT_OP_OR,
        BIT_OP_XOR
    };
}

void bitMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int op);

void bitScalar(const GpuMat& src, cv::Scalar value, bool, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int op);

void cv::cuda::bitwise_or(InputArray src1, InputArray src2, OutputArray dst, InputArray mask, Stream& stream)
{
    arithm_op(src1, src2, dst, mask, 1.0, -1, stream, bitMat, bitScalar, BIT_OP_OR);
}

void cv::cuda::bitwise_and(InputArray src1, InputArray src2, OutputArray dst, InputArray mask, Stream& stream)
{
    arithm_op(src1, src2, dst, mask, 1.0, -1, stream, bitMat, bitScalar, BIT_OP_AND);
}

void cv::cuda::bitwise_xor(InputArray src1, InputArray src2, OutputArray dst, InputArray mask, Stream& stream)
{
    arithm_op(src1, src2, dst, mask, 1.0, -1, stream, bitMat, bitScalar, BIT_OP_XOR);
}

//////////////////////////////////////////////////////////////////////////////
// shift

namespace
{
    template <int DEPTH, int cn> struct NppShiftFunc
    {
        typedef typename NPPTypeTraits<DEPTH>::npp_type npp_type;

        typedef NppStatus (*func_t)(const npp_type* pSrc1, int nSrc1Step, const Npp32u* pConstants, npp_type* pDst,  int nDstStep,  NppiSize oSizeROI);
    };
    template <int DEPTH> struct NppShiftFunc<DEPTH, 1>
    {
        typedef typename NPPTypeTraits<DEPTH>::npp_type npp_type;

        typedef NppStatus (*func_t)(const npp_type* pSrc1, int nSrc1Step, const Npp32u pConstants, npp_type* pDst,  int nDstStep,  NppiSize oSizeROI);
    };

    template <int DEPTH, int cn, typename NppShiftFunc<DEPTH, cn>::func_t func> struct NppShift
    {
        typedef typename NPPTypeTraits<DEPTH>::npp_type npp_type;

        static void call(const GpuMat& src, Scalar_<Npp32u> sc, GpuMat& dst, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize oSizeROI;
            oSizeROI.width = src.cols;
            oSizeROI.height = src.rows;

            nppSafeCall( func(src.ptr<npp_type>(), static_cast<int>(src.step), sc.val, dst.ptr<npp_type>(), static_cast<int>(dst.step), oSizeROI) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template <int DEPTH, typename NppShiftFunc<DEPTH, 1>::func_t func> struct NppShift<DEPTH, 1, func>
    {
        typedef typename NPPTypeTraits<DEPTH>::npp_type npp_type;

        static void call(const GpuMat& src, Scalar_<Npp32u> sc, GpuMat& dst, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize oSizeROI;
            oSizeROI.width = src.cols;
            oSizeROI.height = src.rows;

            nppSafeCall( func(src.ptr<npp_type>(), static_cast<int>(src.step), sc.val[0], dst.ptr<npp_type>(), static_cast<int>(dst.step), oSizeROI) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

void cv::cuda::rshift(InputArray _src, Scalar_<int> val, OutputArray _dst, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, Scalar_<Npp32u> sc, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[5][4] =
    {
        {NppShift<CV_8U , 1, nppiRShiftC_8u_C1R >::call, 0, NppShift<CV_8U , 3, nppiRShiftC_8u_C3R >::call, NppShift<CV_8U , 4, nppiRShiftC_8u_C4R>::call },
        {NppShift<CV_8S , 1, nppiRShiftC_8s_C1R >::call, 0, NppShift<CV_8S , 3, nppiRShiftC_8s_C3R >::call, NppShift<CV_8S , 4, nppiRShiftC_8s_C4R>::call },
        {NppShift<CV_16U, 1, nppiRShiftC_16u_C1R>::call, 0, NppShift<CV_16U, 3, nppiRShiftC_16u_C3R>::call, NppShift<CV_16U, 4, nppiRShiftC_16u_C4R>::call},
        {NppShift<CV_16S, 1, nppiRShiftC_16s_C1R>::call, 0, NppShift<CV_16S, 3, nppiRShiftC_16s_C3R>::call, NppShift<CV_16S, 4, nppiRShiftC_16s_C4R>::call},
        {NppShift<CV_32S, 1, nppiRShiftC_32s_C1R>::call, 0, NppShift<CV_32S, 3, nppiRShiftC_32s_C3R>::call, NppShift<CV_32S, 4, nppiRShiftC_32s_C4R>::call},
    };

    GpuMat src = _src.getGpuMat();

    CV_Assert( src.depth() < CV_32F );
    CV_Assert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    funcs[src.depth()][src.channels() - 1](src, val, dst, StreamAccessor::getStream(stream));
}

void cv::cuda::lshift(InputArray _src, Scalar_<int> val, OutputArray _dst, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, Scalar_<Npp32u> sc, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[5][4] =
    {
        {NppShift<CV_8U , 1, nppiLShiftC_8u_C1R>::call , 0, NppShift<CV_8U , 3, nppiLShiftC_8u_C3R>::call , NppShift<CV_8U , 4, nppiLShiftC_8u_C4R>::call },
        {0                                             , 0, 0                                             , 0                                             },
        {NppShift<CV_16U, 1, nppiLShiftC_16u_C1R>::call, 0, NppShift<CV_16U, 3, nppiLShiftC_16u_C3R>::call, NppShift<CV_16U, 4, nppiLShiftC_16u_C4R>::call},
        {0                                             , 0, 0                                             , 0                                             },
        {NppShift<CV_32S, 1, nppiLShiftC_32s_C1R>::call, 0, NppShift<CV_32S, 3, nppiLShiftC_32s_C3R>::call, NppShift<CV_32S, 4, nppiLShiftC_32s_C4R>::call},
    };

    GpuMat src = _src.getGpuMat();

    CV_Assert( src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32S );
    CV_Assert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    funcs[src.depth()][src.channels() - 1](src, val, dst, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// Minimum and maximum operations

namespace
{
    enum
    {
        MIN_OP,
        MAX_OP
    };
}

namespace arithm
{
    void minMat_v4(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);
    void minMat_v2(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);
    template <typename T> void minMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void minScalar(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);

    void maxMat_v4(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);
    void maxMat_v2(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);
    template <typename T> void maxMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void maxScalar(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
}

void minMaxMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double, Stream& _stream, int op)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[2][7] =
    {
        {
            minMat<unsigned char>,
            minMat<signed char>,
            minMat<unsigned short>,
            minMat<short>,
            minMat<int>,
            minMat<float>,
            minMat<double>
        },
        {
            maxMat<unsigned char>,
            maxMat<signed char>,
            maxMat<unsigned short>,
            maxMat<short>,
            maxMat<int>,
            maxMat<float>,
            maxMat<double>
        }
    };

    typedef void (*opt_func_t)(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);
    static const opt_func_t funcs_v4[2] =
    {
        minMat_v4, maxMat_v4
    };
    static const opt_func_t funcs_v2[2] =
    {
        minMat_v2, maxMat_v2
    };

    const int depth = src1.depth();
    const int cn = src1.channels();

    CV_Assert( depth <= CV_64F );

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    PtrStepSzb src1_(src1.rows, src1.cols * cn, src1.data, src1.step);
    PtrStepSzb src2_(src1.rows, src1.cols * cn, src2.data, src2.step);
    PtrStepSzb dst_(src1.rows, src1.cols * cn, dst.data, dst.step);

    if (depth == CV_8U || depth == CV_16U)
    {
        const intptr_t src1ptr = reinterpret_cast<intptr_t>(src1_.data);
        const intptr_t src2ptr = reinterpret_cast<intptr_t>(src2_.data);
        const intptr_t dstptr = reinterpret_cast<intptr_t>(dst_.data);

        const bool isAllAligned = (src1ptr & 31) == 0 && (src2ptr & 31) == 0 && (dstptr & 31) == 0;

        if (isAllAligned)
        {
            if (depth == CV_8U && (src1_.cols & 3) == 0)
            {
                const int vcols = src1_.cols >> 2;

                funcs_v4[op](PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                             PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                             PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                             stream);

                return;
            }
            else if (depth == CV_16U && (src1_.cols & 1) == 0)
            {
                const int vcols = src1_.cols >> 1;

                funcs_v2[op](PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                             PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                             PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                             stream);

                return;
            }
        }
    }

    const func_t func = funcs[op][depth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, stream);
}

namespace
{
    template <typename T> double castScalar(double val)
    {
        return saturate_cast<T>(val);
    }
}

void minMaxScalar(const GpuMat& src, Scalar val, bool, GpuMat& dst, const GpuMat&, double, Stream& stream, int op)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[2][7] =
    {
        {
            minScalar<unsigned char>,
            minScalar<signed char>,
            minScalar<unsigned short>,
            minScalar<short>,
            minScalar<int>,
            minScalar<float>,
            minScalar<double>
        },
        {
            maxScalar<unsigned char>,
            maxScalar<signed char>,
            maxScalar<unsigned short>,
            maxScalar<short>,
            maxScalar<int>,
            maxScalar<float>,
            maxScalar<double>
        }
    };

    typedef double (*cast_func_t)(double sc);
    static const cast_func_t cast_func[] =
    {
        castScalar<unsigned char>, castScalar<signed char>, castScalar<unsigned short>, castScalar<short>, castScalar<int>, castScalar<float>, castScalar<double>
    };

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    funcs[op][depth](src, cast_func[depth](val[0]), dst, StreamAccessor::getStream(stream));
}

void cv::cuda::min(InputArray src1, InputArray src2, OutputArray dst, Stream& stream)
{
    arithm_op(src1, src2, dst, noArray(), 1.0, -1, stream, minMaxMat, minMaxScalar, MIN_OP);
}

void cv::cuda::max(InputArray src1, InputArray src2, OutputArray dst, Stream& stream)
{
    arithm_op(src1, src2, dst, noArray(), 1.0, -1, stream, minMaxMat, minMaxScalar, MAX_OP);
}

////////////////////////////////////////////////////////////////////////
// addWeighted

namespace arithm
{
    template <typename T1, typename T2, typename D>
    void addWeighted(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
}

void cv::cuda::addWeighted(InputArray _src1, double alpha, InputArray _src2, double beta, double gamma, OutputArray _dst, int ddepth, Stream& stream)
{
    typedef void (*func_t)(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[7][7][7] =
    {
        {
            {
                arithm::addWeighted<unsigned char, unsigned char, unsigned char >,
                arithm::addWeighted<unsigned char, unsigned char, signed char >,
                arithm::addWeighted<unsigned char, unsigned char, unsigned short>,
                arithm::addWeighted<unsigned char, unsigned char, short >,
                arithm::addWeighted<unsigned char, unsigned char, int   >,
                arithm::addWeighted<unsigned char, unsigned char, float >,
                arithm::addWeighted<unsigned char, unsigned char, double>
            },
            {
                arithm::addWeighted<unsigned char, signed char, unsigned char >,
                arithm::addWeighted<unsigned char, signed char, signed char >,
                arithm::addWeighted<unsigned char, signed char, unsigned short>,
                arithm::addWeighted<unsigned char, signed char, short >,
                arithm::addWeighted<unsigned char, signed char, int   >,
                arithm::addWeighted<unsigned char, signed char, float >,
                arithm::addWeighted<unsigned char, signed char, double>
            },
            {
                arithm::addWeighted<unsigned char, unsigned short, unsigned char >,
                arithm::addWeighted<unsigned char, unsigned short, signed char >,
                arithm::addWeighted<unsigned char, unsigned short, unsigned short>,
                arithm::addWeighted<unsigned char, unsigned short, short >,
                arithm::addWeighted<unsigned char, unsigned short, int   >,
                arithm::addWeighted<unsigned char, unsigned short, float >,
                arithm::addWeighted<unsigned char, unsigned short, double>
            },
            {
                arithm::addWeighted<unsigned char, short, unsigned char >,
                arithm::addWeighted<unsigned char, short, signed char >,
                arithm::addWeighted<unsigned char, short, unsigned short>,
                arithm::addWeighted<unsigned char, short, short >,
                arithm::addWeighted<unsigned char, short, int   >,
                arithm::addWeighted<unsigned char, short, float >,
                arithm::addWeighted<unsigned char, short, double>
            },
            {
                arithm::addWeighted<unsigned char, int, unsigned char >,
                arithm::addWeighted<unsigned char, int, signed char >,
                arithm::addWeighted<unsigned char, int, unsigned short>,
                arithm::addWeighted<unsigned char, int, short >,
                arithm::addWeighted<unsigned char, int, int   >,
                arithm::addWeighted<unsigned char, int, float >,
                arithm::addWeighted<unsigned char, int, double>
            },
            {
                arithm::addWeighted<unsigned char, float, unsigned char >,
                arithm::addWeighted<unsigned char, float, signed char >,
                arithm::addWeighted<unsigned char, float, unsigned short>,
                arithm::addWeighted<unsigned char, float, short >,
                arithm::addWeighted<unsigned char, float, int   >,
                arithm::addWeighted<unsigned char, float, float >,
                arithm::addWeighted<unsigned char, float, double>
            },
            {
                arithm::addWeighted<unsigned char, double, unsigned char >,
                arithm::addWeighted<unsigned char, double, signed char >,
                arithm::addWeighted<unsigned char, double, unsigned short>,
                arithm::addWeighted<unsigned char, double, short >,
                arithm::addWeighted<unsigned char, double, int   >,
                arithm::addWeighted<unsigned char, double, float >,
                arithm::addWeighted<unsigned char, double, double>
            }
        },
        {
            {
                0/*arithm::addWeighted<signed char, unsigned char, unsigned char >*/,
                0/*arithm::addWeighted<signed char, unsigned char, signed char >*/,
                0/*arithm::addWeighted<signed char, unsigned char, unsigned short>*/,
                0/*arithm::addWeighted<signed char, unsigned char, short >*/,
                0/*arithm::addWeighted<signed char, unsigned char, int   >*/,
                0/*arithm::addWeighted<signed char, unsigned char, float >*/,
                0/*arithm::addWeighted<signed char, unsigned char, double>*/
            },
            {
                arithm::addWeighted<signed char, signed char, unsigned char >,
                arithm::addWeighted<signed char, signed char, signed char >,
                arithm::addWeighted<signed char, signed char, unsigned short>,
                arithm::addWeighted<signed char, signed char, short >,
                arithm::addWeighted<signed char, signed char, int   >,
                arithm::addWeighted<signed char, signed char, float >,
                arithm::addWeighted<signed char, signed char, double>
            },
            {
                arithm::addWeighted<signed char, unsigned short, unsigned char >,
                arithm::addWeighted<signed char, unsigned short, signed char >,
                arithm::addWeighted<signed char, unsigned short, unsigned short>,
                arithm::addWeighted<signed char, unsigned short, short >,
                arithm::addWeighted<signed char, unsigned short, int   >,
                arithm::addWeighted<signed char, unsigned short, float >,
                arithm::addWeighted<signed char, unsigned short, double>
            },
            {
                arithm::addWeighted<signed char, short, unsigned char >,
                arithm::addWeighted<signed char, short, signed char >,
                arithm::addWeighted<signed char, short, unsigned short>,
                arithm::addWeighted<signed char, short, short >,
                arithm::addWeighted<signed char, short, int   >,
                arithm::addWeighted<signed char, short, float >,
                arithm::addWeighted<signed char, short, double>
            },
            {
                arithm::addWeighted<signed char, int, unsigned char >,
                arithm::addWeighted<signed char, int, signed char >,
                arithm::addWeighted<signed char, int, unsigned short>,
                arithm::addWeighted<signed char, int, short >,
                arithm::addWeighted<signed char, int, int   >,
                arithm::addWeighted<signed char, int, float >,
                arithm::addWeighted<signed char, int, double>
            },
            {
                arithm::addWeighted<signed char, float, unsigned char >,
                arithm::addWeighted<signed char, float, signed char >,
                arithm::addWeighted<signed char, float, unsigned short>,
                arithm::addWeighted<signed char, float, short >,
                arithm::addWeighted<signed char, float, int   >,
                arithm::addWeighted<signed char, float, float >,
                arithm::addWeighted<signed char, float, double>
            },
            {
                arithm::addWeighted<signed char, double, unsigned char >,
                arithm::addWeighted<signed char, double, signed char >,
                arithm::addWeighted<signed char, double, unsigned short>,
                arithm::addWeighted<signed char, double, short >,
                arithm::addWeighted<signed char, double, int   >,
                arithm::addWeighted<signed char, double, float >,
                arithm::addWeighted<signed char, double, double>
            }
        },
        {
            {
                0/*arithm::addWeighted<unsigned short, unsigned char, unsigned char >*/,
                0/*arithm::addWeighted<unsigned short, unsigned char, signed char >*/,
                0/*arithm::addWeighted<unsigned short, unsigned char, unsigned short>*/,
                0/*arithm::addWeighted<unsigned short, unsigned char, short >*/,
                0/*arithm::addWeighted<unsigned short, unsigned char, int   >*/,
                0/*arithm::addWeighted<unsigned short, unsigned char, float >*/,
                0/*arithm::addWeighted<unsigned short, unsigned char, double>*/
            },
            {
                0/*arithm::addWeighted<unsigned short, signed char, unsigned char >*/,
                0/*arithm::addWeighted<unsigned short, signed char, signed char >*/,
                0/*arithm::addWeighted<unsigned short, signed char, unsigned short>*/,
                0/*arithm::addWeighted<unsigned short, signed char, short >*/,
                0/*arithm::addWeighted<unsigned short, signed char, int   >*/,
                0/*arithm::addWeighted<unsigned short, signed char, float >*/,
                0/*arithm::addWeighted<unsigned short, signed char, double>*/
            },
            {
                arithm::addWeighted<unsigned short, unsigned short, unsigned char >,
                arithm::addWeighted<unsigned short, unsigned short, signed char >,
                arithm::addWeighted<unsigned short, unsigned short, unsigned short>,
                arithm::addWeighted<unsigned short, unsigned short, short >,
                arithm::addWeighted<unsigned short, unsigned short, int   >,
                arithm::addWeighted<unsigned short, unsigned short, float >,
                arithm::addWeighted<unsigned short, unsigned short, double>
            },
            {
                arithm::addWeighted<unsigned short, short, unsigned char >,
                arithm::addWeighted<unsigned short, short, signed char >,
                arithm::addWeighted<unsigned short, short, unsigned short>,
                arithm::addWeighted<unsigned short, short, short >,
                arithm::addWeighted<unsigned short, short, int   >,
                arithm::addWeighted<unsigned short, short, float >,
                arithm::addWeighted<unsigned short, short, double>
            },
            {
                arithm::addWeighted<unsigned short, int, unsigned char >,
                arithm::addWeighted<unsigned short, int, signed char >,
                arithm::addWeighted<unsigned short, int, unsigned short>,
                arithm::addWeighted<unsigned short, int, short >,
                arithm::addWeighted<unsigned short, int, int   >,
                arithm::addWeighted<unsigned short, int, float >,
                arithm::addWeighted<unsigned short, int, double>
            },
            {
                arithm::addWeighted<unsigned short, float, unsigned char >,
                arithm::addWeighted<unsigned short, float, signed char >,
                arithm::addWeighted<unsigned short, float, unsigned short>,
                arithm::addWeighted<unsigned short, float, short >,
                arithm::addWeighted<unsigned short, float, int   >,
                arithm::addWeighted<unsigned short, float, float >,
                arithm::addWeighted<unsigned short, float, double>
            },
            {
                arithm::addWeighted<unsigned short, double, unsigned char >,
                arithm::addWeighted<unsigned short, double, signed char >,
                arithm::addWeighted<unsigned short, double, unsigned short>,
                arithm::addWeighted<unsigned short, double, short >,
                arithm::addWeighted<unsigned short, double, int   >,
                arithm::addWeighted<unsigned short, double, float >,
                arithm::addWeighted<unsigned short, double, double>
            }
        },
        {
            {
                0/*arithm::addWeighted<short, unsigned char, unsigned char >*/,
                0/*arithm::addWeighted<short, unsigned char, signed char >*/,
                0/*arithm::addWeighted<short, unsigned char, unsigned short>*/,
                0/*arithm::addWeighted<short, unsigned char, short >*/,
                0/*arithm::addWeighted<short, unsigned char, int   >*/,
                0/*arithm::addWeighted<short, unsigned char, float >*/,
                0/*arithm::addWeighted<short, unsigned char, double>*/
            },
            {
                0/*arithm::addWeighted<short, signed char, unsigned char >*/,
                0/*arithm::addWeighted<short, signed char, signed char >*/,
                0/*arithm::addWeighted<short, signed char, unsigned short>*/,
                0/*arithm::addWeighted<short, signed char, short >*/,
                0/*arithm::addWeighted<short, signed char, int   >*/,
                0/*arithm::addWeighted<short, signed char, float >*/,
                0/*arithm::addWeighted<short, signed char, double>*/
            },
            {
                0/*arithm::addWeighted<short, unsigned short, unsigned char >*/,
                0/*arithm::addWeighted<short, unsigned short, signed char >*/,
                0/*arithm::addWeighted<short, unsigned short, unsigned short>*/,
                0/*arithm::addWeighted<short, unsigned short, short >*/,
                0/*arithm::addWeighted<short, unsigned short, int   >*/,
                0/*arithm::addWeighted<short, unsigned short, float >*/,
                0/*arithm::addWeighted<short, unsigned short, double>*/
            },
            {
                arithm::addWeighted<short, short, unsigned char >,
                arithm::addWeighted<short, short, signed char >,
                arithm::addWeighted<short, short, unsigned short>,
                arithm::addWeighted<short, short, short >,
                arithm::addWeighted<short, short, int   >,
                arithm::addWeighted<short, short, float >,
                arithm::addWeighted<short, short, double>
            },
            {
                arithm::addWeighted<short, int, unsigned char >,
                arithm::addWeighted<short, int, signed char >,
                arithm::addWeighted<short, int, unsigned short>,
                arithm::addWeighted<short, int, short >,
                arithm::addWeighted<short, int, int   >,
                arithm::addWeighted<short, int, float >,
                arithm::addWeighted<short, int, double>
            },
            {
                arithm::addWeighted<short, float, unsigned char >,
                arithm::addWeighted<short, float, signed char >,
                arithm::addWeighted<short, float, unsigned short>,
                arithm::addWeighted<short, float, short >,
                arithm::addWeighted<short, float, int   >,
                arithm::addWeighted<short, float, float >,
                arithm::addWeighted<short, float, double>
            },
            {
                arithm::addWeighted<short, double, unsigned char >,
                arithm::addWeighted<short, double, signed char >,
                arithm::addWeighted<short, double, unsigned short>,
                arithm::addWeighted<short, double, short >,
                arithm::addWeighted<short, double, int   >,
                arithm::addWeighted<short, double, float >,
                arithm::addWeighted<short, double, double>
            }
        },
        {
            {
                0/*arithm::addWeighted<int, unsigned char, unsigned char >*/,
                0/*arithm::addWeighted<int, unsigned char, signed char >*/,
                0/*arithm::addWeighted<int, unsigned char, unsigned short>*/,
                0/*arithm::addWeighted<int, unsigned char, short >*/,
                0/*arithm::addWeighted<int, unsigned char, int   >*/,
                0/*arithm::addWeighted<int, unsigned char, float >*/,
                0/*arithm::addWeighted<int, unsigned char, double>*/
            },
            {
                0/*arithm::addWeighted<int, signed char, unsigned char >*/,
                0/*arithm::addWeighted<int, signed char, signed char >*/,
                0/*arithm::addWeighted<int, signed char, unsigned short>*/,
                0/*arithm::addWeighted<int, signed char, short >*/,
                0/*arithm::addWeighted<int, signed char, int   >*/,
                0/*arithm::addWeighted<int, signed char, float >*/,
                0/*arithm::addWeighted<int, signed char, double>*/
            },
            {
                0/*arithm::addWeighted<int, unsigned short, unsigned char >*/,
                0/*arithm::addWeighted<int, unsigned short, signed char >*/,
                0/*arithm::addWeighted<int, unsigned short, unsigned short>*/,
                0/*arithm::addWeighted<int, unsigned short, short >*/,
                0/*arithm::addWeighted<int, unsigned short, int   >*/,
                0/*arithm::addWeighted<int, unsigned short, float >*/,
                0/*arithm::addWeighted<int, unsigned short, double>*/
            },
            {
                0/*arithm::addWeighted<int, short, unsigned char >*/,
                0/*arithm::addWeighted<int, short, signed char >*/,
                0/*arithm::addWeighted<int, short, unsigned short>*/,
                0/*arithm::addWeighted<int, short, short >*/,
                0/*arithm::addWeighted<int, short, int   >*/,
                0/*arithm::addWeighted<int, short, float >*/,
                0/*arithm::addWeighted<int, short, double>*/
            },
            {
                arithm::addWeighted<int, int, unsigned char >,
                arithm::addWeighted<int, int, signed char >,
                arithm::addWeighted<int, int, unsigned short>,
                arithm::addWeighted<int, int, short >,
                arithm::addWeighted<int, int, int   >,
                arithm::addWeighted<int, int, float >,
                arithm::addWeighted<int, int, double>
            },
            {
                arithm::addWeighted<int, float, unsigned char >,
                arithm::addWeighted<int, float, signed char >,
                arithm::addWeighted<int, float, unsigned short>,
                arithm::addWeighted<int, float, short >,
                arithm::addWeighted<int, float, int   >,
                arithm::addWeighted<int, float, float >,
                arithm::addWeighted<int, float, double>
            },
            {
                arithm::addWeighted<int, double, unsigned char >,
                arithm::addWeighted<int, double, signed char >,
                arithm::addWeighted<int, double, unsigned short>,
                arithm::addWeighted<int, double, short >,
                arithm::addWeighted<int, double, int   >,
                arithm::addWeighted<int, double, float >,
                arithm::addWeighted<int, double, double>
            }
        },
        {
            {
                0/*arithm::addWeighted<float, unsigned char, unsigned char >*/,
                0/*arithm::addWeighted<float, unsigned char, signed char >*/,
                0/*arithm::addWeighted<float, unsigned char, unsigned short>*/,
                0/*arithm::addWeighted<float, unsigned char, short >*/,
                0/*arithm::addWeighted<float, unsigned char, int   >*/,
                0/*arithm::addWeighted<float, unsigned char, float >*/,
                0/*arithm::addWeighted<float, unsigned char, double>*/
            },
            {
                0/*arithm::addWeighted<float, signed char, unsigned char >*/,
                0/*arithm::addWeighted<float, signed char, signed char >*/,
                0/*arithm::addWeighted<float, signed char, unsigned short>*/,
                0/*arithm::addWeighted<float, signed char, short >*/,
                0/*arithm::addWeighted<float, signed char, int   >*/,
                0/*arithm::addWeighted<float, signed char, float >*/,
                0/*arithm::addWeighted<float, signed char, double>*/
            },
            {
                0/*arithm::addWeighted<float, unsigned short, unsigned char >*/,
                0/*arithm::addWeighted<float, unsigned short, signed char >*/,
                0/*arithm::addWeighted<float, unsigned short, unsigned short>*/,
                0/*arithm::addWeighted<float, unsigned short, short >*/,
                0/*arithm::addWeighted<float, unsigned short, int   >*/,
                0/*arithm::addWeighted<float, unsigned short, float >*/,
                0/*arithm::addWeighted<float, unsigned short, double>*/
            },
            {
                0/*arithm::addWeighted<float, short, unsigned char >*/,
                0/*arithm::addWeighted<float, short, signed char >*/,
                0/*arithm::addWeighted<float, short, unsigned short>*/,
                0/*arithm::addWeighted<float, short, short >*/,
                0/*arithm::addWeighted<float, short, int   >*/,
                0/*arithm::addWeighted<float, short, float >*/,
                0/*arithm::addWeighted<float, short, double>*/
            },
            {
                0/*arithm::addWeighted<float, int, unsigned char >*/,
                0/*arithm::addWeighted<float, int, signed char >*/,
                0/*arithm::addWeighted<float, int, unsigned short>*/,
                0/*arithm::addWeighted<float, int, short >*/,
                0/*arithm::addWeighted<float, int, int   >*/,
                0/*arithm::addWeighted<float, int, float >*/,
                0/*arithm::addWeighted<float, int, double>*/
            },
            {
                arithm::addWeighted<float, float, unsigned char >,
                arithm::addWeighted<float, float, signed char >,
                arithm::addWeighted<float, float, unsigned short>,
                arithm::addWeighted<float, float, short >,
                arithm::addWeighted<float, float, int   >,
                arithm::addWeighted<float, float, float >,
                arithm::addWeighted<float, float, double>
            },
            {
                arithm::addWeighted<float, double, unsigned char >,
                arithm::addWeighted<float, double, signed char >,
                arithm::addWeighted<float, double, unsigned short>,
                arithm::addWeighted<float, double, short >,
                arithm::addWeighted<float, double, int   >,
                arithm::addWeighted<float, double, float >,
                arithm::addWeighted<float, double, double>
            }
        },
        {
            {
                0/*arithm::addWeighted<double, unsigned char, unsigned char >*/,
                0/*arithm::addWeighted<double, unsigned char, signed char >*/,
                0/*arithm::addWeighted<double, unsigned char, unsigned short>*/,
                0/*arithm::addWeighted<double, unsigned char, short >*/,
                0/*arithm::addWeighted<double, unsigned char, int   >*/,
                0/*arithm::addWeighted<double, unsigned char, float >*/,
                0/*arithm::addWeighted<double, unsigned char, double>*/
            },
            {
                0/*arithm::addWeighted<double, signed char, unsigned char >*/,
                0/*arithm::addWeighted<double, signed char, signed char >*/,
                0/*arithm::addWeighted<double, signed char, unsigned short>*/,
                0/*arithm::addWeighted<double, signed char, short >*/,
                0/*arithm::addWeighted<double, signed char, int   >*/,
                0/*arithm::addWeighted<double, signed char, float >*/,
                0/*arithm::addWeighted<double, signed char, double>*/
            },
            {
                0/*arithm::addWeighted<double, unsigned short, unsigned char >*/,
                0/*arithm::addWeighted<double, unsigned short, signed char >*/,
                0/*arithm::addWeighted<double, unsigned short, unsigned short>*/,
                0/*arithm::addWeighted<double, unsigned short, short >*/,
                0/*arithm::addWeighted<double, unsigned short, int   >*/,
                0/*arithm::addWeighted<double, unsigned short, float >*/,
                0/*arithm::addWeighted<double, unsigned short, double>*/
            },
            {
                0/*arithm::addWeighted<double, short, unsigned char >*/,
                0/*arithm::addWeighted<double, short, signed char >*/,
                0/*arithm::addWeighted<double, short, unsigned short>*/,
                0/*arithm::addWeighted<double, short, short >*/,
                0/*arithm::addWeighted<double, short, int   >*/,
                0/*arithm::addWeighted<double, short, float >*/,
                0/*arithm::addWeighted<double, short, double>*/
            },
            {
                0/*arithm::addWeighted<double, int, unsigned char >*/,
                0/*arithm::addWeighted<double, int, signed char >*/,
                0/*arithm::addWeighted<double, int, unsigned short>*/,
                0/*arithm::addWeighted<double, int, short >*/,
                0/*arithm::addWeighted<double, int, int   >*/,
                0/*arithm::addWeighted<double, int, float >*/,
                0/*arithm::addWeighted<double, int, double>*/
            },
            {
                0/*arithm::addWeighted<double, float, unsigned char >*/,
                0/*arithm::addWeighted<double, float, signed char >*/,
                0/*arithm::addWeighted<double, float, unsigned short>*/,
                0/*arithm::addWeighted<double, float, short >*/,
                0/*arithm::addWeighted<double, float, int   >*/,
                0/*arithm::addWeighted<double, float, float >*/,
                0/*arithm::addWeighted<double, float, double>*/
            },
            {
                arithm::addWeighted<double, double, unsigned char >,
                arithm::addWeighted<double, double, signed char >,
                arithm::addWeighted<double, double, unsigned short>,
                arithm::addWeighted<double, double, short >,
                arithm::addWeighted<double, double, int   >,
                arithm::addWeighted<double, double, float >,
                arithm::addWeighted<double, double, double>
            }
        }
    };

    GpuMat src1 = _src1.getGpuMat();
    GpuMat src2 = _src2.getGpuMat();

    int sdepth1 = src1.depth();
    int sdepth2 = src2.depth();
    ddepth = ddepth >= 0 ? CV_MAT_DEPTH(ddepth) : std::max(sdepth1, sdepth2);
    const int cn = src1.channels();

    CV_Assert( src2.size() == src1.size() && src2.channels() == cn );
    CV_Assert( sdepth1 <= CV_64F && sdepth2 <= CV_64F && ddepth <= CV_64F );

    if (sdepth1 == CV_64F || sdepth2 == CV_64F || ddepth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    _dst.create(src1.size(), CV_MAKE_TYPE(ddepth, cn));
    GpuMat dst = _dst.getGpuMat();

    PtrStepSzb src1_(src1.rows, src1.cols * cn, src1.data, src1.step);
    PtrStepSzb src2_(src1.rows, src1.cols * cn, src2.data, src2.step);
    PtrStepSzb dst_(src1.rows, src1.cols * cn, dst.data, dst.step);

    if (sdepth1 > sdepth2)
    {
        std::swap(src1_.data, src2_.data);
        std::swap(src1_.step, src2_.step);
        std::swap(alpha, beta);
        std::swap(sdepth1, sdepth2);
    }

    const func_t func = funcs[sdepth1][sdepth2][ddepth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, alpha, src2_, beta, gamma, dst_, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// threshold

namespace arithm
{
    template <typename T>
    void threshold(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
}

double cv::cuda::threshold(InputArray _src, OutputArray _dst, double thresh, double maxVal, int type, Stream& _stream)
{
    GpuMat src = _src.getGpuMat();

    const int depth = src.depth();

    CV_Assert( src.channels() == 1 && depth <= CV_64F );
    CV_Assert( type <= 4/*THRESH_TOZERO_INV*/ );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    if (src.type() == CV_32FC1 && type == 2/*THRESH_TRUNC*/)
    {
        NppStreamHandler h(stream);

        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        nppSafeCall( nppiThreshold_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step),
            dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz, static_cast<Npp32f>(thresh), NPP_CMP_GREATER) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    else
    {
        typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
        static const func_t funcs[] =
        {
            arithm::threshold<unsigned char>,
            arithm::threshold<signed char>,
            arithm::threshold<unsigned short>,
            arithm::threshold<short>,
            arithm::threshold<int>,
            arithm::threshold<float>,
            arithm::threshold<double>
        };

        if (depth != CV_32F && depth != CV_64F)
        {
            thresh = cvFloor(thresh);
            maxVal = cvRound(maxVal);
        }

        funcs[depth](src, dst, thresh, maxVal, type, stream);
    }

    return thresh;
}

////////////////////////////////////////////////////////////////////////
// NPP magnitide

namespace
{
    typedef NppStatus (*nppMagnitude_t)(const Npp32fc* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep, NppiSize oSizeROI);

    void npp_magnitude(const GpuMat& src, GpuMat& dst, nppMagnitude_t func, cudaStream_t stream)
    {
        CV_Assert(src.type() == CV_32FC2);

        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        NppStreamHandler h(stream);

        nppSafeCall( func(src.ptr<Npp32fc>(), static_cast<int>(src.step), dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

void cv::cuda::magnitude(InputArray _src, OutputArray _dst, Stream& stream)
{
    GpuMat src = _src.getGpuMat();

    _dst.create(src.size(), CV_32FC1);
    GpuMat dst = _dst.getGpuMat();

    npp_magnitude(src, dst, nppiMagnitude_32fc32f_C1R, StreamAccessor::getStream(stream));
}

void cv::cuda::magnitudeSqr(InputArray _src, OutputArray _dst, Stream& stream)
{
    GpuMat src = _src.getGpuMat();

    _dst.create(src.size(), CV_32FC1);
    GpuMat dst = _dst.getGpuMat();

    npp_magnitude(src, dst, nppiMagnitudeSqr_32fc32f_C1R, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// Polar <-> Cart

namespace cv { namespace cuda { namespace device
{
    namespace mathfunc
    {
        void cartToPolar_gpu(PtrStepSzf x, PtrStepSzf y, PtrStepSzf mag, bool magSqr, PtrStepSzf angle, bool angleInDegrees, cudaStream_t stream);
        void polarToCart_gpu(PtrStepSzf mag, PtrStepSzf angle, PtrStepSzf x, PtrStepSzf y, bool angleInDegrees, cudaStream_t stream);
    }
}}}

namespace
{
    void cartToPolar_caller(const GpuMat& x, const GpuMat& y, GpuMat* mag, bool magSqr, GpuMat* angle, bool angleInDegrees, cudaStream_t stream)
    {
        using namespace ::cv::cuda::device::mathfunc;

        CV_Assert(x.size() == y.size() && x.type() == y.type());
        CV_Assert(x.depth() == CV_32F);

        GpuMat x1cn = x.reshape(1);
        GpuMat y1cn = y.reshape(1);
        GpuMat mag1cn = mag ? mag->reshape(1) : GpuMat();
        GpuMat angle1cn = angle ? angle->reshape(1) : GpuMat();

        cartToPolar_gpu(x1cn, y1cn, mag1cn, magSqr, angle1cn, angleInDegrees, stream);
    }

    void polarToCart_caller(const GpuMat& mag, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, cudaStream_t stream)
    {
        using namespace ::cv::cuda::device::mathfunc;

        CV_Assert((mag.empty() || mag.size() == angle.size()) && mag.type() == angle.type());
        CV_Assert(mag.depth() == CV_32F);

        GpuMat mag1cn = mag.reshape(1);
        GpuMat angle1cn = angle.reshape(1);
        GpuMat x1cn = x.reshape(1);
        GpuMat y1cn = y.reshape(1);

        polarToCart_gpu(mag1cn, angle1cn, x1cn, y1cn, angleInDegrees, stream);
    }
}

void cv::cuda::magnitude(InputArray _x, InputArray _y, OutputArray _dst, Stream& stream)
{
    GpuMat x = _x.getGpuMat();
    GpuMat y = _y.getGpuMat();

    _dst.create(x.size(), CV_32FC1);
    GpuMat dst = _dst.getGpuMat();

    cartToPolar_caller(x, y, &dst, false, 0, false, StreamAccessor::getStream(stream));
}

void cv::cuda::magnitudeSqr(InputArray _x, InputArray _y, OutputArray _dst, Stream& stream)
{
    GpuMat x = _x.getGpuMat();
    GpuMat y = _y.getGpuMat();

    _dst.create(x.size(), CV_32FC1);
    GpuMat dst = _dst.getGpuMat();

    cartToPolar_caller(x, y, &dst, true, 0, false, StreamAccessor::getStream(stream));
}

void cv::cuda::phase(InputArray _x, InputArray _y, OutputArray _dst, bool angleInDegrees, Stream& stream)
{
    GpuMat x = _x.getGpuMat();
    GpuMat y = _y.getGpuMat();

    _dst.create(x.size(), CV_32FC1);
    GpuMat dst = _dst.getGpuMat();

    cartToPolar_caller(x, y, 0, false, &dst, angleInDegrees, StreamAccessor::getStream(stream));
}

void cv::cuda::cartToPolar(InputArray _x, InputArray _y, OutputArray _mag, OutputArray _angle, bool angleInDegrees, Stream& stream)
{
    GpuMat x = _x.getGpuMat();
    GpuMat y = _y.getGpuMat();

    _mag.create(x.size(), CV_32FC1);
    GpuMat mag = _mag.getGpuMat();

    _angle.create(x.size(), CV_32FC1);
    GpuMat angle = _angle.getGpuMat();

    cartToPolar_caller(x, y, &mag, false, &angle, angleInDegrees, StreamAccessor::getStream(stream));
}

void cv::cuda::polarToCart(InputArray _mag, InputArray _angle, OutputArray _x, OutputArray _y, bool angleInDegrees, Stream& stream)
{
    GpuMat mag = _mag.getGpuMat();
    GpuMat angle = _angle.getGpuMat();

    _x.create(mag.size(), CV_32FC1);
    GpuMat x = _x.getGpuMat();

    _y.create(mag.size(), CV_32FC1);
    GpuMat y = _y.getGpuMat();

    polarToCart_caller(mag, angle, x, y, angleInDegrees, StreamAccessor::getStream(stream));
}

#endif
