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

void minMaxMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double, Stream& stream, int op);

void minMaxScalar(const GpuMat& src, cv::Scalar value, bool, GpuMat& dst, const GpuMat&, double, Stream& stream, int op);

void cv::cuda::min(InputArray src1, InputArray src2, OutputArray dst, Stream& stream)
{
    arithm_op(src1, src2, dst, noArray(), 1.0, -1, stream, minMaxMat, minMaxScalar, MIN_OP);
}

void cv::cuda::max(InputArray src1, InputArray src2, OutputArray dst, Stream& stream)
{
    arithm_op(src1, src2, dst, noArray(), 1.0, -1, stream, minMaxMat, minMaxScalar, MAX_OP);
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

#endif
