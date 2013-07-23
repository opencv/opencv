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
// Basic arithmetical operations (add subtract multiply divide)

namespace
{
    template<int DEPTH> struct NppTypeTraits;
    template<> struct NppTypeTraits<CV_8U>  { typedef Npp8u npp_t; };
    template<> struct NppTypeTraits<CV_8S>  { typedef Npp8s npp_t; };
    template<> struct NppTypeTraits<CV_16U> { typedef Npp16u npp_t; };
    template<> struct NppTypeTraits<CV_16S> { typedef Npp16s npp_t; typedef Npp16sc npp_complex_type; };
    template<> struct NppTypeTraits<CV_32S> { typedef Npp32s npp_t; typedef Npp32sc npp_complex_type; };
    template<> struct NppTypeTraits<CV_32F> { typedef Npp32f npp_t; typedef Npp32fc npp_complex_type; };
    template<> struct NppTypeTraits<CV_64F> { typedef Npp64f npp_t; typedef Npp64fc npp_complex_type; };

    template<int DEPTH, int cn> struct NppArithmScalarFunc
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        typedef NppStatus (*func_ptr)(const npp_t* pSrc1, int nSrc1Step, const npp_t* pConstants,
            npp_t* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    };
    template<int DEPTH> struct NppArithmScalarFunc<DEPTH, 1>
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        typedef NppStatus (*func_ptr)(const npp_t* pSrc1, int nSrc1Step, const npp_t pConstants,
            npp_t* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    };
    template<int DEPTH> struct NppArithmScalarFunc<DEPTH, 2>
    {
        typedef typename NppTypeTraits<DEPTH>::npp_complex_type npp_complex_type;

        typedef NppStatus (*func_ptr)(const npp_complex_type* pSrc1, int nSrc1Step, const npp_complex_type pConstants,
            npp_complex_type* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    };
    template<int cn> struct NppArithmScalarFunc<CV_32F, cn>
    {
        typedef NppStatus (*func_ptr)(const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pConstants, Npp32f* pDst, int nDstStep, NppiSize oSizeROI);
    };
    template<> struct NppArithmScalarFunc<CV_32F, 1>
    {
        typedef NppStatus (*func_ptr)(const Npp32f* pSrc1, int nSrc1Step, const Npp32f pConstants, Npp32f* pDst, int nDstStep, NppiSize oSizeROI);
    };
    template<> struct NppArithmScalarFunc<CV_32F, 2>
    {
        typedef NppStatus (*func_ptr)(const Npp32fc* pSrc1, int nSrc1Step, const Npp32fc pConstants, Npp32fc* pDst, int nDstStep, NppiSize oSizeROI);
    };

    template<int DEPTH, int cn, typename NppArithmScalarFunc<DEPTH, cn>::func_ptr func> struct NppArithmScalar
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        static void call(const PtrStepSzb src, Scalar sc, PtrStepb dst, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            const npp_t pConstants[] = { saturate_cast<npp_t>(sc.val[0]), saturate_cast<npp_t>(sc.val[1]), saturate_cast<npp_t>(sc.val[2]), saturate_cast<npp_t>(sc.val[3]) };

            nppSafeCall( func((const npp_t*)src.data, static_cast<int>(src.step), pConstants, (npp_t*)dst.data, static_cast<int>(dst.step), sz, 0) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int DEPTH, typename NppArithmScalarFunc<DEPTH, 1>::func_ptr func> struct NppArithmScalar<DEPTH, 1, func>
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        static void call(const PtrStepSzb src, Scalar sc, PtrStepb dst, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            nppSafeCall( func((const npp_t*)src.data, static_cast<int>(src.step), saturate_cast<npp_t>(sc.val[0]), (npp_t*)dst.data, static_cast<int>(dst.step), sz, 0) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int DEPTH, typename NppArithmScalarFunc<DEPTH, 2>::func_ptr func> struct NppArithmScalar<DEPTH, 2, func>
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;
        typedef typename NppTypeTraits<DEPTH>::npp_complex_type npp_complex_type;

        static void call(const PtrStepSzb src, Scalar sc, PtrStepb dst, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            npp_complex_type nConstant;
            nConstant.re = saturate_cast<npp_t>(sc.val[0]);
            nConstant.im = saturate_cast<npp_t>(sc.val[1]);

            nppSafeCall( func((const npp_complex_type*)src.data, static_cast<int>(src.step), nConstant,
                              (npp_complex_type*)dst.data, static_cast<int>(dst.step), sz, 0) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<int cn, typename NppArithmScalarFunc<CV_32F, cn>::func_ptr func> struct NppArithmScalar<CV_32F, cn, func>
    {
        typedef typename NppTypeTraits<CV_32F>::npp_t npp_t;

        static void call(const PtrStepSzb src, Scalar sc, PtrStepb dst, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            const Npp32f pConstants[] = { saturate_cast<Npp32f>(sc.val[0]), saturate_cast<Npp32f>(sc.val[1]), saturate_cast<Npp32f>(sc.val[2]), saturate_cast<Npp32f>(sc.val[3]) };

            nppSafeCall( func((const npp_t*)src.data, static_cast<int>(src.step), pConstants, (npp_t*)dst.data, static_cast<int>(dst.step), sz) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<typename NppArithmScalarFunc<CV_32F, 1>::func_ptr func> struct NppArithmScalar<CV_32F, 1, func>
    {
        typedef typename NppTypeTraits<CV_32F>::npp_t npp_t;

        static void call(const PtrStepSzb src, Scalar sc, PtrStepb dst, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            nppSafeCall( func((const npp_t*)src.data, static_cast<int>(src.step), saturate_cast<Npp32f>(sc.val[0]), (npp_t*)dst.data, static_cast<int>(dst.step), sz) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template<typename NppArithmScalarFunc<CV_32F, 2>::func_ptr func> struct NppArithmScalar<CV_32F, 2, func>
    {
        typedef typename NppTypeTraits<CV_32F>::npp_t npp_t;
        typedef typename NppTypeTraits<CV_32F>::npp_complex_type npp_complex_type;

        static void call(const PtrStepSzb src, Scalar sc, PtrStepb dst, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize sz;
            sz.width = src.cols;
            sz.height = src.rows;

            Npp32fc nConstant;
            nConstant.re = saturate_cast<Npp32f>(sc.val[0]);
            nConstant.im = saturate_cast<Npp32f>(sc.val[1]);

            nppSafeCall( func((const npp_complex_type*)src.data, static_cast<int>(src.step), nConstant, (npp_complex_type*)dst.data, static_cast<int>(dst.step), sz) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

////////////////////////////////////////////////////////////////////////
// add

namespace arithm
{
    void addMat_v4(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);
    void addMat_v2(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);

    template <typename T, typename D>
    void addMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

static void addMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, double, Stream& _stream, int)
{
    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            arithm::addMat<unsigned char, unsigned char>,
            arithm::addMat<unsigned char, signed char>,
            arithm::addMat<unsigned char, unsigned short>,
            arithm::addMat<unsigned char, short>,
            arithm::addMat<unsigned char, int>,
            arithm::addMat<unsigned char, float>,
            arithm::addMat<unsigned char, double>
        },
        {
            arithm::addMat<signed char, unsigned char>,
            arithm::addMat<signed char, signed char>,
            arithm::addMat<signed char, unsigned short>,
            arithm::addMat<signed char, short>,
            arithm::addMat<signed char, int>,
            arithm::addMat<signed char, float>,
            arithm::addMat<signed char, double>
        },
        {
            0 /*arithm::addMat<unsigned short, unsigned char>*/,
            0 /*arithm::addMat<unsigned short, signed char>*/,
            arithm::addMat<unsigned short, unsigned short>,
            arithm::addMat<unsigned short, short>,
            arithm::addMat<unsigned short, int>,
            arithm::addMat<unsigned short, float>,
            arithm::addMat<unsigned short, double>
        },
        {
            0 /*arithm::addMat<short, unsigned char>*/,
            0 /*arithm::addMat<short, signed char>*/,
            arithm::addMat<short, unsigned short>,
            arithm::addMat<short, short>,
            arithm::addMat<short, int>,
            arithm::addMat<short, float>,
            arithm::addMat<short, double>
        },
        {
            0 /*arithm::addMat<int, unsigned char>*/,
            0 /*arithm::addMat<int, signed char>*/,
            0 /*arithm::addMat<int, unsigned short>*/,
            0 /*arithm::addMat<int, short>*/,
            arithm::addMat<int, int>,
            arithm::addMat<int, float>,
            arithm::addMat<int, double>
        },
        {
            0 /*arithm::addMat<float, unsigned char>*/,
            0 /*arithm::addMat<float, signed char>*/,
            0 /*arithm::addMat<float, unsigned short>*/,
            0 /*arithm::addMat<float, short>*/,
            0 /*arithm::addMat<float, int>*/,
            arithm::addMat<float, float>,
            arithm::addMat<float, double>
        },
        {
            0 /*arithm::addMat<double, unsigned char>*/,
            0 /*arithm::addMat<double, signed char>*/,
            0 /*arithm::addMat<double, unsigned short>*/,
            0 /*arithm::addMat<double, short>*/,
            0 /*arithm::addMat<double, int>*/,
            0 /*arithm::addMat<double, float>*/,
            arithm::addMat<double, double>
        }
    };

    const int sdepth = src1.depth();
    const int ddepth = dst.depth();
    const int cn = src1.channels();

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    PtrStepSzb src1_(src1.rows, src1.cols * cn, src1.data, src1.step);
    PtrStepSzb src2_(src1.rows, src1.cols * cn, src2.data, src2.step);
    PtrStepSzb dst_(src1.rows, src1.cols * cn, dst.data, dst.step);

    if (mask.empty() && (sdepth == CV_8U || sdepth == CV_16U) && ddepth == sdepth)
    {
        const intptr_t src1ptr = reinterpret_cast<intptr_t>(src1_.data);
        const intptr_t src2ptr = reinterpret_cast<intptr_t>(src2_.data);
        const intptr_t dstptr = reinterpret_cast<intptr_t>(dst_.data);

        const bool isAllAligned = (src1ptr & 31) == 0 && (src2ptr & 31) == 0 && (dstptr & 31) == 0;

        if (isAllAligned)
        {
            if (sdepth == CV_8U && (src1_.cols & 3) == 0)
            {
                const int vcols = src1_.cols >> 2;

                arithm::addMat_v4(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                                  PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                                  PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                                  stream);

                return;
            }
            else if (sdepth == CV_16U && (src1_.cols & 1) == 0)
            {
                const int vcols = src1_.cols >> 1;

                arithm::addMat_v2(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                                  PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                                  PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                                  stream);

                return;
            }
        }
    }

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, mask, stream);
}

namespace arithm
{
    template <typename T, typename S, typename D>
    void addScalar(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

static void addScalar(const GpuMat& src, Scalar val, bool, GpuMat& dst, const GpuMat& mask, double, Stream& _stream, int)
{
    typedef void (*func_t)(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            arithm::addScalar<unsigned char, float, unsigned char>,
            arithm::addScalar<unsigned char, float, signed char>,
            arithm::addScalar<unsigned char, float, unsigned short>,
            arithm::addScalar<unsigned char, float, short>,
            arithm::addScalar<unsigned char, float, int>,
            arithm::addScalar<unsigned char, float, float>,
            arithm::addScalar<unsigned char, double, double>
        },
        {
            arithm::addScalar<signed char, float, unsigned char>,
            arithm::addScalar<signed char, float, signed char>,
            arithm::addScalar<signed char, float, unsigned short>,
            arithm::addScalar<signed char, float, short>,
            arithm::addScalar<signed char, float, int>,
            arithm::addScalar<signed char, float, float>,
            arithm::addScalar<signed char, double, double>
        },
        {
            0 /*arithm::addScalar<unsigned short, float, unsigned char>*/,
            0 /*arithm::addScalar<unsigned short, float, signed char>*/,
            arithm::addScalar<unsigned short, float, unsigned short>,
            arithm::addScalar<unsigned short, float, short>,
            arithm::addScalar<unsigned short, float, int>,
            arithm::addScalar<unsigned short, float, float>,
            arithm::addScalar<unsigned short, double, double>
        },
        {
            0 /*arithm::addScalar<short, float, unsigned char>*/,
            0 /*arithm::addScalar<short, float, signed char>*/,
            arithm::addScalar<short, float, unsigned short>,
            arithm::addScalar<short, float, short>,
            arithm::addScalar<short, float, int>,
            arithm::addScalar<short, float, float>,
            arithm::addScalar<short, double, double>
        },
        {
            0 /*arithm::addScalar<int, float, unsigned char>*/,
            0 /*arithm::addScalar<int, float, signed char>*/,
            0 /*arithm::addScalar<int, float, unsigned short>*/,
            0 /*arithm::addScalar<int, float, short>*/,
            arithm::addScalar<int, float, int>,
            arithm::addScalar<int, float, float>,
            arithm::addScalar<int, double, double>
        },
        {
            0 /*arithm::addScalar<float, float, unsigned char>*/,
            0 /*arithm::addScalar<float, float, signed char>*/,
            0 /*arithm::addScalar<float, float, unsigned short>*/,
            0 /*arithm::addScalar<float, float, short>*/,
            0 /*arithm::addScalar<float, float, int>*/,
            arithm::addScalar<float, float, float>,
            arithm::addScalar<float, double, double>
        },
        {
            0 /*arithm::addScalar<double, double, unsigned char>*/,
            0 /*arithm::addScalar<double, double, signed char>*/,
            0 /*arithm::addScalar<double, double, unsigned short>*/,
            0 /*arithm::addScalar<double, double, short>*/,
            0 /*arithm::addScalar<double, double, int>*/,
            0 /*arithm::addScalar<double, double, float>*/,
            arithm::addScalar<double, double, double>
        }
    };

    typedef void (*npp_func_t)(const PtrStepSzb src, Scalar sc, PtrStepb dst, cudaStream_t stream);
    static const npp_func_t npp_funcs[7][4] =
    {
        {NppArithmScalar<CV_8U , 1, nppiAddC_8u_C1RSfs >::call, 0                                                     , NppArithmScalar<CV_8U , 3, nppiAddC_8u_C3RSfs >::call, NppArithmScalar<CV_8U , 4, nppiAddC_8u_C4RSfs >::call},
        {0                                                    , 0                                                     , 0                                                    , 0                                                    },
        {NppArithmScalar<CV_16U, 1, nppiAddC_16u_C1RSfs>::call, 0                                                     , NppArithmScalar<CV_16U, 3, nppiAddC_16u_C3RSfs>::call, NppArithmScalar<CV_16U, 4, nppiAddC_16u_C4RSfs>::call},
        {NppArithmScalar<CV_16S, 1, nppiAddC_16s_C1RSfs>::call, NppArithmScalar<CV_16S, 2, nppiAddC_16sc_C1RSfs>::call, NppArithmScalar<CV_16S, 3, nppiAddC_16s_C3RSfs>::call, NppArithmScalar<CV_16S, 4, nppiAddC_16s_C4RSfs>::call},
        {NppArithmScalar<CV_32S, 1, nppiAddC_32s_C1RSfs>::call, NppArithmScalar<CV_32S, 2, nppiAddC_32sc_C1RSfs>::call, NppArithmScalar<CV_32S, 3, nppiAddC_32s_C3RSfs>::call, 0                                                    },
        {NppArithmScalar<CV_32F, 1, nppiAddC_32f_C1R   >::call, NppArithmScalar<CV_32F, 2, nppiAddC_32fc_C1R   >::call, NppArithmScalar<CV_32F, 3, nppiAddC_32f_C3R   >::call, NppArithmScalar<CV_32F, 4, nppiAddC_32f_C4R   >::call},
        {0                                                    , 0                                                     , 0                                                    , 0                                                    }
    };

    const int sdepth = src.depth();
    const int ddepth = dst.depth();
    const int cn = src.channels();

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    const npp_func_t npp_func = npp_funcs[sdepth][cn - 1];
    if (ddepth == sdepth && cn > 1 && npp_func != 0)
    {
        npp_func(src, val, dst, stream);
        return;
    }

    CV_Assert( cn == 1 );

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, val[0], dst, mask, stream);
}

void cv::cuda::add(InputArray src1, InputArray src2, OutputArray dst, InputArray mask, int dtype, Stream& stream)
{
    arithm_op(src1, src2, dst, mask, 1.0, dtype, stream, addMat, addScalar);
}

////////////////////////////////////////////////////////////////////////
// subtract

namespace arithm
{
    void subMat_v4(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);
    void subMat_v2(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);

    template <typename T, typename D>
    void subMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

static void subMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, double, Stream& _stream, int)
{
    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            arithm::subMat<unsigned char, unsigned char>,
            arithm::subMat<unsigned char, signed char>,
            arithm::subMat<unsigned char, unsigned short>,
            arithm::subMat<unsigned char, short>,
            arithm::subMat<unsigned char, int>,
            arithm::subMat<unsigned char, float>,
            arithm::subMat<unsigned char, double>
        },
        {
            arithm::subMat<signed char, unsigned char>,
            arithm::subMat<signed char, signed char>,
            arithm::subMat<signed char, unsigned short>,
            arithm::subMat<signed char, short>,
            arithm::subMat<signed char, int>,
            arithm::subMat<signed char, float>,
            arithm::subMat<signed char, double>
        },
        {
            0 /*arithm::subMat<unsigned short, unsigned char>*/,
            0 /*arithm::subMat<unsigned short, signed char>*/,
            arithm::subMat<unsigned short, unsigned short>,
            arithm::subMat<unsigned short, short>,
            arithm::subMat<unsigned short, int>,
            arithm::subMat<unsigned short, float>,
            arithm::subMat<unsigned short, double>
        },
        {
            0 /*arithm::subMat<short, unsigned char>*/,
            0 /*arithm::subMat<short, signed char>*/,
            arithm::subMat<short, unsigned short>,
            arithm::subMat<short, short>,
            arithm::subMat<short, int>,
            arithm::subMat<short, float>,
            arithm::subMat<short, double>
        },
        {
            0 /*arithm::subMat<int, unsigned char>*/,
            0 /*arithm::subMat<int, signed char>*/,
            0 /*arithm::subMat<int, unsigned short>*/,
            0 /*arithm::subMat<int, short>*/,
            arithm::subMat<int, int>,
            arithm::subMat<int, float>,
            arithm::subMat<int, double>
        },
        {
            0 /*arithm::subMat<float, unsigned char>*/,
            0 /*arithm::subMat<float, signed char>*/,
            0 /*arithm::subMat<float, unsigned short>*/,
            0 /*arithm::subMat<float, short>*/,
            0 /*arithm::subMat<float, int>*/,
            arithm::subMat<float, float>,
            arithm::subMat<float, double>
        },
        {
            0 /*arithm::subMat<double, unsigned char>*/,
            0 /*arithm::subMat<double, signed char>*/,
            0 /*arithm::subMat<double, unsigned short>*/,
            0 /*arithm::subMat<double, short>*/,
            0 /*arithm::subMat<double, int>*/,
            0 /*arithm::subMat<double, float>*/,
            arithm::subMat<double, double>
        }
    };

    const int sdepth = src1.depth();
    const int ddepth = dst.depth();
    const int cn = src1.channels();

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    PtrStepSzb src1_(src1.rows, src1.cols * cn, src1.data, src1.step);
    PtrStepSzb src2_(src1.rows, src1.cols * cn, src2.data, src2.step);
    PtrStepSzb dst_(src1.rows, src1.cols * cn, dst.data, dst.step);

    if (mask.empty() && (sdepth == CV_8U || sdepth == CV_16U) && ddepth == sdepth)
    {
        const intptr_t src1ptr = reinterpret_cast<intptr_t>(src1_.data);
        const intptr_t src2ptr = reinterpret_cast<intptr_t>(src2_.data);
        const intptr_t dstptr = reinterpret_cast<intptr_t>(dst_.data);

        const bool isAllAligned = (src1ptr & 31) == 0 && (src2ptr & 31) == 0 && (dstptr & 31) == 0;

        if (isAllAligned)
        {
            if (sdepth == CV_8U && (src1_.cols & 3) == 0)
            {
                const int vcols = src1_.cols >> 2;

                arithm::subMat_v4(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                                  PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                                  PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                                  stream);

                return;
            }
            else if (sdepth == CV_16U && (src1_.cols & 1) == 0)
            {
                const int vcols = src1_.cols >> 1;

                arithm::subMat_v2(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                                  PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                                  PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                                  stream);

                return;
            }
        }
    }

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, mask, stream);
}

namespace arithm
{
    template <typename T, typename S, typename D>
    void subScalar(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

static void subScalar(const GpuMat& src, Scalar val, bool inv, GpuMat& dst, const GpuMat& mask, double, Stream& _stream, int)
{
    typedef void (*func_t)(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            arithm::subScalar<unsigned char, float, unsigned char>,
            arithm::subScalar<unsigned char, float, signed char>,
            arithm::subScalar<unsigned char, float, unsigned short>,
            arithm::subScalar<unsigned char, float, short>,
            arithm::subScalar<unsigned char, float, int>,
            arithm::subScalar<unsigned char, float, float>,
            arithm::subScalar<unsigned char, double, double>
        },
        {
            arithm::subScalar<signed char, float, unsigned char>,
            arithm::subScalar<signed char, float, signed char>,
            arithm::subScalar<signed char, float, unsigned short>,
            arithm::subScalar<signed char, float, short>,
            arithm::subScalar<signed char, float, int>,
            arithm::subScalar<signed char, float, float>,
            arithm::subScalar<signed char, double, double>
        },
        {
            0 /*arithm::subScalar<unsigned short, float, unsigned char>*/,
            0 /*arithm::subScalar<unsigned short, float, signed char>*/,
            arithm::subScalar<unsigned short, float, unsigned short>,
            arithm::subScalar<unsigned short, float, short>,
            arithm::subScalar<unsigned short, float, int>,
            arithm::subScalar<unsigned short, float, float>,
            arithm::subScalar<unsigned short, double, double>
        },
        {
            0 /*arithm::subScalar<short, float, unsigned char>*/,
            0 /*arithm::subScalar<short, float, signed char>*/,
            arithm::subScalar<short, float, unsigned short>,
            arithm::subScalar<short, float, short>,
            arithm::subScalar<short, float, int>,
            arithm::subScalar<short, float, float>,
            arithm::subScalar<short, double, double>
        },
        {
            0 /*arithm::subScalar<int, float, unsigned char>*/,
            0 /*arithm::subScalar<int, float, signed char>*/,
            0 /*arithm::subScalar<int, float, unsigned short>*/,
            0 /*arithm::subScalar<int, float, short>*/,
            arithm::subScalar<int, float, int>,
            arithm::subScalar<int, float, float>,
            arithm::subScalar<int, double, double>
        },
        {
            0 /*arithm::subScalar<float, float, unsigned char>*/,
            0 /*arithm::subScalar<float, float, signed char>*/,
            0 /*arithm::subScalar<float, float, unsigned short>*/,
            0 /*arithm::subScalar<float, float, short>*/,
            0 /*arithm::subScalar<float, float, int>*/,
            arithm::subScalar<float, float, float>,
            arithm::subScalar<float, double, double>
        },
        {
            0 /*arithm::subScalar<double, double, unsigned char>*/,
            0 /*arithm::subScalar<double, double, signed char>*/,
            0 /*arithm::subScalar<double, double, unsigned short>*/,
            0 /*arithm::subScalar<double, double, short>*/,
            0 /*arithm::subScalar<double, double, int>*/,
            0 /*arithm::subScalar<double, double, float>*/,
            arithm::subScalar<double, double, double>
        }
    };

    typedef void (*npp_func_t)(const PtrStepSzb src, Scalar sc, PtrStepb dst, cudaStream_t stream);
    static const npp_func_t npp_funcs[7][4] =
    {
        {NppArithmScalar<CV_8U , 1, nppiSubC_8u_C1RSfs >::call, 0                                                     , NppArithmScalar<CV_8U , 3, nppiSubC_8u_C3RSfs >::call, NppArithmScalar<CV_8U , 4, nppiSubC_8u_C4RSfs >::call},
        {0                                                    , 0                                                     , 0                                                    , 0                                                    },
        {NppArithmScalar<CV_16U, 1, nppiSubC_16u_C1RSfs>::call, 0                                                     , NppArithmScalar<CV_16U, 3, nppiSubC_16u_C3RSfs>::call, NppArithmScalar<CV_16U, 4, nppiSubC_16u_C4RSfs>::call},
        {NppArithmScalar<CV_16S, 1, nppiSubC_16s_C1RSfs>::call, NppArithmScalar<CV_16S, 2, nppiSubC_16sc_C1RSfs>::call, NppArithmScalar<CV_16S, 3, nppiSubC_16s_C3RSfs>::call, NppArithmScalar<CV_16S, 4, nppiSubC_16s_C4RSfs>::call},
        {NppArithmScalar<CV_32S, 1, nppiSubC_32s_C1RSfs>::call, NppArithmScalar<CV_32S, 2, nppiSubC_32sc_C1RSfs>::call, NppArithmScalar<CV_32S, 3, nppiSubC_32s_C3RSfs>::call, 0                                                    },
        {NppArithmScalar<CV_32F, 1, nppiSubC_32f_C1R   >::call, NppArithmScalar<CV_32F, 2, nppiSubC_32fc_C1R   >::call, NppArithmScalar<CV_32F, 3, nppiSubC_32f_C3R   >::call, NppArithmScalar<CV_32F, 4, nppiSubC_32f_C4R   >::call},
        {0                                                    , 0                                                     , 0                                                    , 0                                                    }
    };

    const int sdepth = src.depth();
    const int ddepth = dst.depth();
    const int cn = src.channels();

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    const npp_func_t npp_func = npp_funcs[sdepth][cn - 1];
    if (ddepth == sdepth && cn > 1 && npp_func != 0 && !inv)
    {
        npp_func(src, val, dst, stream);
        return;
    }

    CV_Assert( cn == 1 );

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, val[0], inv, dst, mask, stream);
}

void cv::cuda::subtract(InputArray src1, InputArray src2, OutputArray dst, InputArray mask, int dtype, Stream& stream)
{
    arithm_op(src1, src2, dst, mask, 1.0, dtype, stream, subMat, subScalar);
}

////////////////////////////////////////////////////////////////////////
// multiply

namespace arithm
{
    void mulMat_8uc4_32f(PtrStepSz<unsigned int> src1, PtrStepSzf src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);

    void mulMat_16sc4_32f(PtrStepSz<short4> src1, PtrStepSzf src2, PtrStepSz<short4> dst, cudaStream_t stream);

    template <typename T, typename S, typename D>
    void mulMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
}

static void mulMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double scale, Stream& _stream, int)
{
    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            arithm::mulMat<unsigned char, float, unsigned char>,
            arithm::mulMat<unsigned char, float, signed char>,
            arithm::mulMat<unsigned char, float, unsigned short>,
            arithm::mulMat<unsigned char, float, short>,
            arithm::mulMat<unsigned char, float, int>,
            arithm::mulMat<unsigned char, float, float>,
            arithm::mulMat<unsigned char, double, double>
        },
        {
            arithm::mulMat<signed char, float, unsigned char>,
            arithm::mulMat<signed char, float, signed char>,
            arithm::mulMat<signed char, float, unsigned short>,
            arithm::mulMat<signed char, float, short>,
            arithm::mulMat<signed char, float, int>,
            arithm::mulMat<signed char, float, float>,
            arithm::mulMat<signed char, double, double>
        },
        {
            0 /*arithm::mulMat<unsigned short, float, unsigned char>*/,
            0 /*arithm::mulMat<unsigned short, float, signed char>*/,
            arithm::mulMat<unsigned short, float, unsigned short>,
            arithm::mulMat<unsigned short, float, short>,
            arithm::mulMat<unsigned short, float, int>,
            arithm::mulMat<unsigned short, float, float>,
            arithm::mulMat<unsigned short, double, double>
        },
        {
            0 /*arithm::mulMat<short, float, unsigned char>*/,
            0 /*arithm::mulMat<short, float, signed char>*/,
            arithm::mulMat<short, float, unsigned short>,
            arithm::mulMat<short, float, short>,
            arithm::mulMat<short, float, int>,
            arithm::mulMat<short, float, float>,
            arithm::mulMat<short, double, double>
        },
        {
            0 /*arithm::mulMat<int, float, unsigned char>*/,
            0 /*arithm::mulMat<int, float, signed char>*/,
            0 /*arithm::mulMat<int, float, unsigned short>*/,
            0 /*arithm::mulMat<int, float, short>*/,
            arithm::mulMat<int, float, int>,
            arithm::mulMat<int, float, float>,
            arithm::mulMat<int, double, double>
        },
        {
            0 /*arithm::mulMat<float, float, unsigned char>*/,
            0 /*arithm::mulMat<float, float, signed char>*/,
            0 /*arithm::mulMat<float, float, unsigned short>*/,
            0 /*arithm::mulMat<float, float, short>*/,
            0 /*arithm::mulMat<float, float, int>*/,
            arithm::mulMat<float, float, float>,
            arithm::mulMat<float, double, double>
        },
        {
            0 /*arithm::mulMat<double, double, unsigned char>*/,
            0 /*arithm::mulMat<double, double, signed char>*/,
            0 /*arithm::mulMat<double, double, unsigned short>*/,
            0 /*arithm::mulMat<double, double, short>*/,
            0 /*arithm::mulMat<double, double, int>*/,
            0 /*arithm::mulMat<double, double, float>*/,
            arithm::mulMat<double, double, double>
        }
    };

    const int sdepth = src1.depth();
    const int ddepth = dst.depth();
    const int cn = src1.channels();

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    PtrStepSzb src1_(src1.rows, src1.cols * cn, src1.data, src1.step);
    PtrStepSzb src2_(src1.rows, src1.cols * cn, src2.data, src2.step);
    PtrStepSzb dst_(src1.rows, src1.cols * cn, dst.data, dst.step);

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, scale, stream);
}

namespace arithm
{
    template <typename T, typename S, typename D>
    void mulScalar(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
}

static void mulScalar(const GpuMat& src, Scalar val, bool, GpuMat& dst, const GpuMat&, double scale, Stream& _stream, int)
{
    typedef void (*func_t)(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            arithm::mulScalar<unsigned char, float, unsigned char>,
            arithm::mulScalar<unsigned char, float, signed char>,
            arithm::mulScalar<unsigned char, float, unsigned short>,
            arithm::mulScalar<unsigned char, float, short>,
            arithm::mulScalar<unsigned char, float, int>,
            arithm::mulScalar<unsigned char, float, float>,
            arithm::mulScalar<unsigned char, double, double>
        },
        {
            arithm::mulScalar<signed char, float, unsigned char>,
            arithm::mulScalar<signed char, float, signed char>,
            arithm::mulScalar<signed char, float, unsigned short>,
            arithm::mulScalar<signed char, float, short>,
            arithm::mulScalar<signed char, float, int>,
            arithm::mulScalar<signed char, float, float>,
            arithm::mulScalar<signed char, double, double>
        },
        {
            0 /*arithm::mulScalar<unsigned short, float, unsigned char>*/,
            0 /*arithm::mulScalar<unsigned short, float, signed char>*/,
            arithm::mulScalar<unsigned short, float, unsigned short>,
            arithm::mulScalar<unsigned short, float, short>,
            arithm::mulScalar<unsigned short, float, int>,
            arithm::mulScalar<unsigned short, float, float>,
            arithm::mulScalar<unsigned short, double, double>
        },
        {
            0 /*arithm::mulScalar<short, float, unsigned char>*/,
            0 /*arithm::mulScalar<short, float, signed char>*/,
            arithm::mulScalar<short, float, unsigned short>,
            arithm::mulScalar<short, float, short>,
            arithm::mulScalar<short, float, int>,
            arithm::mulScalar<short, float, float>,
            arithm::mulScalar<short, double, double>
        },
        {
            0 /*arithm::mulScalar<int, float, unsigned char>*/,
            0 /*arithm::mulScalar<int, float, signed char>*/,
            0 /*arithm::mulScalar<int, float, unsigned short>*/,
            0 /*arithm::mulScalar<int, float, short>*/,
            arithm::mulScalar<int, float, int>,
            arithm::mulScalar<int, float, float>,
            arithm::mulScalar<int, double, double>
        },
        {
            0 /*arithm::mulScalar<float, float, unsigned char>*/,
            0 /*arithm::mulScalar<float, float, signed char>*/,
            0 /*arithm::mulScalar<float, float, unsigned short>*/,
            0 /*arithm::mulScalar<float, float, short>*/,
            0 /*arithm::mulScalar<float, float, int>*/,
            arithm::mulScalar<float, float, float>,
            arithm::mulScalar<float, double, double>
        },
        {
            0 /*arithm::mulScalar<double, double, unsigned char>*/,
            0 /*arithm::mulScalar<double, double, signed char>*/,
            0 /*arithm::mulScalar<double, double, unsigned short>*/,
            0 /*arithm::mulScalar<double, double, short>*/,
            0 /*arithm::mulScalar<double, double, int>*/,
            0 /*arithm::mulScalar<double, double, float>*/,
            arithm::mulScalar<double, double, double>
        }
    };

    typedef void (*npp_func_t)(const PtrStepSzb src, Scalar sc, PtrStepb dst, cudaStream_t stream);
    static const npp_func_t npp_funcs[7][4] =
    {
        {NppArithmScalar<CV_8U , 1, nppiMulC_8u_C1RSfs >::call, 0, NppArithmScalar<CV_8U , 3, nppiMulC_8u_C3RSfs >::call, NppArithmScalar<CV_8U , 4, nppiMulC_8u_C4RSfs >::call},
        {0                                                    , 0, 0                                                    , 0                                                    },
        {NppArithmScalar<CV_16U, 1, nppiMulC_16u_C1RSfs>::call, 0, NppArithmScalar<CV_16U, 3, nppiMulC_16u_C3RSfs>::call, NppArithmScalar<CV_16U, 4, nppiMulC_16u_C4RSfs>::call},
        {NppArithmScalar<CV_16S, 1, nppiMulC_16s_C1RSfs>::call, 0, NppArithmScalar<CV_16S, 3, nppiMulC_16s_C3RSfs>::call, NppArithmScalar<CV_16S, 4, nppiMulC_16s_C4RSfs>::call},
        {NppArithmScalar<CV_32S, 1, nppiMulC_32s_C1RSfs>::call, 0, NppArithmScalar<CV_32S, 3, nppiMulC_32s_C3RSfs>::call, 0                                                    },
        {NppArithmScalar<CV_32F, 1, nppiMulC_32f_C1R   >::call, 0, NppArithmScalar<CV_32F, 3, nppiMulC_32f_C3R   >::call, NppArithmScalar<CV_32F, 4, nppiMulC_32f_C4R   >::call},
        {0                                                    , 0, 0                                                    , 0                                                    }
    };

    const int sdepth = src.depth();
    const int ddepth = dst.depth();
    const int cn = src.channels();

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    val[0] *= scale;
    val[1] *= scale;
    val[2] *= scale;
    val[3] *= scale;

    const npp_func_t npp_func = npp_funcs[sdepth][cn - 1];
    if (ddepth == sdepth && cn > 1 && npp_func != 0)
    {
        npp_func(src, val, dst, stream);
        return;
    }

    CV_Assert( cn == 1 );

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, val[0], dst, stream);
}

void cv::cuda::multiply(InputArray _src1, InputArray _src2, OutputArray _dst, double scale, int dtype, Stream& stream)
{
    if (_src1.type() == CV_8UC4 && _src2.type() == CV_32FC1)
    {
        GpuMat src1 = _src1.getGpuMat();
        GpuMat src2 = _src2.getGpuMat();

        CV_Assert( src1.size() == src2.size() );

        _dst.create(src1.size(), src1.type());
        GpuMat dst = _dst.getGpuMat();

        arithm::mulMat_8uc4_32f(src1, src2, dst, StreamAccessor::getStream(stream));
    }
    else if (_src1.type() == CV_16SC4 && _src2.type() == CV_32FC1)
    {
        GpuMat src1 = _src1.getGpuMat();
        GpuMat src2 = _src2.getGpuMat();

        CV_Assert( src1.size() == src2.size() );

        _dst.create(src1.size(), src1.type());
        GpuMat dst = _dst.getGpuMat();

        arithm::mulMat_16sc4_32f(src1, src2, dst, StreamAccessor::getStream(stream));
    }
    else
    {
        arithm_op(_src1, _src2, _dst, GpuMat(), scale, dtype, stream, mulMat, mulScalar);
    }
}

////////////////////////////////////////////////////////////////////////
// divide

namespace arithm
{
    void divMat_8uc4_32f(PtrStepSz<unsigned int> src1, PtrStepSzf src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);

    void divMat_16sc4_32f(PtrStepSz<short4> src1, PtrStepSzf src2, PtrStepSz<short4> dst, cudaStream_t stream);

    template <typename T, typename S, typename D>
    void divMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
}

static void divMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double scale, Stream& _stream, int)
{
    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            arithm::divMat<unsigned char, float, unsigned char>,
            arithm::divMat<unsigned char, float, signed char>,
            arithm::divMat<unsigned char, float, unsigned short>,
            arithm::divMat<unsigned char, float, short>,
            arithm::divMat<unsigned char, float, int>,
            arithm::divMat<unsigned char, float, float>,
            arithm::divMat<unsigned char, double, double>
        },
        {
            arithm::divMat<signed char, float, unsigned char>,
            arithm::divMat<signed char, float, signed char>,
            arithm::divMat<signed char, float, unsigned short>,
            arithm::divMat<signed char, float, short>,
            arithm::divMat<signed char, float, int>,
            arithm::divMat<signed char, float, float>,
            arithm::divMat<signed char, double, double>
        },
        {
            0 /*arithm::divMat<unsigned short, float, unsigned char>*/,
            0 /*arithm::divMat<unsigned short, float, signed char>*/,
            arithm::divMat<unsigned short, float, unsigned short>,
            arithm::divMat<unsigned short, float, short>,
            arithm::divMat<unsigned short, float, int>,
            arithm::divMat<unsigned short, float, float>,
            arithm::divMat<unsigned short, double, double>
        },
        {
            0 /*arithm::divMat<short, float, unsigned char>*/,
            0 /*arithm::divMat<short, float, signed char>*/,
            arithm::divMat<short, float, unsigned short>,
            arithm::divMat<short, float, short>,
            arithm::divMat<short, float, int>,
            arithm::divMat<short, float, float>,
            arithm::divMat<short, double, double>
        },
        {
            0 /*arithm::divMat<int, float, unsigned char>*/,
            0 /*arithm::divMat<int, float, signed char>*/,
            0 /*arithm::divMat<int, float, unsigned short>*/,
            0 /*arithm::divMat<int, float, short>*/,
            arithm::divMat<int, float, int>,
            arithm::divMat<int, float, float>,
            arithm::divMat<int, double, double>
        },
        {
            0 /*arithm::divMat<float, float, unsigned char>*/,
            0 /*arithm::divMat<float, float, signed char>*/,
            0 /*arithm::divMat<float, float, unsigned short>*/,
            0 /*arithm::divMat<float, float, short>*/,
            0 /*arithm::divMat<float, float, int>*/,
            arithm::divMat<float, float, float>,
            arithm::divMat<float, double, double>
        },
        {
            0 /*arithm::divMat<double, double, unsigned char>*/,
            0 /*arithm::divMat<double, double, signed char>*/,
            0 /*arithm::divMat<double, double, unsigned short>*/,
            0 /*arithm::divMat<double, double, short>*/,
            0 /*arithm::divMat<double, double, int>*/,
            0 /*arithm::divMat<double, double, float>*/,
            arithm::divMat<double, double, double>
        }
    };

    const int sdepth = src1.depth();
    const int ddepth = dst.depth();
    const int cn = src1.channels();

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    PtrStepSzb src1_(src1.rows, src1.cols * cn, src1.data, src1.step);
    PtrStepSzb src2_(src1.rows, src1.cols * cn, src2.data, src2.step);
    PtrStepSzb dst_(src1.rows, src1.cols * cn, dst.data, dst.step);

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, scale, stream);
}

namespace arithm
{
    template <typename T, typename S, typename D>
    void divScalar(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
}

static void divScalar(const GpuMat& src, Scalar val, bool inv, GpuMat& dst, const GpuMat&, double scale, Stream& _stream, int)
{
    typedef void (*func_t)(PtrStepSzb src1, double val, bool inv, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            arithm::divScalar<unsigned char, float, unsigned char>,
            arithm::divScalar<unsigned char, float, signed char>,
            arithm::divScalar<unsigned char, float, unsigned short>,
            arithm::divScalar<unsigned char, float, short>,
            arithm::divScalar<unsigned char, float, int>,
            arithm::divScalar<unsigned char, float, float>,
            arithm::divScalar<unsigned char, double, double>
        },
        {
            arithm::divScalar<signed char, float, unsigned char>,
            arithm::divScalar<signed char, float, signed char>,
            arithm::divScalar<signed char, float, unsigned short>,
            arithm::divScalar<signed char, float, short>,
            arithm::divScalar<signed char, float, int>,
            arithm::divScalar<signed char, float, float>,
            arithm::divScalar<signed char, double, double>
        },
        {
            0 /*arithm::divScalar<unsigned short, float, unsigned char>*/,
            0 /*arithm::divScalar<unsigned short, float, signed char>*/,
            arithm::divScalar<unsigned short, float, unsigned short>,
            arithm::divScalar<unsigned short, float, short>,
            arithm::divScalar<unsigned short, float, int>,
            arithm::divScalar<unsigned short, float, float>,
            arithm::divScalar<unsigned short, double, double>
        },
        {
            0 /*arithm::divScalar<short, float, unsigned char>*/,
            0 /*arithm::divScalar<short, float, signed char>*/,
            arithm::divScalar<short, float, unsigned short>,
            arithm::divScalar<short, float, short>,
            arithm::divScalar<short, float, int>,
            arithm::divScalar<short, float, float>,
            arithm::divScalar<short, double, double>
        },
        {
            0 /*arithm::divScalar<int, float, unsigned char>*/,
            0 /*arithm::divScalar<int, float, signed char>*/,
            0 /*arithm::divScalar<int, float, unsigned short>*/,
            0 /*arithm::divScalar<int, float, short>*/,
            arithm::divScalar<int, float, int>,
            arithm::divScalar<int, float, float>,
            arithm::divScalar<int, double, double>
        },
        {
            0 /*arithm::divScalar<float, float, unsigned char>*/,
            0 /*arithm::divScalar<float, float, signed char>*/,
            0 /*arithm::divScalar<float, float, unsigned short>*/,
            0 /*arithm::divScalar<float, float, short>*/,
            0 /*arithm::divScalar<float, float, int>*/,
            arithm::divScalar<float, float, float>,
            arithm::divScalar<float, double, double>
        },
        {
            0 /*arithm::divScalar<double, double, unsigned char>*/,
            0 /*arithm::divScalar<double, double, signed char>*/,
            0 /*arithm::divScalar<double, double, unsigned short>*/,
            0 /*arithm::divScalar<double, double, short>*/,
            0 /*arithm::divScalar<double, double, int>*/,
            0 /*arithm::divScalar<double, double, float>*/,
            arithm::divScalar<double, double, double>
        }
    };

    typedef void (*npp_func_t)(const PtrStepSzb src, Scalar sc, PtrStepb dst, cudaStream_t stream);
    static const npp_func_t npp_funcs[7][4] =
    {
        {NppArithmScalar<CV_8U , 1, nppiDivC_8u_C1RSfs >::call, 0, NppArithmScalar<CV_8U , 3, nppiDivC_8u_C3RSfs >::call, NppArithmScalar<CV_8U , 4, nppiDivC_8u_C4RSfs >::call},
        {0                                                    , 0, 0                                                    , 0                                                    },
        {NppArithmScalar<CV_16U, 1, nppiDivC_16u_C1RSfs>::call, 0, NppArithmScalar<CV_16U, 3, nppiDivC_16u_C3RSfs>::call, NppArithmScalar<CV_16U, 4, nppiDivC_16u_C4RSfs>::call},
        {NppArithmScalar<CV_16S, 1, nppiDivC_16s_C1RSfs>::call, 0, NppArithmScalar<CV_16S, 3, nppiDivC_16s_C3RSfs>::call, NppArithmScalar<CV_16S, 4, nppiDivC_16s_C4RSfs>::call},
        {NppArithmScalar<CV_32S, 1, nppiDivC_32s_C1RSfs>::call, 0, NppArithmScalar<CV_32S, 3, nppiDivC_32s_C3RSfs>::call, 0                                                    },
        {NppArithmScalar<CV_32F, 1, nppiDivC_32f_C1R   >::call, 0, NppArithmScalar<CV_32F, 3, nppiDivC_32f_C3R   >::call, NppArithmScalar<CV_32F, 4, nppiDivC_32f_C4R   >::call},
        {0                                                    , 0, 0                                                    , 0                                                    }
    };

    const int sdepth = src.depth();
    const int ddepth = dst.depth();
    const int cn = src.channels();

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    if (inv)
    {
        val[0] *= scale;
        val[1] *= scale;
        val[2] *= scale;
        val[3] *= scale;
    }
    else
    {
        val[0] /= scale;
        val[1] /= scale;
        val[2] /= scale;
        val[3] /= scale;
    }

    const npp_func_t npp_func = npp_funcs[sdepth][cn - 1];
    if (ddepth == sdepth && cn > 1 && npp_func != 0 && !inv)
    {
        npp_func(src, val, dst, stream);
        return;
    }

    CV_Assert( cn == 1 );

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, val[0], inv, dst, stream);
}

void cv::cuda::divide(InputArray _src1, InputArray _src2, OutputArray _dst, double scale, int dtype, Stream& stream)
{
    if (_src1.type() == CV_8UC4 && _src2.type() == CV_32FC1)
    {
        GpuMat src1 = _src1.getGpuMat();
        GpuMat src2 = _src2.getGpuMat();

        CV_Assert( src1.size() == src2.size() );

        _dst.create(src1.size(), src1.type());
        GpuMat dst = _dst.getGpuMat();

        arithm::divMat_8uc4_32f(src1, src2, dst, StreamAccessor::getStream(stream));
    }
    else if (_src1.type() == CV_16SC4 && _src2.type() == CV_32FC1)
    {
        GpuMat src1 = _src1.getGpuMat();
        GpuMat src2 = _src2.getGpuMat();

        CV_Assert( src1.size() == src2.size() );

        _dst.create(src1.size(), src1.type());
        GpuMat dst = _dst.getGpuMat();

        arithm::divMat_16sc4_32f(src1, src2, dst, StreamAccessor::getStream(stream));
    }
    else
    {
        arithm_op(_src1, _src2, _dst, GpuMat(), scale, dtype, stream, divMat, divScalar);
    }
}

//////////////////////////////////////////////////////////////////////////////
// absdiff

namespace arithm
{
    void absDiffMat_v4(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);
    void absDiffMat_v2(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);

    template <typename T>
    void absDiffMat(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
}

static void absDiffMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double, Stream& _stream, int)
{
    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        arithm::absDiffMat<unsigned char>,
        arithm::absDiffMat<signed char>,
        arithm::absDiffMat<unsigned short>,
        arithm::absDiffMat<short>,
        arithm::absDiffMat<int>,
        arithm::absDiffMat<float>,
        arithm::absDiffMat<double>
    };

    const int depth = src1.depth();
    const int cn = src1.channels();

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

                arithm::absDiffMat_v4(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                                      PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                                      PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                                      stream);

                return;
            }
            else if (depth == CV_16U && (src1_.cols & 1) == 0)
            {
                const int vcols = src1_.cols >> 1;

                arithm::absDiffMat_v2(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                                      PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                                      PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                                      stream);

                return;
            }
        }
    }

    const func_t func = funcs[depth];

    if (!func)
        CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, stream);
}

namespace arithm
{
    template <typename T, typename S>
    void absDiffScalar(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
}

static void absDiffScalar(const GpuMat& src, Scalar val, bool, GpuMat& dst, const GpuMat&, double, Stream& stream, int)
{
    typedef void (*func_t)(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        arithm::absDiffScalar<unsigned char, float>,
        arithm::absDiffScalar<signed char, float>,
        arithm::absDiffScalar<unsigned short, float>,
        arithm::absDiffScalar<short, float>,
        arithm::absDiffScalar<int, float>,
        arithm::absDiffScalar<float, float>,
        arithm::absDiffScalar<double, double>
    };

    const int depth = src.depth();

    funcs[depth](src, val[0], dst, StreamAccessor::getStream(stream));
}

void cv::cuda::absdiff(InputArray src1, InputArray src2, OutputArray dst, Stream& stream)
{
    arithm_op(src1, src2, dst, noArray(), 1.0, -1, stream, absDiffMat, absDiffScalar);
}

//////////////////////////////////////////////////////////////////////////////
// abs

namespace arithm
{
    template <typename T>
    void absMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

void cv::cuda::abs(InputArray _src, OutputArray _dst, Stream& stream)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        absMat<unsigned char>,
        absMat<signed char>,
        absMat<unsigned short>,
        absMat<short>,
        absMat<int>,
        absMat<float>,
        absMat<double>
    };

    GpuMat src = _src.getGpuMat();

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    funcs[depth](src, dst, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// sqr

namespace arithm
{
    template <typename T>
    void sqrMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

void cv::cuda::sqr(InputArray _src, OutputArray _dst, Stream& stream)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        sqrMat<unsigned char>,
        sqrMat<signed char>,
        sqrMat<unsigned short>,
        sqrMat<short>,
        sqrMat<int>,
        sqrMat<float>,
        sqrMat<double>
    };

    GpuMat src = _src.getGpuMat();

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    funcs[depth](src, dst, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// sqrt

namespace arithm
{
    template <typename T>
    void sqrtMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

void cv::cuda::sqrt(InputArray _src, OutputArray _dst, Stream& stream)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        sqrtMat<unsigned char>,
        sqrtMat<signed char>,
        sqrtMat<unsigned short>,
        sqrtMat<short>,
        sqrtMat<int>,
        sqrtMat<float>,
        sqrtMat<double>
    };

    GpuMat src = _src.getGpuMat();

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    funcs[depth](src, dst, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// exp

namespace arithm
{
    template <typename T>
    void expMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

void cv::cuda::exp(InputArray _src, OutputArray _dst, Stream& stream)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        expMat<unsigned char>,
        expMat<signed char>,
        expMat<unsigned short>,
        expMat<short>,
        expMat<int>,
        expMat<float>,
        expMat<double>
    };

    GpuMat src = _src.getGpuMat();

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    funcs[depth](src, dst, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// log

namespace arithm
{
    template <typename T>
    void logMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

void cv::cuda::log(InputArray _src, OutputArray _dst, Stream& stream)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        logMat<unsigned char>,
        logMat<signed char>,
        logMat<unsigned short>,
        logMat<short>,
        logMat<int>,
        logMat<float>,
        logMat<double>
    };

    GpuMat src = _src.getGpuMat();

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    funcs[depth](src, dst, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// pow

namespace arithm
{
    template<typename T> void pow(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
}

void cv::cuda::pow(InputArray _src, double power, OutputArray _dst, Stream& stream)
{
    typedef void (*func_t)(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        arithm::pow<unsigned char>,
        arithm::pow<signed char>,
        arithm::pow<unsigned short>,
        arithm::pow<short>,
        arithm::pow<int>,
        arithm::pow<float>,
        arithm::pow<double>
    };

    GpuMat src = _src.getGpuMat();

    const int depth = src.depth();
    const int cn = src.channels();

    CV_Assert(depth <= CV_64F);

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    PtrStepSzb src_(src.rows, src.cols * cn, src.data, src.step);
    PtrStepSzb dst_(src.rows, src.cols * cn, dst.data, dst.step);

    funcs[depth](src_, power, dst_, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// compare

namespace arithm
{
    void cmpMatEq_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream);
    void cmpMatNe_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream);
    void cmpMatLt_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream);
    void cmpMatLe_v4(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream);

    template <typename T> void cmpMatEq(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void cmpMatNe(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void cmpMatLt(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void cmpMatLe(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
}

static void cmpMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat&, double, Stream& _stream, int cmpop)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[7][4] =
    {
        {cmpMatEq<unsigned char> , cmpMatNe<unsigned char> , cmpMatLt<unsigned char> , cmpMatLe<unsigned char> },
        {cmpMatEq<signed char>   , cmpMatNe<signed char>   , cmpMatLt<signed char>   , cmpMatLe<signed char>   },
        {cmpMatEq<unsigned short>, cmpMatNe<unsigned short>, cmpMatLt<unsigned short>, cmpMatLe<unsigned short>},
        {cmpMatEq<short>         , cmpMatNe<short>         , cmpMatLt<short>         , cmpMatLe<short>         },
        {cmpMatEq<int>           , cmpMatNe<int>           , cmpMatLt<int>           , cmpMatLe<int>           },
        {cmpMatEq<float>         , cmpMatNe<float>         , cmpMatLt<float>         , cmpMatLe<float>         },
        {cmpMatEq<double>        , cmpMatNe<double>        , cmpMatLt<double>        , cmpMatLe<double>        }
    };

    typedef void (*func_v4_t)(PtrStepSz<uint> src1, PtrStepSz<uint> src2, PtrStepSz<uint> dst, cudaStream_t stream);
    static const func_v4_t funcs_v4[] =
    {
        cmpMatEq_v4, cmpMatNe_v4, cmpMatLt_v4, cmpMatLe_v4
    };

    const int depth = src1.depth();
    const int cn = src1.channels();

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    static const int codes[] =
    {
        0, 2, 3, 2, 3, 1
    };
    const GpuMat* psrc1[] =
    {
        &src1, &src2, &src2, &src1, &src1, &src1
    };
    const GpuMat* psrc2[] =
    {
        &src2, &src1, &src1, &src2, &src2, &src2
    };

    const int code = codes[cmpop];
    PtrStepSzb src1_(src1.rows, src1.cols * cn, psrc1[cmpop]->data, psrc1[cmpop]->step);
    PtrStepSzb src2_(src1.rows, src1.cols * cn, psrc2[cmpop]->data, psrc2[cmpop]->step);
    PtrStepSzb dst_(src1.rows, src1.cols * cn, dst.data, dst.step);

    if (depth == CV_8U && (src1_.cols & 3) == 0)
    {
        const intptr_t src1ptr = reinterpret_cast<intptr_t>(src1_.data);
        const intptr_t src2ptr = reinterpret_cast<intptr_t>(src2_.data);
        const intptr_t dstptr = reinterpret_cast<intptr_t>(dst_.data);

        const bool isAllAligned = (src1ptr & 31) == 0 && (src2ptr & 31) == 0 && (dstptr & 31) == 0;

        if (isAllAligned)
        {
            const int vcols = src1_.cols >> 2;

            funcs_v4[code](PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                           PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                           PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                           stream);

            return;
        }
    }

    const func_t func = funcs[depth][code];

    func(src1_, src2_, dst_, stream);
}

namespace arithm
{
    template <typename T> void cmpScalarEq(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void cmpScalarNe(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void cmpScalarLt(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void cmpScalarLe(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void cmpScalarGt(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void cmpScalarGe(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
}

namespace
{
    template <typename T> void castScalar(Scalar& sc)
    {
        sc.val[0] = saturate_cast<T>(sc.val[0]);
        sc.val[1] = saturate_cast<T>(sc.val[1]);
        sc.val[2] = saturate_cast<T>(sc.val[2]);
        sc.val[3] = saturate_cast<T>(sc.val[3]);
    }
}

static void cmpScalar(const GpuMat& src, Scalar val, bool inv, GpuMat& dst, const GpuMat&, double, Stream& stream, int cmpop)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src, int cn, double val[4], PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[7][6] =
    {
        {cmpScalarEq<unsigned char> , cmpScalarGt<unsigned char> , cmpScalarGe<unsigned char> , cmpScalarLt<unsigned char> , cmpScalarLe<unsigned char> , cmpScalarNe<unsigned char> },
        {cmpScalarEq<signed char>   , cmpScalarGt<signed char>   , cmpScalarGe<signed char>   , cmpScalarLt<signed char>   , cmpScalarLe<signed char>   , cmpScalarNe<signed char>   },
        {cmpScalarEq<unsigned short>, cmpScalarGt<unsigned short>, cmpScalarGe<unsigned short>, cmpScalarLt<unsigned short>, cmpScalarLe<unsigned short>, cmpScalarNe<unsigned short>},
        {cmpScalarEq<short>         , cmpScalarGt<short>         , cmpScalarGe<short>         , cmpScalarLt<short>         , cmpScalarLe<short>         , cmpScalarNe<short>         },
        {cmpScalarEq<int>           , cmpScalarGt<int>           , cmpScalarGe<int>           , cmpScalarLt<int>           , cmpScalarLe<int>           , cmpScalarNe<int>           },
        {cmpScalarEq<float>         , cmpScalarGt<float>         , cmpScalarGe<float>         , cmpScalarLt<float>         , cmpScalarLe<float>         , cmpScalarNe<float>         },
        {cmpScalarEq<double>        , cmpScalarGt<double>        , cmpScalarGe<double>        , cmpScalarLt<double>        , cmpScalarLe<double>        , cmpScalarNe<double>        }
    };

    typedef void (*cast_func_t)(Scalar& sc);
    static const cast_func_t cast_func[] =
    {
        castScalar<unsigned char>, castScalar<signed char>, castScalar<unsigned short>, castScalar<short>, castScalar<int>, castScalar<float>, castScalar<double>
    };

    if (inv)
    {
        // src1 is a scalar; swap it with src2
        cmpop = cmpop == CMP_LT ? CMP_GT : cmpop == CMP_LE ? CMP_GE :
            cmpop == CMP_GE ? CMP_LE : cmpop == CMP_GT ? CMP_LT : cmpop;
    }

    const int depth = src.depth();
    const int cn = src.channels();

    cast_func[depth](val);

    funcs[depth][cmpop](src, cn, val.val, dst, StreamAccessor::getStream(stream));
}

void cv::cuda::compare(InputArray src1, InputArray src2, OutputArray dst, int cmpop, Stream& stream)
{
    arithm_op(src1, src2, dst, noArray(), 1.0, CV_8U, stream, cmpMat, cmpScalar, cmpop);
}

//////////////////////////////////////////////////////////////////////////////
// bitwise_not

namespace arithm
{
    template <typename T> void bitMatNot(PtrStepSzb src, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

void cv::cuda::bitwise_not(InputArray _src, OutputArray _dst, InputArray _mask, Stream& _stream)
{
    using namespace arithm;

    GpuMat src = _src.getGpuMat();
    GpuMat mask = _mask.getGpuMat();

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    _dst.create(src.size(), src.type());
    GpuMat dst = _dst.getGpuMat();

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    const int bcols = (int) (src.cols * src.elemSize());

    if ((bcols & 3) == 0)
    {
        const int vcols = bcols >> 2;

        bitMatNot<unsigned int>(
                    PtrStepSzb(src.rows, vcols, src.data, src.step),
                    PtrStepSzb(src.rows, vcols, dst.data, dst.step),
                    mask, stream);
    }
    else if ((bcols & 1) == 0)
    {
        const int vcols = bcols >> 1;

        bitMatNot<unsigned short>(
                    PtrStepSzb(src.rows, vcols, src.data, src.step),
                    PtrStepSzb(src.rows, vcols, dst.data, dst.step),
                    mask, stream);
    }
    else
    {
        bitMatNot<unsigned char>(
                    PtrStepSzb(src.rows, bcols, src.data, src.step),
                    PtrStepSzb(src.rows, bcols, dst.data, dst.step),
                    mask, stream);
    }
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

namespace arithm
{
    template <typename T> void bitMatAnd(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template <typename T> void bitMatOr(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template <typename T> void bitMatXor(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

static void bitMat(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, double, Stream& _stream, int op)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    static const func_t funcs32[] =
    {
        bitMatAnd<uint>,
        bitMatOr<uint>,
        bitMatXor<uint>
    };
    static const func_t funcs16[] =
    {
        bitMatAnd<ushort>,
        bitMatOr<ushort>,
        bitMatXor<ushort>
    };
    static const func_t funcs8[] =
    {
        bitMatAnd<uchar>,
        bitMatOr<uchar>,
        bitMatXor<uchar>
    };

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    const int bcols = (int) (src1.cols * src1.elemSize());

    if ((bcols & 3) == 0)
    {
        const int vcols = bcols >> 2;

        funcs32[op](PtrStepSzb(src1.rows, vcols, src1.data, src1.step),
                    PtrStepSzb(src1.rows, vcols, src2.data, src2.step),
                    PtrStepSzb(src1.rows, vcols, dst.data, dst.step),
                    mask, stream);
    }
    else if ((bcols & 1) == 0)
    {
        const int vcols = bcols >> 1;

        funcs16[op](PtrStepSzb(src1.rows, vcols, src1.data, src1.step),
                    PtrStepSzb(src1.rows, vcols, src2.data, src2.step),
                    PtrStepSzb(src1.rows, vcols, dst.data, dst.step),
                    mask, stream);
    }
    else
    {

        funcs8[op](PtrStepSzb(src1.rows, bcols, src1.data, src1.step),
                   PtrStepSzb(src1.rows, bcols, src2.data, src2.step),
                   PtrStepSzb(src1.rows, bcols, dst.data, dst.step),
                   mask, stream);
    }
}

namespace arithm
{
    template <typename T> void bitScalarAnd(PtrStepSzb src1, unsigned int src2, PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void bitScalarOr(PtrStepSzb src1, unsigned int src2, PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void bitScalarXor(PtrStepSzb src1, unsigned int src2, PtrStepSzb dst, cudaStream_t stream);
}

namespace
{
    typedef void (*bit_scalar_func_t)(PtrStepSzb src1, unsigned int src2, PtrStepSzb dst, cudaStream_t stream);

    template <typename T, bit_scalar_func_t func> struct BitScalar
    {
        static void call(const GpuMat& src, Scalar sc, GpuMat& dst, cudaStream_t stream)
        {
            func(src, saturate_cast<T>(sc.val[0]), dst, stream);
        }
    };

    template <bit_scalar_func_t func> struct BitScalar4
    {
        static void call(const GpuMat& src, Scalar sc, GpuMat& dst, cudaStream_t stream)
        {
            unsigned int packedVal = 0;

            packedVal |= (saturate_cast<unsigned char>(sc.val[0]) & 0xffff);
            packedVal |= (saturate_cast<unsigned char>(sc.val[1]) & 0xffff) << 8;
            packedVal |= (saturate_cast<unsigned char>(sc.val[2]) & 0xffff) << 16;
            packedVal |= (saturate_cast<unsigned char>(sc.val[3]) & 0xffff) << 24;

            func(src, packedVal, dst, stream);
        }
    };

    template <int DEPTH, int cn> struct NppBitwiseCFunc
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        typedef NppStatus (*func_t)(const npp_t* pSrc1, int nSrc1Step, const npp_t* pConstants, npp_t* pDst, int nDstStep, NppiSize oSizeROI);
    };
    template <int DEPTH> struct NppBitwiseCFunc<DEPTH, 1>
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        typedef NppStatus (*func_t)(const npp_t* pSrc1, int nSrc1Step, const npp_t pConstant, npp_t* pDst, int nDstStep, NppiSize oSizeROI);
    };

    template <int DEPTH, int cn, typename NppBitwiseCFunc<DEPTH, cn>::func_t func> struct NppBitwiseC
    {
        typedef typename NppBitwiseCFunc<DEPTH, cn>::npp_t npp_t;

        static void call(const GpuMat& src, Scalar sc, GpuMat& dst, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize oSizeROI;
            oSizeROI.width = src.cols;
            oSizeROI.height = src.rows;

            const npp_t pConstants[] = {saturate_cast<npp_t>(sc.val[0]), saturate_cast<npp_t>(sc.val[1]), saturate_cast<npp_t>(sc.val[2]), saturate_cast<npp_t>(sc.val[3])};

            nppSafeCall( func(src.ptr<npp_t>(), static_cast<int>(src.step), pConstants, dst.ptr<npp_t>(), static_cast<int>(dst.step), oSizeROI) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template <int DEPTH, typename NppBitwiseCFunc<DEPTH, 1>::func_t func> struct NppBitwiseC<DEPTH, 1, func>
    {
        typedef typename NppBitwiseCFunc<DEPTH, 1>::npp_t npp_t;

        static void call(const GpuMat& src, Scalar sc, GpuMat& dst, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize oSizeROI;
            oSizeROI.width = src.cols;
            oSizeROI.height = src.rows;

            nppSafeCall( func(src.ptr<npp_t>(), static_cast<int>(src.step), saturate_cast<npp_t>(sc.val[0]), dst.ptr<npp_t>(), static_cast<int>(dst.step), oSizeROI) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

static void bitScalar(const GpuMat& src, Scalar val, bool, GpuMat& dst, const GpuMat& mask, double, Stream& stream, int op)
{
    using namespace arithm;

    typedef void (*func_t)(const GpuMat& src, Scalar sc, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[3][5][4] =
    {
        {
            {BitScalar<unsigned char, bitScalarAnd<unsigned char> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiAndC_8u_C3R >::call, BitScalar4< bitScalarAnd<unsigned int> >::call},
            {0,0,0,0},
            {BitScalar<unsigned short, bitScalarAnd<unsigned short> >::call, 0, NppBitwiseC<CV_16U, 3, nppiAndC_16u_C3R>::call, NppBitwiseC<CV_16U, 4, nppiAndC_16u_C4R>::call},
            {0,0,0,0},
            {BitScalar<int, bitScalarAnd<int> >::call                      , 0, NppBitwiseC<CV_32S, 3, nppiAndC_32s_C3R>::call, NppBitwiseC<CV_32S, 4, nppiAndC_32s_C4R>::call}
        },
        {
            {BitScalar<unsigned char, bitScalarOr<unsigned char> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiOrC_8u_C3R >::call, BitScalar4< bitScalarOr<unsigned int> >::call},
            {0,0,0,0},
            {BitScalar<unsigned short, bitScalarOr<unsigned short> >::call, 0, NppBitwiseC<CV_16U, 3, nppiOrC_16u_C3R>::call, NppBitwiseC<CV_16U, 4, nppiOrC_16u_C4R>::call},
            {0,0,0,0},
            {BitScalar<int, bitScalarOr<int> >::call                      , 0, NppBitwiseC<CV_32S, 3, nppiOrC_32s_C3R>::call, NppBitwiseC<CV_32S, 4, nppiOrC_32s_C4R>::call}
        },
        {
            {BitScalar<unsigned char, bitScalarXor<unsigned char> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiXorC_8u_C3R >::call, BitScalar4< bitScalarXor<unsigned int> >::call},
            {0,0,0,0},
            {BitScalar<unsigned short, bitScalarXor<unsigned short> >::call, 0, NppBitwiseC<CV_16U, 3, nppiXorC_16u_C3R>::call, NppBitwiseC<CV_16U, 4, nppiXorC_16u_C4R>::call},
            {0,0,0,0},
            {BitScalar<int, bitScalarXor<int> >::call                      , 0, NppBitwiseC<CV_32S, 3, nppiXorC_32s_C3R>::call, NppBitwiseC<CV_32S, 4, nppiXorC_32s_C4R>::call}
        }
    };

    const int depth = src.depth();
    const int cn = src.channels();

    CV_Assert( depth == CV_8U || depth == CV_16U || depth == CV_32S );
    CV_Assert( cn == 1 || cn == 3 || cn == 4 );
    CV_Assert( mask.empty() );

    funcs[op][depth][cn - 1](src, val, dst, StreamAccessor::getStream(stream));
}

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
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        typedef NppStatus (*func_t)(const npp_t* pSrc1, int nSrc1Step, const Npp32u* pConstants, npp_t* pDst,  int nDstStep,  NppiSize oSizeROI);
    };
    template <int DEPTH> struct NppShiftFunc<DEPTH, 1>
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        typedef NppStatus (*func_t)(const npp_t* pSrc1, int nSrc1Step, const Npp32u pConstants, npp_t* pDst,  int nDstStep,  NppiSize oSizeROI);
    };

    template <int DEPTH, int cn, typename NppShiftFunc<DEPTH, cn>::func_t func> struct NppShift
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        static void call(const GpuMat& src, Scalar_<Npp32u> sc, GpuMat& dst, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize oSizeROI;
            oSizeROI.width = src.cols;
            oSizeROI.height = src.rows;

            nppSafeCall( func(src.ptr<npp_t>(), static_cast<int>(src.step), sc.val, dst.ptr<npp_t>(), static_cast<int>(dst.step), oSizeROI) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
    template <int DEPTH, typename NppShiftFunc<DEPTH, 1>::func_t func> struct NppShift<DEPTH, 1, func>
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        static void call(const GpuMat& src, Scalar_<Npp32u> sc, GpuMat& dst, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize oSizeROI;
            oSizeROI.width = src.cols;
            oSizeROI.height = src.rows;

            nppSafeCall( func(src.ptr<npp_t>(), static_cast<int>(src.step), sc.val[0], dst.ptr<npp_t>(), static_cast<int>(dst.step), oSizeROI) );

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
