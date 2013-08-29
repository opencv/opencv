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

void cv::gpu::add(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::add(const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::subtract(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::subtract(const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::multiply(const GpuMat&, const GpuMat&, GpuMat&, double, int, Stream&) { throw_nogpu(); }
void cv::gpu::multiply(const GpuMat&, const Scalar&, GpuMat&, double, int, Stream&) { throw_nogpu(); }
void cv::gpu::divide(const GpuMat&, const GpuMat&, GpuMat&, double, int, Stream&) { throw_nogpu(); }
void cv::gpu::divide(const GpuMat&, const Scalar&, GpuMat&, double, int, Stream&) { throw_nogpu(); }
void cv::gpu::divide(double, const GpuMat&, GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::absdiff(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::absdiff(const GpuMat&, const Scalar&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::abs(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::sqr(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::sqrt(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::exp(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::log(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::compare(const GpuMat&, const GpuMat&, GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::compare(const GpuMat&, Scalar, GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_not(const GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_or(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_or(const GpuMat&, const Scalar&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_and(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_and(const GpuMat&, const Scalar&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_xor(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_xor(const GpuMat&, const Scalar&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::rshift(const GpuMat&, Scalar_<int>, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::lshift(const GpuMat&, Scalar_<int>, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::min(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::min(const GpuMat&, double, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::max(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::max(const GpuMat&, double, GpuMat&, Stream&) { throw_nogpu(); }
double cv::gpu::threshold(const GpuMat&, GpuMat&, double, double, int, Stream&) {throw_nogpu(); return 0.0;}
void cv::gpu::pow(const GpuMat&, double, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::alphaComp(const GpuMat&, const GpuMat&, GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::addWeighted(const GpuMat&, double, const GpuMat&, double, double, GpuMat&, int, Stream&) { throw_nogpu(); }

#else

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

void cv::gpu::add(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, int dtype, Stream& s)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            addMat<unsigned char, unsigned char>,
            addMat<unsigned char, signed char>,
            addMat<unsigned char, unsigned short>,
            addMat<unsigned char, short>,
            addMat<unsigned char, int>,
            addMat<unsigned char, float>,
            addMat<unsigned char, double>
        },
        {
            addMat<signed char, unsigned char>,
            addMat<signed char, signed char>,
            addMat<signed char, unsigned short>,
            addMat<signed char, short>,
            addMat<signed char, int>,
            addMat<signed char, float>,
            addMat<signed char, double>
        },
        {
            0 /*addMat<unsigned short, unsigned char>*/,
            0 /*addMat<unsigned short, signed char>*/,
            addMat<unsigned short, unsigned short>,
            addMat<unsigned short, short>,
            addMat<unsigned short, int>,
            addMat<unsigned short, float>,
            addMat<unsigned short, double>
        },
        {
            0 /*addMat<short, unsigned char>*/,
            0 /*addMat<short, signed char>*/,
            addMat<short, unsigned short>,
            addMat<short, short>,
            addMat<short, int>,
            addMat<short, float>,
            addMat<short, double>
        },
        {
            0 /*addMat<int, unsigned char>*/,
            0 /*addMat<int, signed char>*/,
            0 /*addMat<int, unsigned short>*/,
            0 /*addMat<int, short>*/,
            addMat<int, int>,
            addMat<int, float>,
            addMat<int, double>
        },
        {
            0 /*addMat<float, unsigned char>*/,
            0 /*addMat<float, signed char>*/,
            0 /*addMat<float, unsigned short>*/,
            0 /*addMat<float, short>*/,
            0 /*addMat<float, int>*/,
            addMat<float, float>,
            addMat<float, double>
        },
        {
            0 /*addMat<double, unsigned char>*/,
            0 /*addMat<double, signed char>*/,
            0 /*addMat<double, unsigned short>*/,
            0 /*addMat<double, short>*/,
            0 /*addMat<double, int>*/,
            0 /*addMat<double, float>*/,
            addMat<double, double>
        }
    };

    if (dtype < 0)
        dtype = src1.depth();

    const int sdepth = src1.depth();
    const int ddepth = CV_MAT_DEPTH(dtype);
    const int cn = src1.channels();

    CV_Assert( sdepth <= CV_64F && ddepth <= CV_64F );
    CV_Assert( src2.type() == src1.type() && src2.size() == src1.size() );
    CV_Assert( mask.empty() || (cn == 1 && mask.size() == src1.size() && mask.type() == CV_8U) );

    if (sdepth == CV_64F || ddepth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src1.size(), CV_MAKE_TYPE(ddepth, cn));

    cudaStream_t stream = StreamAccessor::getStream(s);

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

                addMat_v4(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                          stream);

                return;
            }
            else if (sdepth == CV_16U && (src1_.cols & 1) == 0)
            {
                const int vcols = src1_.cols >> 1;

                addMat_v2(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                          stream);

                return;
            }
        }
    }

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, mask, stream);
}

namespace arithm
{
    template <typename T, typename S, typename D>
    void addScalar(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

void cv::gpu::add(const GpuMat& src, const Scalar& sc, GpuMat& dst, const GpuMat& mask, int dtype, Stream& s)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            addScalar<unsigned char, float, unsigned char>,
            addScalar<unsigned char, float, signed char>,
            addScalar<unsigned char, float, unsigned short>,
            addScalar<unsigned char, float, short>,
            addScalar<unsigned char, float, int>,
            addScalar<unsigned char, float, float>,
            addScalar<unsigned char, double, double>
        },
        {
            addScalar<signed char, float, unsigned char>,
            addScalar<signed char, float, signed char>,
            addScalar<signed char, float, unsigned short>,
            addScalar<signed char, float, short>,
            addScalar<signed char, float, int>,
            addScalar<signed char, float, float>,
            addScalar<signed char, double, double>
        },
        {
            0 /*addScalar<unsigned short, float, unsigned char>*/,
            0 /*addScalar<unsigned short, float, signed char>*/,
            addScalar<unsigned short, float, unsigned short>,
            addScalar<unsigned short, float, short>,
            addScalar<unsigned short, float, int>,
            addScalar<unsigned short, float, float>,
            addScalar<unsigned short, double, double>
        },
        {
            0 /*addScalar<short, float, unsigned char>*/,
            0 /*addScalar<short, float, signed char>*/,
            addScalar<short, float, unsigned short>,
            addScalar<short, float, short>,
            addScalar<short, float, int>,
            addScalar<short, float, float>,
            addScalar<short, double, double>
        },
        {
            0 /*addScalar<int, float, unsigned char>*/,
            0 /*addScalar<int, float, signed char>*/,
            0 /*addScalar<int, float, unsigned short>*/,
            0 /*addScalar<int, float, short>*/,
            addScalar<int, float, int>,
            addScalar<int, float, float>,
            addScalar<int, double, double>
        },
        {
            0 /*addScalar<float, float, unsigned char>*/,
            0 /*addScalar<float, float, signed char>*/,
            0 /*addScalar<float, float, unsigned short>*/,
            0 /*addScalar<float, float, short>*/,
            0 /*addScalar<float, float, int>*/,
            addScalar<float, float, float>,
            addScalar<float, double, double>
        },
        {
            0 /*addScalar<double, double, unsigned char>*/,
            0 /*addScalar<double, double, signed char>*/,
            0 /*addScalar<double, double, unsigned short>*/,
            0 /*addScalar<double, double, short>*/,
            0 /*addScalar<double, double, int>*/,
            0 /*addScalar<double, double, float>*/,
            addScalar<double, double, double>
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

    if (dtype < 0)
        dtype = src.depth();

    const int sdepth = src.depth();
    const int ddepth = CV_MAT_DEPTH(dtype);
    const int cn = src.channels();

    CV_Assert( sdepth <= CV_64F && ddepth <= CV_64F );
    CV_Assert( cn <= 4 );
    CV_Assert( mask.empty() || (cn == 1 && mask.size() == src.size() && mask.type() == CV_8U) );

    if (sdepth == CV_64F || ddepth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), CV_MAKE_TYPE(ddepth, cn));

    cudaStream_t stream = StreamAccessor::getStream(s);

    const npp_func_t npp_func = npp_funcs[sdepth][cn - 1];
    if (ddepth == sdepth && cn > 1 && npp_func != 0)
    {
        npp_func(src, sc, dst, stream);
        return;
    }

    CV_Assert( cn == 1 );

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, sc.val[0], dst, mask, stream);
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

void cv::gpu::subtract(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, int dtype, Stream& s)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            subMat<unsigned char, unsigned char>,
            subMat<unsigned char, signed char>,
            subMat<unsigned char, unsigned short>,
            subMat<unsigned char, short>,
            subMat<unsigned char, int>,
            subMat<unsigned char, float>,
            subMat<unsigned char, double>
        },
        {
            subMat<signed char, unsigned char>,
            subMat<signed char, signed char>,
            subMat<signed char, unsigned short>,
            subMat<signed char, short>,
            subMat<signed char, int>,
            subMat<signed char, float>,
            subMat<signed char, double>
        },
        {
            0 /*subMat<unsigned short, unsigned char>*/,
            0 /*subMat<unsigned short, signed char>*/,
            subMat<unsigned short, unsigned short>,
            subMat<unsigned short, short>,
            subMat<unsigned short, int>,
            subMat<unsigned short, float>,
            subMat<unsigned short, double>
        },
        {
            0 /*subMat<short, unsigned char>*/,
            0 /*subMat<short, signed char>*/,
            subMat<short, unsigned short>,
            subMat<short, short>,
            subMat<short, int>,
            subMat<short, float>,
            subMat<short, double>
        },
        {
            0 /*subMat<int, unsigned char>*/,
            0 /*subMat<int, signed char>*/,
            0 /*subMat<int, unsigned short>*/,
            0 /*subMat<int, short>*/,
            subMat<int, int>,
            subMat<int, float>,
            subMat<int, double>
        },
        {
            0 /*subMat<float, unsigned char>*/,
            0 /*subMat<float, signed char>*/,
            0 /*subMat<float, unsigned short>*/,
            0 /*subMat<float, short>*/,
            0 /*subMat<float, int>*/,
            subMat<float, float>,
            subMat<float, double>
        },
        {
            0 /*subMat<double, unsigned char>*/,
            0 /*subMat<double, signed char>*/,
            0 /*subMat<double, unsigned short>*/,
            0 /*subMat<double, short>*/,
            0 /*subMat<double, int>*/,
            0 /*subMat<double, float>*/,
            subMat<double, double>
        }
    };

    if (dtype < 0)
        dtype = src1.depth();

    const int sdepth = src1.depth();
    const int ddepth = CV_MAT_DEPTH(dtype);
    const int cn = src1.channels();

    CV_Assert( sdepth <= CV_64F && ddepth <= CV_64F );
    CV_Assert( src2.type() == src1.type() && src2.size() == src1.size() );
    CV_Assert( mask.empty() || (cn == 1 && mask.size() == src1.size() && mask.type() == CV_8U) );

    if (sdepth == CV_64F || ddepth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src1.size(), CV_MAKE_TYPE(ddepth, cn));

    cudaStream_t stream = StreamAccessor::getStream(s);

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

                subMat_v4(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                          stream);

                return;
            }
            else if (sdepth == CV_16U && (src1_.cols & 1) == 0)
            {
                const int vcols = src1_.cols >> 1;

                subMat_v2(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                          stream);

                return;
            }
        }
    }

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, mask, stream);
}

namespace arithm
{
    template <typename T, typename S, typename D>
    void subScalar(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

void cv::gpu::subtract(const GpuMat& src, const Scalar& sc, GpuMat& dst, const GpuMat& mask, int dtype, Stream& s)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, double val, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            subScalar<unsigned char, float, unsigned char>,
            subScalar<unsigned char, float, signed char>,
            subScalar<unsigned char, float, unsigned short>,
            subScalar<unsigned char, float, short>,
            subScalar<unsigned char, float, int>,
            subScalar<unsigned char, float, float>,
            subScalar<unsigned char, double, double>
        },
        {
            subScalar<signed char, float, unsigned char>,
            subScalar<signed char, float, signed char>,
            subScalar<signed char, float, unsigned short>,
            subScalar<signed char, float, short>,
            subScalar<signed char, float, int>,
            subScalar<signed char, float, float>,
            subScalar<signed char, double, double>
        },
        {
            0 /*subScalar<unsigned short, float, unsigned char>*/,
            0 /*subScalar<unsigned short, float, signed char>*/,
            subScalar<unsigned short, float, unsigned short>,
            subScalar<unsigned short, float, short>,
            subScalar<unsigned short, float, int>,
            subScalar<unsigned short, float, float>,
            subScalar<unsigned short, double, double>
        },
        {
            0 /*subScalar<short, float, unsigned char>*/,
            0 /*subScalar<short, float, signed char>*/,
            subScalar<short, float, unsigned short>,
            subScalar<short, float, short>,
            subScalar<short, float, int>,
            subScalar<short, float, float>,
            subScalar<short, double, double>
        },
        {
            0 /*subScalar<int, float, unsigned char>*/,
            0 /*subScalar<int, float, signed char>*/,
            0 /*subScalar<int, float, unsigned short>*/,
            0 /*subScalar<int, float, short>*/,
            subScalar<int, float, int>,
            subScalar<int, float, float>,
            subScalar<int, double, double>
        },
        {
            0 /*subScalar<float, float, unsigned char>*/,
            0 /*subScalar<float, float, signed char>*/,
            0 /*subScalar<float, float, unsigned short>*/,
            0 /*subScalar<float, float, short>*/,
            0 /*subScalar<float, float, int>*/,
            subScalar<float, float, float>,
            subScalar<float, double, double>
        },
        {
            0 /*subScalar<double, double, unsigned char>*/,
            0 /*subScalar<double, double, signed char>*/,
            0 /*subScalar<double, double, unsigned short>*/,
            0 /*subScalar<double, double, short>*/,
            0 /*subScalar<double, double, int>*/,
            0 /*subScalar<double, double, float>*/,
            subScalar<double, double, double>
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

    if (dtype < 0)
        dtype = src.depth();

    const int sdepth = src.depth();
    const int ddepth = CV_MAT_DEPTH(dtype);
    const int cn = src.channels();

    CV_Assert( sdepth <= CV_64F && ddepth <= CV_64F );
    CV_Assert( cn <= 4 );
    CV_Assert( mask.empty() || (cn == 1 && mask.size() == src.size() && mask.type() == CV_8U) );

    if (sdepth == CV_64F || ddepth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), CV_MAKE_TYPE(ddepth, cn));

    cudaStream_t stream = StreamAccessor::getStream(s);

    const npp_func_t npp_func = npp_funcs[sdepth][cn - 1];
    if (ddepth == sdepth && cn > 1 && npp_func != 0)
    {
        npp_func(src, sc, dst, stream);
        return;
    }

    CV_Assert( cn == 1 );

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, sc.val[0], dst, mask, stream);
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

void cv::gpu::multiply(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, double scale, int dtype, Stream& s)
{
    using namespace arithm;

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (src1.type() == CV_8UC4 && src2.type() == CV_32FC1)
    {
        CV_Assert( src1.size() == src2.size() );

        dst.create(src1.size(), src1.type());

        mulMat_8uc4_32f(src1, src2, dst, stream);
    }
    else if (src1.type() == CV_16SC4 && src2.type() == CV_32FC1)
    {
        CV_Assert( src1.size() == src2.size() );

        dst.create(src1.size(), src1.type());

        mulMat_16sc4_32f(src1, src2, dst, stream);
    }
    else
    {
        typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
        static const func_t funcs[7][7] =
        {
            {
                mulMat<unsigned char, float, unsigned char>,
                mulMat<unsigned char, float, signed char>,
                mulMat<unsigned char, float, unsigned short>,
                mulMat<unsigned char, float, short>,
                mulMat<unsigned char, float, int>,
                mulMat<unsigned char, float, float>,
                mulMat<unsigned char, double, double>
            },
            {
                mulMat<signed char, float, unsigned char>,
                mulMat<signed char, float, signed char>,
                mulMat<signed char, float, unsigned short>,
                mulMat<signed char, float, short>,
                mulMat<signed char, float, int>,
                mulMat<signed char, float, float>,
                mulMat<signed char, double, double>
            },
            {
                0 /*mulMat<unsigned short, float, unsigned char>*/,
                0 /*mulMat<unsigned short, float, signed char>*/,
                mulMat<unsigned short, float, unsigned short>,
                mulMat<unsigned short, float, short>,
                mulMat<unsigned short, float, int>,
                mulMat<unsigned short, float, float>,
                mulMat<unsigned short, double, double>
            },
            {
                0 /*mulMat<short, float, unsigned char>*/,
                0 /*mulMat<short, float, signed char>*/,
                mulMat<short, float, unsigned short>,
                mulMat<short, float, short>,
                mulMat<short, float, int>,
                mulMat<short, float, float>,
                mulMat<short, double, double>
            },
            {
                0 /*mulMat<int, float, unsigned char>*/,
                0 /*mulMat<int, float, signed char>*/,
                0 /*mulMat<int, float, unsigned short>*/,
                0 /*mulMat<int, float, short>*/,
                mulMat<int, float, int>,
                mulMat<int, float, float>,
                mulMat<int, double, double>
            },
            {
                0 /*mulMat<float, float, unsigned char>*/,
                0 /*mulMat<float, float, signed char>*/,
                0 /*mulMat<float, float, unsigned short>*/,
                0 /*mulMat<float, float, short>*/,
                0 /*mulMat<float, float, int>*/,
                mulMat<float, float, float>,
                mulMat<float, double, double>
            },
            {
                0 /*mulMat<double, double, unsigned char>*/,
                0 /*mulMat<double, double, signed char>*/,
                0 /*mulMat<double, double, unsigned short>*/,
                0 /*mulMat<double, double, short>*/,
                0 /*mulMat<double, double, int>*/,
                0 /*mulMat<double, double, float>*/,
                mulMat<double, double, double>
            }
        };

        if (dtype < 0)
            dtype = src1.depth();

        const int sdepth = src1.depth();
        const int ddepth = CV_MAT_DEPTH(dtype);
        const int cn = src1.channels();

        CV_Assert( sdepth <= CV_64F && ddepth <= CV_64F );
        CV_Assert( src2.type() == src1.type() && src2.size() == src1.size() );

        if (sdepth == CV_64F || ddepth == CV_64F)
        {
            if (!deviceSupports(NATIVE_DOUBLE))
                CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
        }

        dst.create(src1.size(), CV_MAKE_TYPE(ddepth, cn));

        PtrStepSzb src1_(src1.rows, src1.cols * cn, src1.data, src1.step);
        PtrStepSzb src2_(src1.rows, src1.cols * cn, src2.data, src2.step);
        PtrStepSzb dst_(src1.rows, src1.cols * cn, dst.data, dst.step);

        const func_t func = funcs[sdepth][ddepth];

        if (!func)
            CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

        func(src1_, src2_, dst_, scale, stream);
    }
}

namespace arithm
{
    template <typename T, typename S, typename D>
    void mulScalar(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
}

void cv::gpu::multiply(const GpuMat& src, const Scalar& sc, GpuMat& dst, double scale, int dtype, Stream& s)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            mulScalar<unsigned char, float, unsigned char>,
            mulScalar<unsigned char, float, signed char>,
            mulScalar<unsigned char, float, unsigned short>,
            mulScalar<unsigned char, float, short>,
            mulScalar<unsigned char, float, int>,
            mulScalar<unsigned char, float, float>,
            mulScalar<unsigned char, double, double>
        },
        {
            mulScalar<signed char, float, unsigned char>,
            mulScalar<signed char, float, signed char>,
            mulScalar<signed char, float, unsigned short>,
            mulScalar<signed char, float, short>,
            mulScalar<signed char, float, int>,
            mulScalar<signed char, float, float>,
            mulScalar<signed char, double, double>
        },
        {
            0 /*mulScalar<unsigned short, float, unsigned char>*/,
            0 /*mulScalar<unsigned short, float, signed char>*/,
            mulScalar<unsigned short, float, unsigned short>,
            mulScalar<unsigned short, float, short>,
            mulScalar<unsigned short, float, int>,
            mulScalar<unsigned short, float, float>,
            mulScalar<unsigned short, double, double>
        },
        {
            0 /*mulScalar<short, float, unsigned char>*/,
            0 /*mulScalar<short, float, signed char>*/,
            mulScalar<short, float, unsigned short>,
            mulScalar<short, float, short>,
            mulScalar<short, float, int>,
            mulScalar<short, float, float>,
            mulScalar<short, double, double>
        },
        {
            0 /*mulScalar<int, float, unsigned char>*/,
            0 /*mulScalar<int, float, signed char>*/,
            0 /*mulScalar<int, float, unsigned short>*/,
            0 /*mulScalar<int, float, short>*/,
            mulScalar<int, float, int>,
            mulScalar<int, float, float>,
            mulScalar<int, double, double>
        },
        {
            0 /*mulScalar<float, float, unsigned char>*/,
            0 /*mulScalar<float, float, signed char>*/,
            0 /*mulScalar<float, float, unsigned short>*/,
            0 /*mulScalar<float, float, short>*/,
            0 /*mulScalar<float, float, int>*/,
            mulScalar<float, float, float>,
            mulScalar<float, double, double>
        },
        {
            0 /*mulScalar<double, double, unsigned char>*/,
            0 /*mulScalar<double, double, signed char>*/,
            0 /*mulScalar<double, double, unsigned short>*/,
            0 /*mulScalar<double, double, short>*/,
            0 /*mulScalar<double, double, int>*/,
            0 /*mulScalar<double, double, float>*/,
            mulScalar<double, double, double>
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

    if (dtype < 0)
        dtype = src.depth();

    const int sdepth = src.depth();
    const int ddepth = CV_MAT_DEPTH(dtype);
    const int cn = src.channels();

    CV_Assert( sdepth <= CV_64F && ddepth <= CV_64F );
    CV_Assert( cn <= 4 );

    if (sdepth == CV_64F || ddepth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), CV_MAKE_TYPE(ddepth, cn));

    cudaStream_t stream = StreamAccessor::getStream(s);

    const Scalar nsc(sc.val[0] * scale, sc.val[1] * scale, sc.val[2] * scale, sc.val[3] * scale);

    const npp_func_t npp_func = npp_funcs[sdepth][cn - 1];
    if (ddepth == sdepth && cn > 1 && npp_func != 0)
    {
        npp_func(src, nsc, dst, stream);
        return;
    }

    CV_Assert( cn == 1 );

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, nsc.val[0], dst, stream);
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

void cv::gpu::divide(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, double scale, int dtype, Stream& s)
{
    using namespace arithm;

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (src1.type() == CV_8UC4 && src2.type() == CV_32FC1)
    {
        CV_Assert( src1.size() == src2.size() );

        dst.create(src1.size(), src1.type());

        divMat_8uc4_32f(src1, src2, dst, stream);
    }
    else if (src1.type() == CV_16SC4 && src2.type() == CV_32FC1)
    {
        CV_Assert( src1.size() == src2.size() );

        dst.create(src1.size(), src1.type());

        divMat_16sc4_32f(src1, src2, dst, stream);
    }
    else
    {
        typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, double scale, cudaStream_t stream);
        static const func_t funcs[7][7] =
        {
            {
                divMat<unsigned char, float, unsigned char>,
                divMat<unsigned char, float, signed char>,
                divMat<unsigned char, float, unsigned short>,
                divMat<unsigned char, float, short>,
                divMat<unsigned char, float, int>,
                divMat<unsigned char, float, float>,
                divMat<unsigned char, double, double>
            },
            {
                divMat<signed char, float, unsigned char>,
                divMat<signed char, float, signed char>,
                divMat<signed char, float, unsigned short>,
                divMat<signed char, float, short>,
                divMat<signed char, float, int>,
                divMat<signed char, float, float>,
                divMat<signed char, double, double>
            },
            {
                0 /*divMat<unsigned short, float, unsigned char>*/,
                0 /*divMat<unsigned short, float, signed char>*/,
                divMat<unsigned short, float, unsigned short>,
                divMat<unsigned short, float, short>,
                divMat<unsigned short, float, int>,
                divMat<unsigned short, float, float>,
                divMat<unsigned short, double, double>
            },
            {
                0 /*divMat<short, float, unsigned char>*/,
                0 /*divMat<short, float, signed char>*/,
                divMat<short, float, unsigned short>,
                divMat<short, float, short>,
                divMat<short, float, int>,
                divMat<short, float, float>,
                divMat<short, double, double>
            },
            {
                0 /*divMat<int, float, unsigned char>*/,
                0 /*divMat<int, float, signed char>*/,
                0 /*divMat<int, float, unsigned short>*/,
                0 /*divMat<int, float, short>*/,
                divMat<int, float, int>,
                divMat<int, float, float>,
                divMat<int, double, double>
            },
            {
                0 /*divMat<float, float, unsigned char>*/,
                0 /*divMat<float, float, signed char>*/,
                0 /*divMat<float, float, unsigned short>*/,
                0 /*divMat<float, float, short>*/,
                0 /*divMat<float, float, int>*/,
                divMat<float, float, float>,
                divMat<float, double, double>
            },
            {
                0 /*divMat<double, double, unsigned char>*/,
                0 /*divMat<double, double, signed char>*/,
                0 /*divMat<double, double, unsigned short>*/,
                0 /*divMat<double, double, short>*/,
                0 /*divMat<double, double, int>*/,
                0 /*divMat<double, double, float>*/,
                divMat<double, double, double>
            }
        };

        if (dtype < 0)
            dtype = src1.depth();

        const int sdepth = src1.depth();
        const int ddepth = CV_MAT_DEPTH(dtype);
        const int cn = src1.channels();

        CV_Assert( sdepth <= CV_64F && ddepth <= CV_64F );
        CV_Assert( src2.type() == src1.type() && src2.size() == src1.size() );

        if (sdepth == CV_64F || ddepth == CV_64F)
        {
            if (!deviceSupports(NATIVE_DOUBLE))
                CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
        }

        dst.create(src1.size(), CV_MAKE_TYPE(ddepth, cn));

        PtrStepSzb src1_(src1.rows, src1.cols * cn, src1.data, src1.step);
        PtrStepSzb src2_(src1.rows, src1.cols * cn, src2.data, src2.step);
        PtrStepSzb dst_(src1.rows, src1.cols * cn, dst.data, dst.step);

        const func_t func = funcs[sdepth][ddepth];

        if (!func)
            CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

        func(src1_, src2_, dst_, scale, stream);
    }
}

namespace arithm
{
    template <typename T, typename S, typename D>
    void divScalar(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
}

void cv::gpu::divide(const GpuMat& src, const Scalar& sc, GpuMat& dst, double scale, int dtype, Stream& s)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            divScalar<unsigned char, float, unsigned char>,
            divScalar<unsigned char, float, signed char>,
            divScalar<unsigned char, float, unsigned short>,
            divScalar<unsigned char, float, short>,
            divScalar<unsigned char, float, int>,
            divScalar<unsigned char, float, float>,
            divScalar<unsigned char, double, double>
        },
        {
            divScalar<signed char, float, unsigned char>,
            divScalar<signed char, float, signed char>,
            divScalar<signed char, float, unsigned short>,
            divScalar<signed char, float, short>,
            divScalar<signed char, float, int>,
            divScalar<signed char, float, float>,
            divScalar<signed char, double, double>
        },
        {
            0 /*divScalar<unsigned short, float, unsigned char>*/,
            0 /*divScalar<unsigned short, float, signed char>*/,
            divScalar<unsigned short, float, unsigned short>,
            divScalar<unsigned short, float, short>,
            divScalar<unsigned short, float, int>,
            divScalar<unsigned short, float, float>,
            divScalar<unsigned short, double, double>
        },
        {
            0 /*divScalar<short, float, unsigned char>*/,
            0 /*divScalar<short, float, signed char>*/,
            divScalar<short, float, unsigned short>,
            divScalar<short, float, short>,
            divScalar<short, float, int>,
            divScalar<short, float, float>,
            divScalar<short, double, double>
        },
        {
            0 /*divScalar<int, float, unsigned char>*/,
            0 /*divScalar<int, float, signed char>*/,
            0 /*divScalar<int, float, unsigned short>*/,
            0 /*divScalar<int, float, short>*/,
            divScalar<int, float, int>,
            divScalar<int, float, float>,
            divScalar<int, double, double>
        },
        {
            0 /*divScalar<float, float, unsigned char>*/,
            0 /*divScalar<float, float, signed char>*/,
            0 /*divScalar<float, float, unsigned short>*/,
            0 /*divScalar<float, float, short>*/,
            0 /*divScalar<float, float, int>*/,
            divScalar<float, float, float>,
            divScalar<float, double, double>
        },
        {
            0 /*divScalar<double, double, unsigned char>*/,
            0 /*divScalar<double, double, signed char>*/,
            0 /*divScalar<double, double, unsigned short>*/,
            0 /*divScalar<double, double, short>*/,
            0 /*divScalar<double, double, int>*/,
            0 /*divScalar<double, double, float>*/,
            divScalar<double, double, double>
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

    if (dtype < 0)
        dtype = src.depth();

    const int sdepth = src.depth();
    const int ddepth = CV_MAT_DEPTH(dtype);
    const int cn = src.channels();

    CV_Assert( sdepth <= CV_64F && ddepth <= CV_64F );
    CV_Assert( cn <= 4 );

    if (sdepth == CV_64F || ddepth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), CV_MAKE_TYPE(ddepth, cn));

    cudaStream_t stream = StreamAccessor::getStream(s);

    const Scalar nsc(sc.val[0] / scale, sc.val[1] / scale, sc.val[2] / scale, sc.val[3] / scale);

    const npp_func_t npp_func = npp_funcs[sdepth][cn - 1];
    if (ddepth == sdepth && cn > 1 && npp_func != 0)
    {
        npp_func(src, nsc, dst, stream);
        return;
    }

    CV_Assert( cn == 1 );

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, nsc.val[0], dst, stream);
}

namespace arithm
{
    template <typename T, typename S, typename D>
    void divInv(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
}

void cv::gpu::divide(double scale, const GpuMat& src, GpuMat& dst, int dtype, Stream& s)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[7][7] =
    {
        {
            divInv<unsigned char, float, unsigned char>,
            divInv<unsigned char, float, signed char>,
            divInv<unsigned char, float, unsigned short>,
            divInv<unsigned char, float, short>,
            divInv<unsigned char, float, int>,
            divInv<unsigned char, float, float>,
            divInv<unsigned char, double, double>
        },
        {
            divInv<signed char, float, unsigned char>,
            divInv<signed char, float, signed char>,
            divInv<signed char, float, unsigned short>,
            divInv<signed char, float, short>,
            divInv<signed char, float, int>,
            divInv<signed char, float, float>,
            divInv<signed char, double, double>
        },
        {
            0 /*divInv<unsigned short, float, unsigned char>*/,
            0 /*divInv<unsigned short, float, signed char>*/,
            divInv<unsigned short, float, unsigned short>,
            divInv<unsigned short, float, short>,
            divInv<unsigned short, float, int>,
            divInv<unsigned short, float, float>,
            divInv<unsigned short, double, double>
        },
        {
            0 /*divInv<short, float, unsigned char>*/,
            0 /*divInv<short, float, signed char>*/,
            divInv<short, float, unsigned short>,
            divInv<short, float, short>,
            divInv<short, float, int>,
            divInv<short, float, float>,
            divInv<short, double, double>
        },
        {
            0 /*divInv<int, float, unsigned char>*/,
            0 /*divInv<int, float, signed char>*/,
            0 /*divInv<int, float, unsigned short>*/,
            0 /*divInv<int, float, short>*/,
            divInv<int, float, int>,
            divInv<int, float, float>,
            divInv<int, double, double>
        },
        {
            0 /*divInv<float, float, unsigned char>*/,
            0 /*divInv<float, float, signed char>*/,
            0 /*divInv<float, float, unsigned short>*/,
            0 /*divInv<float, float, short>*/,
            0 /*divInv<float, float, int>*/,
            divInv<float, float, float>,
            divInv<float, double, double>
        },
        {
            0 /*divInv<double, double, unsigned char>*/,
            0 /*divInv<double, double, signed char>*/,
            0 /*divInv<double, double, unsigned short>*/,
            0 /*divInv<double, double, short>*/,
            0 /*divInv<double, double, int>*/,
            0 /*divInv<double, double, float>*/,
            divInv<double, double, double>
        }
    };

    if (dtype < 0)
        dtype = src.depth();

    const int sdepth = src.depth();
    const int ddepth = CV_MAT_DEPTH(dtype);
    const int cn = src.channels();

    CV_Assert( sdepth <= CV_64F && ddepth <= CV_64F );
    CV_Assert( cn == 1 );

    if (sdepth == CV_64F || ddepth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), CV_MAKE_TYPE(ddepth, cn));

    cudaStream_t stream = StreamAccessor::getStream(s);

    const func_t func = funcs[sdepth][ddepth];

    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src, scale, dst, stream);
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

void cv::gpu::absdiff(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& s)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        absDiffMat<unsigned char>,
        absDiffMat<signed char>,
        absDiffMat<unsigned short>,
        absDiffMat<short>,
        absDiffMat<int>,
        absDiffMat<float>,
        absDiffMat<double>
    };

    const int depth = src1.depth();
    const int cn = src1.channels();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src2.type() == src1.type() && src2.size() == src1.size() );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src1.size(), src1.type());

    cudaStream_t stream = StreamAccessor::getStream(s);

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

                absDiffMat_v4(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                              PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                              PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                              stream);

                return;
            }
            else if (depth == CV_16U && (src1_.cols & 1) == 0)
            {
                const int vcols = src1_.cols >> 1;

                absDiffMat_v2(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                              PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                              PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                              stream);

                return;
            }
        }
    }

    const func_t func = funcs[depth];

    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, stream);
}

namespace arithm
{
    template <typename T, typename S>
    void absDiffScalar(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
}

void cv::gpu::absdiff(const GpuMat& src1, const Scalar& src2, GpuMat& dst, Stream& stream)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, double val, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        absDiffScalar<unsigned char, float>,
        absDiffScalar<signed char, float>,
        absDiffScalar<unsigned short, float>,
        absDiffScalar<short, float>,
        absDiffScalar<int, float>,
        absDiffScalar<float, float>,
        absDiffScalar<double, double>
    };

    const int depth = src1.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src1.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src1.size(), src1.type());

    funcs[depth](src1, src2.val[0], dst, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// abs

namespace arithm
{
    template <typename T>
    void absMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

void cv::gpu::abs(const GpuMat& src, GpuMat& dst, Stream& stream)
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

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), src.type());

    funcs[depth](src, dst, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// sqr

namespace arithm
{
    template <typename T>
    void sqrMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

void cv::gpu::sqr(const GpuMat& src, GpuMat& dst, Stream& stream)
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

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), src.type());

    funcs[depth](src, dst, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// sqrt

namespace arithm
{
    template <typename T>
    void sqrtMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

void cv::gpu::sqrt(const GpuMat& src, GpuMat& dst, Stream& stream)
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

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), src.type());

    funcs[depth](src, dst, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// log

namespace arithm
{
    template <typename T>
    void logMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

void cv::gpu::log(const GpuMat& src, GpuMat& dst, Stream& stream)
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

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), src.type());

    funcs[depth](src, dst, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// exp

namespace arithm
{
    template <typename T>
    void expMat(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
}

void cv::gpu::exp(const GpuMat& src, GpuMat& dst, Stream& stream)
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

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), src.type());

    funcs[depth](src, dst, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// compare

namespace arithm
{
    void cmpMatEq_v4(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);
    void cmpMatNe_v4(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);
    void cmpMatLt_v4(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);
    void cmpMatLe_v4(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);

    template <typename T> void cmpMatEq(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void cmpMatNe(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void cmpMatLt(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    template <typename T> void cmpMatLe(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
}

void cv::gpu::compare(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, int cmpop, Stream& s)
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

    typedef void (*func_v4_t)(PtrStepSz<unsigned int> src1, PtrStepSz<unsigned int> src2, PtrStepSz<unsigned int> dst, cudaStream_t stream);
    static const func_v4_t funcs_v4[] =
    {
        cmpMatEq_v4, cmpMatNe_v4, cmpMatLt_v4, cmpMatLe_v4
    };

    const int depth = src1.depth();
    const int cn = src1.channels();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src2.size() == src1.size() && src2.type() == src1.type() );
    CV_Assert( cmpop >= CMP_EQ && cmpop <= CMP_NE );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src1.size(), CV_MAKE_TYPE(CV_8U, cn));

    cudaStream_t stream = StreamAccessor::getStream(s);

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

void cv::gpu::compare(const GpuMat& src, Scalar sc, GpuMat& dst, int cmpop, Stream& stream)
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

    const int depth = src.depth();
    const int cn = src.channels();

    CV_Assert( depth <= CV_64F );
    CV_Assert( cn <= 4 );
    CV_Assert( cmpop >= CMP_EQ && cmpop <= CMP_NE );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), CV_MAKE_TYPE(CV_8U, cn));

    cast_func[depth](sc);

    funcs[depth][cmpop](src, cn, sc.val, dst, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// Unary bitwise logical operations

namespace arithm
{
    template <typename T> void bitMatNot(PtrStepSzb src, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

void cv::gpu::bitwise_not(const GpuMat& src, GpuMat& dst, const GpuMat& mask, Stream& s)
{
    using namespace arithm;

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    dst.create(src.size(), src.type());

    cudaStream_t stream = StreamAccessor::getStream(s);

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

namespace arithm
{
    template <typename T> void bitMatAnd(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template <typename T> void bitMatOr(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
    template <typename T> void bitMatXor(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, PtrStepb mask, cudaStream_t stream);
}

void cv::gpu::bitwise_and(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, Stream& s)
{
    using namespace arithm;

    const int depth = src1.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src2.size() == src1.size() && src2.type() == src1.type() );
    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src1.size()) );

    dst.create(src1.size(), src1.type());

    cudaStream_t stream = StreamAccessor::getStream(s);

    const int bcols = (int) (src1.cols * src1.elemSize());

    if ((bcols & 3) == 0)
    {
        const int vcols = bcols >> 2;

        bitMatAnd<unsigned int>(
                    PtrStepSzb(src1.rows, vcols, src1.data, src1.step),
                    PtrStepSzb(src1.rows, vcols, src2.data, src2.step),
                    PtrStepSzb(src1.rows, vcols, dst.data, dst.step),
                    mask, stream);
    }
    else if ((bcols & 1) == 0)
    {
        const int vcols = bcols >> 1;

        bitMatAnd<unsigned short>(
                    PtrStepSzb(src1.rows, vcols, src1.data, src1.step),
                    PtrStepSzb(src1.rows, vcols, src2.data, src2.step),
                    PtrStepSzb(src1.rows, vcols, dst.data, dst.step),
                    mask, stream);
    }
    else
    {

        bitMatAnd<unsigned char>(
                    PtrStepSzb(src1.rows, bcols, src1.data, src1.step),
                    PtrStepSzb(src1.rows, bcols, src2.data, src2.step),
                    PtrStepSzb(src1.rows, bcols, dst.data, dst.step),
                    mask, stream);
    }
}

void cv::gpu::bitwise_or(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, Stream& s)
{
    using namespace arithm;

    const int depth = src1.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src2.size() == src1.size() && src2.type() == src1.type() );
    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src1.size()) );

    dst.create(src1.size(), src1.type());

    cudaStream_t stream = StreamAccessor::getStream(s);

    const int bcols = (int) (src1.cols * src1.elemSize());

    if ((bcols & 3) == 0)
    {
        const int vcols = bcols >> 2;

        bitMatOr<unsigned int>(
                    PtrStepSzb(src1.rows, vcols, src1.data, src1.step),
                    PtrStepSzb(src1.rows, vcols, src2.data, src2.step),
                    PtrStepSzb(src1.rows, vcols, dst.data, dst.step),
                    mask, stream);
    }
    else if ((bcols & 1) == 0)
    {
        const int vcols = bcols >> 1;

        bitMatOr<unsigned short>(
                    PtrStepSzb(src1.rows, vcols, src1.data, src1.step),
                    PtrStepSzb(src1.rows, vcols, src2.data, src2.step),
                    PtrStepSzb(src1.rows, vcols, dst.data, dst.step),
                    mask, stream);
    }
    else
    {

        bitMatOr<unsigned char>(
                    PtrStepSzb(src1.rows, bcols, src1.data, src1.step),
                    PtrStepSzb(src1.rows, bcols, src2.data, src2.step),
                    PtrStepSzb(src1.rows, bcols, dst.data, dst.step),
                    mask, stream);
    }
}

void cv::gpu::bitwise_xor(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, Stream& s)
{
    using namespace arithm;

    const int depth = src1.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src2.size() == src1.size() && src2.type() == src1.type() );
    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src1.size()) );

    dst.create(src1.size(), src1.type());

    cudaStream_t stream = StreamAccessor::getStream(s);

    const int bcols = (int) (src1.cols * src1.elemSize());

    if ((bcols & 3) == 0)
    {
        const int vcols = bcols >> 2;

        bitMatXor<unsigned int>(
                    PtrStepSzb(src1.rows, vcols, src1.data, src1.step),
                    PtrStepSzb(src1.rows, vcols, src2.data, src2.step),
                    PtrStepSzb(src1.rows, vcols, dst.data, dst.step),
                    mask, stream);
    }
    else if ((bcols & 1) == 0)
    {
        const int vcols = bcols >> 1;

        bitMatXor<unsigned short>(
                    PtrStepSzb(src1.rows, vcols, src1.data, src1.step),
                    PtrStepSzb(src1.rows, vcols, src2.data, src2.step),
                    PtrStepSzb(src1.rows, vcols, dst.data, dst.step),
                    mask, stream);
    }
    else
    {

        bitMatXor<unsigned char>(
                    PtrStepSzb(src1.rows, bcols, src1.data, src1.step),
                    PtrStepSzb(src1.rows, bcols, src2.data, src2.step),
                    PtrStepSzb(src1.rows, bcols, dst.data, dst.step),
                    mask, stream);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Binary bitwise logical operations with scalars

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

void cv::gpu::bitwise_and(const GpuMat& src, const Scalar& sc, GpuMat& dst, Stream& stream)
{
    using namespace arithm;

    typedef void (*func_t)(const GpuMat& src, Scalar sc, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[5][4] =
    {
        {BitScalar<unsigned char, bitScalarAnd<unsigned char> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiAndC_8u_C3R >::call, BitScalar4< bitScalarAnd<unsigned int> >::call},
        {0,0,0,0},
        {BitScalar<unsigned short, bitScalarAnd<unsigned short> >::call, 0, NppBitwiseC<CV_16U, 3, nppiAndC_16u_C3R>::call, NppBitwiseC<CV_16U, 4, nppiAndC_16u_C4R>::call},
        {0,0,0,0},
        {BitScalar<int, bitScalarAnd<int> >::call                      , 0, NppBitwiseC<CV_32S, 3, nppiAndC_32s_C3R>::call, NppBitwiseC<CV_32S, 4, nppiAndC_32s_C4R>::call}
    };

    const int depth = src.depth();
    const int cn = src.channels();

    CV_Assert( depth == CV_8U || depth == CV_16U || depth == CV_32S );
    CV_Assert( cn == 1 || cn == 3 || cn == 4 );

    dst.create(src.size(), src.type());

    funcs[depth][cn - 1](src, sc, dst, StreamAccessor::getStream(stream));
}

void cv::gpu::bitwise_or(const GpuMat& src, const Scalar& sc, GpuMat& dst, Stream& stream)
{
    using namespace arithm;

    typedef void (*func_t)(const GpuMat& src, Scalar sc, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[5][4] =
    {
        {BitScalar<unsigned char, bitScalarOr<unsigned char> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiOrC_8u_C3R >::call, BitScalar4< bitScalarOr<unsigned int> >::call},
        {0,0,0,0},
        {BitScalar<unsigned short, bitScalarOr<unsigned short> >::call, 0, NppBitwiseC<CV_16U, 3, nppiOrC_16u_C3R>::call, NppBitwiseC<CV_16U, 4, nppiOrC_16u_C4R>::call},
        {0,0,0,0},
        {BitScalar<int, bitScalarOr<int> >::call                      , 0, NppBitwiseC<CV_32S, 3, nppiOrC_32s_C3R>::call, NppBitwiseC<CV_32S, 4, nppiOrC_32s_C4R>::call}
    };

    const int depth = src.depth();
    const int cn = src.channels();

    CV_Assert( depth == CV_8U || depth == CV_16U || depth == CV_32S );
    CV_Assert( cn == 1 || cn == 3 || cn == 4 );

    dst.create(src.size(), src.type());

    funcs[depth][cn - 1](src, sc, dst, StreamAccessor::getStream(stream));
}

void cv::gpu::bitwise_xor(const GpuMat& src, const Scalar& sc, GpuMat& dst, Stream& stream)
{
    using namespace arithm;

    typedef void (*func_t)(const GpuMat& src, Scalar sc, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[5][4] =
    {
        {BitScalar<unsigned char, bitScalarXor<unsigned char> >::call  , 0, NppBitwiseC<CV_8U , 3, nppiXorC_8u_C3R >::call, BitScalar4< bitScalarXor<unsigned int> >::call},
        {0,0,0,0},
        {BitScalar<unsigned short, bitScalarXor<unsigned short> >::call, 0, NppBitwiseC<CV_16U, 3, nppiXorC_16u_C3R>::call, NppBitwiseC<CV_16U, 4, nppiXorC_16u_C4R>::call},
        {0,0,0,0},
        {BitScalar<int, bitScalarXor<int> >::call                      , 0, NppBitwiseC<CV_32S, 3, nppiXorC_32s_C3R>::call, NppBitwiseC<CV_32S, 4, nppiXorC_32s_C4R>::call}
    };

    const int depth = src.depth();
    const int cn = src.channels();

    CV_Assert( depth == CV_8U || depth == CV_16U || depth == CV_32S );
    CV_Assert( cn == 1 || cn == 3 || cn == 4 );

    dst.create(src.size(), src.type());

    funcs[depth][cn - 1](src, sc, dst, StreamAccessor::getStream(stream));
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

void cv::gpu::rshift(const GpuMat& src, Scalar_<int> sc, GpuMat& dst, Stream& stream)
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

    CV_Assert(src.depth() < CV_32F);
    CV_Assert(src.channels() == 1 || src.channels() == 3 || src.channels() == 4);

    dst.create(src.size(), src.type());

    funcs[src.depth()][src.channels() - 1](src, sc, dst, StreamAccessor::getStream(stream));
}

void cv::gpu::lshift(const GpuMat& src, Scalar_<int> sc, GpuMat& dst, Stream& stream)
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

    CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32S);
    CV_Assert(src.channels() == 1 || src.channels() == 3 || src.channels() == 4);

    dst.create(src.size(), src.type());

    funcs[src.depth()][src.channels() - 1](src, sc, dst, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// Minimum and maximum operations

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

void cv::gpu::min(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& s)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        minMat<unsigned char>,
        minMat<signed char>,
        minMat<unsigned short>,
        minMat<short>,
        minMat<int>,
        minMat<float>,
        minMat<double>
    };

    const int depth = src1.depth();
    const int cn = src1.channels();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src2.type() == src1.type() && src2.size() == src1.size() );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src1.size(), src1.type());

    cudaStream_t stream = StreamAccessor::getStream(s);

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

                minMat_v4(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                          stream);

                return;
            }
            else if (depth == CV_16U && (src1_.cols & 1) == 0)
            {
                const int vcols = src1_.cols >> 1;

                minMat_v2(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                          stream);

                return;
            }
        }
    }

    const func_t func = funcs[depth];

    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, stream);
}

void cv::gpu::max(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& s)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, PtrStepSzb src2, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        maxMat<unsigned char>,
        maxMat<signed char>,
        maxMat<unsigned short>,
        maxMat<short>,
        maxMat<int>,
        maxMat<float>,
        maxMat<double>
    };

    const int depth = src1.depth();
    const int cn = src1.channels();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src2.type() == src1.type() && src2.size() == src1.size() );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src1.size(), src1.type());

    cudaStream_t stream = StreamAccessor::getStream(s);

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

                maxMat_v4(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                          stream);

                return;
            }
            else if (depth == CV_16U && (src1_.cols & 1) == 0)
            {
                const int vcols = src1_.cols >> 1;

                maxMat_v2(PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src1_.data, src1_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) src2_.data, src2_.step),
                          PtrStepSz<unsigned int>(src1_.rows, vcols, (unsigned int*) dst_.data, dst_.step),
                          stream);

                return;
            }
        }
    }

    const func_t func = funcs[depth];

    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, src2_, dst_, stream);
}

namespace
{
    template <typename T> double castScalar(double val)
    {
        return saturate_cast<T>(val);
    }
}

void cv::gpu::min(const GpuMat& src, double val, GpuMat& dst, Stream& stream)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        minScalar<unsigned char>,
        minScalar<signed char>,
        minScalar<unsigned short>,
        minScalar<short>,
        minScalar<int>,
        minScalar<float>,
        minScalar<double>
    };

    typedef double (*cast_func_t)(double sc);
    static const cast_func_t cast_func[] =
    {
        castScalar<unsigned char>, castScalar<signed char>, castScalar<unsigned short>, castScalar<short>, castScalar<int>, castScalar<float>, castScalar<double>
    };

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), src.type());

    funcs[depth](src, cast_func[depth](val), dst, StreamAccessor::getStream(stream));
}

void cv::gpu::max(const GpuMat& src, double val, GpuMat& dst, Stream& stream)
{
    using namespace arithm;

    typedef void (*func_t)(PtrStepSzb src1, double src2, PtrStepSzb dst, cudaStream_t stream);
    static const func_t funcs[] =
    {
        maxScalar<unsigned char>,
        maxScalar<signed char>,
        maxScalar<unsigned short>,
        maxScalar<short>,
        maxScalar<int>,
        maxScalar<float>,
        maxScalar<double>
    };

    typedef double (*cast_func_t)(double sc);
    static const cast_func_t cast_func[] =
    {
        castScalar<unsigned char>, castScalar<signed char>, castScalar<unsigned short>, castScalar<short>, castScalar<int>, castScalar<float>, castScalar<double>
    };

    const int depth = src.depth();

    CV_Assert( depth <= CV_64F );
    CV_Assert( src.channels() == 1 );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), src.type());

    funcs[depth](src, cast_func[depth](val), dst, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// threshold

namespace arithm
{
    template <typename T>
    void threshold(PtrStepSzb src, PtrStepSzb dst, double thresh, double maxVal, int type, cudaStream_t stream);
}

double cv::gpu::threshold(const GpuMat& src, GpuMat& dst, double thresh, double maxVal, int type, Stream& s)
{
    const int depth = src.depth();

    CV_Assert( src.channels() == 1 && depth <= CV_64F );
    CV_Assert( type <= THRESH_TOZERO_INV );

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), src.type());

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (src.type() == CV_32FC1 && type == THRESH_TRUNC)
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
// pow

namespace arithm
{
    template<typename T> void pow(PtrStepSzb src, double power, PtrStepSzb dst, cudaStream_t stream);
}

void cv::gpu::pow(const GpuMat& src, double power, GpuMat& dst, Stream& stream)
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

    const int depth = src.depth();
    const int cn = src.channels();

    CV_Assert(depth <= CV_64F);

    if (depth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src.size(), src.type());

    PtrStepSzb src_(src.rows, src.cols * cn, src.data, src.step);
    PtrStepSzb dst_(src.rows, src.cols * cn, dst.data, dst.step);

    funcs[depth](src_, power, dst_, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// alphaComp

namespace
{
    template <int DEPTH> struct NppAlphaCompFunc
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        typedef NppStatus (*func_t)(const npp_t* pSrc1, int nSrc1Step, const npp_t* pSrc2, int nSrc2Step, npp_t* pDst, int nDstStep, NppiSize oSizeROI, NppiAlphaOp eAlphaOp);
    };

    template <int DEPTH, typename NppAlphaCompFunc<DEPTH>::func_t func> struct NppAlphaComp
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        static void call(const GpuMat& img1, const GpuMat& img2, GpuMat& dst, NppiAlphaOp eAlphaOp, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize oSizeROI;
            oSizeROI.width = img1.cols;
            oSizeROI.height = img2.rows;

            nppSafeCall( func(img1.ptr<npp_t>(), static_cast<int>(img1.step), img2.ptr<npp_t>(), static_cast<int>(img2.step),
                              dst.ptr<npp_t>(), static_cast<int>(dst.step), oSizeROI, eAlphaOp) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

void cv::gpu::alphaComp(const GpuMat& img1, const GpuMat& img2, GpuMat& dst, int alpha_op, Stream& stream)
{
    static const NppiAlphaOp npp_alpha_ops[] = {
        NPPI_OP_ALPHA_OVER,
        NPPI_OP_ALPHA_IN,
        NPPI_OP_ALPHA_OUT,
        NPPI_OP_ALPHA_ATOP,
        NPPI_OP_ALPHA_XOR,
        NPPI_OP_ALPHA_PLUS,
        NPPI_OP_ALPHA_OVER_PREMUL,
        NPPI_OP_ALPHA_IN_PREMUL,
        NPPI_OP_ALPHA_OUT_PREMUL,
        NPPI_OP_ALPHA_ATOP_PREMUL,
        NPPI_OP_ALPHA_XOR_PREMUL,
        NPPI_OP_ALPHA_PLUS_PREMUL,
        NPPI_OP_ALPHA_PREMUL
    };

    typedef void (*func_t)(const GpuMat& img1, const GpuMat& img2, GpuMat& dst, NppiAlphaOp eAlphaOp, cudaStream_t stream);

    static const func_t funcs[] =
    {
        NppAlphaComp<CV_8U, nppiAlphaComp_8u_AC4R>::call,
        0,
        NppAlphaComp<CV_16U, nppiAlphaComp_16u_AC4R>::call,
        0,
        NppAlphaComp<CV_32S, nppiAlphaComp_32s_AC4R>::call,
        NppAlphaComp<CV_32F, nppiAlphaComp_32f_AC4R>::call
    };

    CV_Assert( img1.type() == CV_8UC4 || img1.type() == CV_16UC4 || img1.type() == CV_32SC4 || img1.type() == CV_32FC4 );
    CV_Assert( img1.size() == img2.size() && img1.type() == img2.type() );

    dst.create(img1.size(), img1.type());

    const func_t func = funcs[img1.depth()];

    func(img1, img2, dst, npp_alpha_ops[alpha_op], StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// addWeighted

namespace arithm
{
    template <typename T1, typename T2, typename D>
    void addWeighted(PtrStepSzb src1, double alpha, PtrStepSzb src2, double beta, double gamma, PtrStepSzb dst, cudaStream_t stream);
}

void cv::gpu::addWeighted(const GpuMat& src1, double alpha, const GpuMat& src2, double beta, double gamma, GpuMat& dst, int ddepth, Stream& stream)
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

    int sdepth1 = src1.depth();
    int sdepth2 = src2.depth();
    ddepth = ddepth >= 0 ? CV_MAT_DEPTH(ddepth) : std::max(sdepth1, sdepth2);
    const int cn = src1.channels();

    CV_Assert( src2.size() == src1.size() && src2.channels() == cn );
    CV_Assert( sdepth1 <= CV_64F && sdepth2 <= CV_64F && ddepth <= CV_64F );

    if (sdepth1 == CV_64F || sdepth2 == CV_64F || ddepth == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    dst.create(src1.size(), CV_MAKE_TYPE(ddepth, cn));

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
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    func(src1_, alpha, src2_, beta, gamma, dst_, StreamAccessor::getStream(stream));
}

#endif
