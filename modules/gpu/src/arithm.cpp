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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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
using namespace std;

#if !defined (HAVE_CUDA)

void cv::gpu::add(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::add(const GpuMat&, const Scalar&, GpuMat&) { throw_nogpu(); }
void cv::gpu::subtract(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::subtract(const GpuMat&, const Scalar&, GpuMat&) { throw_nogpu(); }
void cv::gpu::multiply(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::multiply(const GpuMat&, const Scalar&, GpuMat&) { throw_nogpu(); }
void cv::gpu::divide(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::divide(const GpuMat&, const Scalar&, GpuMat&) { throw_nogpu(); }
void cv::gpu::transpose(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::absdiff(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::absdiff(const GpuMat&, const Scalar&, GpuMat&) { throw_nogpu(); }
void cv::gpu::compare(const GpuMat&, const GpuMat&, GpuMat&, int) { throw_nogpu(); }
void cv::gpu::meanStdDev(const GpuMat&, Scalar&, Scalar&) { throw_nogpu(); }
double cv::gpu::norm(const GpuMat&, int) { throw_nogpu(); return 0.0; }
double cv::gpu::norm(const GpuMat&, const GpuMat&, int) { throw_nogpu(); return 0.0; }
void cv::gpu::flip(const GpuMat&, GpuMat&, int) { throw_nogpu(); }
Scalar cv::gpu::sum(const GpuMat&) { throw_nogpu(); return Scalar(); }
void cv::gpu::minMax(const GpuMat&, double*, double*) { throw_nogpu(); }
void cv::gpu::minMaxLoc(const GpuMat&, double*, double*, Point*, Point*) { throw_nogpu(); }
void cv::gpu::LUT(const GpuMat&, const Mat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::exp(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::log(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::magnitude(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::magnitudeSqr(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::magnitude(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::magnitude(const GpuMat&, const GpuMat&, GpuMat&, const Stream&) { throw_nogpu(); }
void cv::gpu::magnitudeSqr(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::magnitudeSqr(const GpuMat&, const GpuMat&, GpuMat&, const Stream&) { throw_nogpu(); }
void cv::gpu::phase(const GpuMat&, const GpuMat&, GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::phase(const GpuMat&, const GpuMat&, GpuMat&, bool, const Stream&) { throw_nogpu(); }
void cv::gpu::cartToPolar(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::cartToPolar(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool, const Stream&) { throw_nogpu(); }
void cv::gpu::polarToCart(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool) { throw_nogpu(); }
void cv::gpu::polarToCart(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool, const Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_not(const GpuMat&, GpuMat&, const GpuMat&) { throw_nogpu(); }
void cv::gpu::bitwise_not(const GpuMat&, GpuMat&, const GpuMat&, const Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_or(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&) { throw_nogpu(); }
void cv::gpu::bitwise_or(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, const Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_and(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&) { throw_nogpu(); }
void cv::gpu::bitwise_and(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, const Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_xor(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&) { throw_nogpu(); }
void cv::gpu::bitwise_xor(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, const Stream&) { throw_nogpu(); }
cv::gpu::GpuMat cv::gpu::operator ~ (const GpuMat&) { throw_nogpu(); return GpuMat(); }
cv::gpu::GpuMat cv::gpu::operator | (const GpuMat&, const GpuMat&) { throw_nogpu(); return GpuMat(); }
cv::gpu::GpuMat cv::gpu::operator & (const GpuMat&, const GpuMat&) { throw_nogpu(); return GpuMat(); }
cv::gpu::GpuMat cv::gpu::operator ^ (const GpuMat&, const GpuMat&) { throw_nogpu(); return GpuMat(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// add subtract multiply divide

namespace
{
    typedef NppStatus (*npp_arithm_8u_t)(const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step, Npp8u* pDst, int nDstStep,
                                         NppiSize oSizeROI, int nScaleFactor);
    typedef NppStatus (*npp_arithm_32s_t)(const Npp32s* pSrc1, int nSrc1Step, const Npp32s* pSrc2, int nSrc2Step, Npp32s* pDst,
                                          int nDstStep, NppiSize oSizeROI);
    typedef NppStatus (*npp_arithm_32f_t)(const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step, Npp32f* pDst,
                                          int nDstStep, NppiSize oSizeROI);

    void nppArithmCaller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst,
                         npp_arithm_8u_t npp_func_8uc1, npp_arithm_8u_t npp_func_8uc4,
                         npp_arithm_32s_t npp_func_32sc1, npp_arithm_32f_t npp_func_32fc1)
    {
        CV_DbgAssert(src1.size() == src2.size() && src1.type() == src2.type());

        CV_Assert(src1.type() == CV_8UC1 || src1.type() == CV_8UC4 || src1.type() == CV_32SC1 || src1.type() == CV_32FC1);

        dst.create( src1.size(), src1.type() );

        NppiSize sz;
        sz.width  = src1.cols;
        sz.height = src1.rows;

        switch (src1.type())
        {
        case CV_8UC1:
            nppSafeCall( npp_func_8uc1(src1.ptr<Npp8u>(), src1.step,
                src2.ptr<Npp8u>(), src2.step,
                dst.ptr<Npp8u>(), dst.step, sz, 0) );
            break;
        case CV_8UC4:
            nppSafeCall( npp_func_8uc4(src1.ptr<Npp8u>(), src1.step,
                src2.ptr<Npp8u>(), src2.step,
                dst.ptr<Npp8u>(), dst.step, sz, 0) );
            break;
        case CV_32SC1:
            nppSafeCall( npp_func_32sc1(src1.ptr<Npp32s>(), src1.step,
                src2.ptr<Npp32s>(), src2.step,
                dst.ptr<Npp32s>(), dst.step, sz) );
            break;
        case CV_32FC1:
            nppSafeCall( npp_func_32fc1(src1.ptr<Npp32f>(), src1.step,
                src2.ptr<Npp32f>(), src2.step,
                dst.ptr<Npp32f>(), dst.step, sz) );
            break;
        default:
            CV_Assert(!"Unsupported source type");
        }
    }

    template<int SCN> struct NppArithmScalarFunc;
    template<> struct NppArithmScalarFunc<1>
    {
        typedef NppStatus (*func_ptr)(const Npp32f *pSrc, int nSrcStep, Npp32f nValue, Npp32f *pDst,
                                      int nDstStep, NppiSize oSizeROI);
    };
    template<> struct NppArithmScalarFunc<2>
    {
        typedef NppStatus (*func_ptr)(const Npp32fc *pSrc, int nSrcStep, Npp32fc nValue, Npp32fc *pDst,
                                      int nDstStep, NppiSize oSizeROI);
    };

    template<int SCN, typename NppArithmScalarFunc<SCN>::func_ptr func> struct NppArithmScalar;
    template<typename NppArithmScalarFunc<1>::func_ptr func> struct NppArithmScalar<1, func>
    {
        static void calc(const GpuMat& src, const Scalar& sc, GpuMat& dst)
        {
            dst.create(src.size(), src.type());

            NppiSize sz;
            sz.width  = src.cols;
            sz.height = src.rows;

            nppSafeCall( func(src.ptr<Npp32f>(), src.step, (Npp32f)sc[0], dst.ptr<Npp32f>(), dst.step, sz) );
        }
    };
    template<typename NppArithmScalarFunc<2>::func_ptr func> struct NppArithmScalar<2, func>
    {
        static void calc(const GpuMat& src, const Scalar& sc, GpuMat& dst)
        {
            dst.create(src.size(), src.type());

            NppiSize sz;
            sz.width  = src.cols;
            sz.height = src.rows;

            Npp32fc nValue;
            nValue.re = (Npp32f)sc[0];
            nValue.im = (Npp32f)sc[1];

            nppSafeCall( func(src.ptr<Npp32fc>(), src.step, nValue, dst.ptr<Npp32fc>(), dst.step, sz) );
        }
    };
}

void cv::gpu::add(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)
{
    nppArithmCaller(src1, src2, dst, nppiAdd_8u_C1RSfs, nppiAdd_8u_C4RSfs, nppiAdd_32s_C1R, nppiAdd_32f_C1R);
}

void cv::gpu::subtract(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)
{
    nppArithmCaller(src2, src1, dst, nppiSub_8u_C1RSfs, nppiSub_8u_C4RSfs, nppiSub_32s_C1R, nppiSub_32f_C1R);
}

void cv::gpu::multiply(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)
{
    nppArithmCaller(src1, src2, dst, nppiMul_8u_C1RSfs, nppiMul_8u_C4RSfs, nppiMul_32s_C1R, nppiMul_32f_C1R);
}

void cv::gpu::divide(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)
{
    nppArithmCaller(src2, src1, dst, nppiDiv_8u_C1RSfs, nppiDiv_8u_C4RSfs, nppiDiv_32s_C1R, nppiDiv_32f_C1R);
}

void cv::gpu::add(const GpuMat& src, const Scalar& sc, GpuMat& dst)
{
    typedef void (*caller_t)(const GpuMat& src, const Scalar& sc, GpuMat& dst);
    static const caller_t callers[] = {0, NppArithmScalar<1, nppiAddC_32f_C1R>::calc, NppArithmScalar<2, nppiAddC_32fc_C1R>::calc};

    CV_Assert(src.type() == CV_32FC1 || src.type() == CV_32FC2);

    callers[src.channels()](src, sc, dst);
}

void cv::gpu::subtract(const GpuMat& src, const Scalar& sc, GpuMat& dst)
{
    typedef void (*caller_t)(const GpuMat& src, const Scalar& sc, GpuMat& dst);
    static const caller_t callers[] = {0, NppArithmScalar<1, nppiSubC_32f_C1R>::calc, NppArithmScalar<2, nppiSubC_32fc_C1R>::calc};

    CV_Assert(src.type() == CV_32FC1 || src.type() == CV_32FC2);

    callers[src.channels()](src, sc, dst);
}

void cv::gpu::multiply(const GpuMat& src, const Scalar& sc, GpuMat& dst)
{
    typedef void (*caller_t)(const GpuMat& src, const Scalar& sc, GpuMat& dst);
    static const caller_t callers[] = {0, NppArithmScalar<1, nppiMulC_32f_C1R>::calc, NppArithmScalar<2, nppiMulC_32fc_C1R>::calc};

    CV_Assert(src.type() == CV_32FC1 || src.type() == CV_32FC2);

    callers[src.channels()](src, sc, dst);
}

void cv::gpu::divide(const GpuMat& src, const Scalar& sc, GpuMat& dst)
{
    typedef void (*caller_t)(const GpuMat& src, const Scalar& sc, GpuMat& dst);
    static const caller_t callers[] = {0, NppArithmScalar<1, nppiDivC_32f_C1R>::calc, NppArithmScalar<2, nppiDivC_32fc_C1R>::calc};

    CV_Assert(src.type() == CV_32FC1 || src.type() == CV_32FC2);

    callers[src.channels()](src, sc, dst);
}

////////////////////////////////////////////////////////////////////////
// transpose

void cv::gpu::transpose(const GpuMat& src, GpuMat& dst)
{
    CV_Assert(src.type() == CV_8UC1);

    dst.create( src.cols, src.rows, src.type() );

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiTranspose_8u_C1R(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz) );
}

////////////////////////////////////////////////////////////////////////
// absdiff

void cv::gpu::absdiff(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)
{
    CV_DbgAssert(src1.size() == src2.size() && src1.type() == src2.type());

    CV_Assert(src1.type() == CV_8UC1 || src1.type() == CV_8UC4 || src1.type() == CV_32SC1 || src1.type() == CV_32FC1);

    dst.create( src1.size(), src1.type() );

    NppiSize sz;
    sz.width  = src1.cols;
    sz.height = src1.rows;

    switch (src1.type())
    {
    case CV_8UC1:
        nppSafeCall( nppiAbsDiff_8u_C1R(src1.ptr<Npp8u>(), src1.step,
            src2.ptr<Npp8u>(), src2.step,
            dst.ptr<Npp8u>(), dst.step, sz) );
        break;
    case CV_8UC4:
        nppSafeCall( nppiAbsDiff_8u_C4R(src1.ptr<Npp8u>(), src1.step,
            src2.ptr<Npp8u>(), src2.step,
            dst.ptr<Npp8u>(), dst.step, sz) );
        break;
    case CV_32SC1:
        nppSafeCall( nppiAbsDiff_32s_C1R(src1.ptr<Npp32s>(), src1.step,
            src2.ptr<Npp32s>(), src2.step,
            dst.ptr<Npp32s>(), dst.step, sz) );
        break;
    case CV_32FC1:
        nppSafeCall( nppiAbsDiff_32f_C1R(src1.ptr<Npp32f>(), src1.step,
            src2.ptr<Npp32f>(), src2.step,
            dst.ptr<Npp32f>(), dst.step, sz) );
        break;
    default:
        CV_Assert(!"Unsupported source type");
    }
}

void cv::gpu::absdiff(const GpuMat& src, const Scalar& s, GpuMat& dst)
{
    CV_Assert(src.type() == CV_32FC1);

    dst.create( src.size(), src.type() );

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiAbsDiffC_32f_C1R(src.ptr<Npp32f>(), src.step, dst.ptr<Npp32f>(), dst.step, sz, (Npp32f)s[0]) );
}

////////////////////////////////////////////////////////////////////////
// compare

namespace cv { namespace gpu { namespace mathfunc
{
    void compare_ne_8uc4(const DevMem2D& src1, const DevMem2D& src2, const DevMem2D& dst);
    void compare_ne_32f(const DevMem2D& src1, const DevMem2D& src2, const DevMem2D& dst);
}}}

void cv::gpu::compare(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, int cmpop)
{
    CV_DbgAssert(src1.size() == src2.size() && src1.type() == src2.type());

    CV_Assert(src1.type() == CV_8UC4 || src1.type() == CV_32FC1);

    dst.create( src1.size(), CV_8UC1 );

    static const NppCmpOp nppCmpOp[] = { NPP_CMP_EQ, NPP_CMP_GREATER, NPP_CMP_GREATER_EQ, NPP_CMP_LESS, NPP_CMP_LESS_EQ };

    NppiSize sz;
    sz.width  = src1.cols;
    sz.height = src1.rows;

    if (src1.type() == CV_8UC4)
    {
        if (cmpop != CMP_NE)
        {
            nppSafeCall( nppiCompare_8u_C4R(src1.ptr<Npp8u>(), src1.step,
                src2.ptr<Npp8u>(), src2.step,
                dst.ptr<Npp8u>(), dst.step, sz, nppCmpOp[cmpop]) );
        }
        else
        {
            mathfunc::compare_ne_8uc4(src1, src2, dst);
        }
    }
    else
    {
        if (cmpop != CMP_NE)
        {
            nppSafeCall( nppiCompare_32f_C1R(src1.ptr<Npp32f>(), src1.step,
                src2.ptr<Npp32f>(), src2.step,
                dst.ptr<Npp8u>(), dst.step, sz, nppCmpOp[cmpop]) );
        }
        else
        {
            mathfunc::compare_ne_32f(src1, src2, dst);
        }
    }
}

////////////////////////////////////////////////////////////////////////
// meanStdDev

void cv::gpu::meanStdDev(const GpuMat& src, Scalar& mean, Scalar& stddev)
{
    CV_Assert(src.type() == CV_8UC1);

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiMean_StdDev_8u_C1R(src.ptr<Npp8u>(), src.step, sz, mean.val, stddev.val) );
}

////////////////////////////////////////////////////////////////////////
// norm

double cv::gpu::norm(const GpuMat& src1, int normType)
{
    return norm(src1, GpuMat(src1.size(), src1.type(), Scalar::all(0.0)), normType);
}

double cv::gpu::norm(const GpuMat& src1, const GpuMat& src2, int normType)
{
    CV_DbgAssert(src1.size() == src2.size() && src1.type() == src2.type());

    CV_Assert(src1.type() == CV_8UC1);
    CV_Assert(normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2);

    typedef NppStatus (*npp_norm_diff_func_t)(const Npp8u* pSrc1, int nSrcStep1, const Npp8u* pSrc2, int nSrcStep2,
        NppiSize oSizeROI, Npp64f* pRetVal);

    static const npp_norm_diff_func_t npp_norm_diff_func[] = {nppiNormDiff_Inf_8u_C1R, nppiNormDiff_L1_8u_C1R, nppiNormDiff_L2_8u_C1R};

    NppiSize sz;
    sz.width  = src1.cols;
    sz.height = src1.rows;

    int funcIdx = normType >> 1;
    double retVal;

    nppSafeCall( npp_norm_diff_func[funcIdx](src1.ptr<Npp8u>(), src1.step,
        src2.ptr<Npp8u>(), src2.step,
        sz, &retVal) );

    return retVal;
}

////////////////////////////////////////////////////////////////////////
// flip

void cv::gpu::flip(const GpuMat& src, GpuMat& dst, int flipCode)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);

    dst.create( src.size(), src.type() );

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    if (src.type() == CV_8UC1)
    {
        nppSafeCall( nppiMirror_8u_C1R(src.ptr<Npp8u>(), src.step,
            dst.ptr<Npp8u>(), dst.step, sz,
            (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS))) );
    }
    else
    {
        nppSafeCall( nppiMirror_8u_C4R(src.ptr<Npp8u>(), src.step,
            dst.ptr<Npp8u>(), dst.step, sz,
            (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS))) );
    }
}

////////////////////////////////////////////////////////////////////////
// sum

Scalar cv::gpu::sum(const GpuMat& src)
{
    CV_Assert(!"disabled until fix crash");

    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC4);

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    Scalar res;

    int bufsz;

    if (src.type() == CV_8UC1)
    {
        nppiReductionGetBufferHostSize_8u_C1R(sz, &bufsz);
        GpuMat buf(1, bufsz, CV_32S);

        nppSafeCall( nppiSum_8u_C1R(src.ptr<Npp8u>(), src.step, sz, buf.ptr<Npp32s>(), res.val) );
    }
    else
    {
        nppiReductionGetBufferHostSize_8u_C4R(sz, &bufsz);
        GpuMat buf(1, bufsz, CV_32S);

        nppSafeCall( nppiSum_8u_C4R(src.ptr<Npp8u>(), src.step, sz, buf.ptr<Npp32s>(), res.val) );
    }

    return res;
}

////////////////////////////////////////////////////////////////////////
// minMax

namespace cv { namespace gpu { namespace mathfunc {
    template <typename T> 
    void min_max_caller(const DevMem2D src, double* minval, double* maxval);
}}}

void cv::gpu::minMax(const GpuMat& src, double* minVal, double* maxVal)
{
    GpuMat src_ = src.reshape(1);

    double maxVal_;
    if (!maxVal) 
        maxVal = &maxVal_;
  
    switch (src_.type())
    {
    case CV_8U:
        mathfunc::min_max_caller<unsigned char>(src_, minVal, maxVal);
        break;
    case CV_8S:
        mathfunc::min_max_caller<signed char>(src_, minVal, maxVal);
        break;
    case CV_16U:
        mathfunc::min_max_caller<unsigned short>(src_, minVal, maxVal);
        break;
    case CV_16S:
        mathfunc::min_max_caller<signed short>(src_, minVal, maxVal);
        break;
    case CV_32S:
        mathfunc::min_max_caller<int>(src_, minVal, maxVal);
        break;
    case CV_32F:
        mathfunc::min_max_caller<float>(src_, minVal, maxVal);
        break;
    case CV_64F:
        mathfunc::min_max_caller<double>(src_, minVal, maxVal);
        break;
    default:
        CV_Error(CV_StsBadArg, "Unsupported type");
    }
}


////////////////////////////////////////////////////////////////////////
// minMaxLoc

namespace cv { namespace gpu { namespace mathfunc {
    template <typename T> 
    void min_max_loc_caller(const DevMem2D src, double* minval, double* maxval, int* minlocx, int* minlocy,
                                                                                int* maxlocx, int* maxlocy);
}}}

void cv::gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc)
{
    CV_Assert(src.channels() == 1);

    double maxVal_;
    if (!maxVal) maxVal = &maxVal_;

    cv::Point minLoc_;
    if (!minLoc) minLoc = &minLoc_;

    cv::Point maxLoc_;
    if (!maxLoc) maxLoc = &maxLoc_;
  
    switch (src.type())
    {
    case CV_8U:
        mathfunc::min_max_loc_caller<unsigned char>(src, minVal, maxVal, &minLoc->x, &minLoc->y, &maxLoc->x, &maxLoc->y);
        break;
    case CV_8S:
        mathfunc::min_max_loc_caller<signed char>(src, minVal, maxVal, &minLoc->x, &minLoc->y, &maxLoc->x, &maxLoc->y);
        break;
    case CV_16U:
        mathfunc::min_max_loc_caller<unsigned short>(src, minVal, maxVal, &minLoc->x, &minLoc->y, &maxLoc->x, &maxLoc->y);
        break;
    case CV_16S:
        mathfunc::min_max_loc_caller<signed short>(src, minVal, maxVal, &minLoc->x, &minLoc->y, &maxLoc->x, &maxLoc->y);
        break;
    case CV_32S:
        mathfunc::min_max_loc_caller<int>(src, minVal, maxVal, &minLoc->x, &minLoc->y, &maxLoc->x, &maxLoc->y);
        break;
    case CV_32F:
        mathfunc::min_max_loc_caller<float>(src, minVal, maxVal, &minLoc->x, &minLoc->y, &maxLoc->x, &maxLoc->y);
        break;
    case CV_64F:
        mathfunc::min_max_loc_caller<double>(src, minVal, maxVal, &minLoc->x, &minLoc->y, &maxLoc->x, &maxLoc->y);
        break;
    default:
        CV_Error(CV_StsBadArg, "Unsupported type");
    }
}

////////////////////////////////////////////////////////////////////////
// LUT

void cv::gpu::LUT(const GpuMat& src, const Mat& lut, GpuMat& dst)
{
    class LevelsInit
    {
    public:
        Npp32s pLevels[256];
        const Npp32s* pLevels3[3];
        int nValues3[3];

        LevelsInit()
        {
            nValues3[0] = nValues3[1] = nValues3[2] = 256;
            for (int i = 0; i < 256; ++i)
                pLevels[i] = i;
            pLevels3[0] = pLevels3[1] = pLevels3[2] = pLevels;
        }
    };
    static LevelsInit lvls;

    int cn = src.channels();

    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
    CV_Assert(lut.depth() == CV_8U && (lut.channels() == 1 || lut.channels() == cn) && lut.rows * lut.cols == 256 && lut.isContinuous());

    dst.create(src.size(), CV_MAKETYPE(lut.depth(), cn));

    NppiSize sz;
    sz.height = src.rows;
    sz.width = src.cols;

    Mat nppLut;
    lut.convertTo(nppLut, CV_32S);

    if (src.type() == CV_8UC1)
    {
        nppSafeCall( nppiLUT_Linear_8u_C1R(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz,
            nppLut.ptr<Npp32s>(), lvls.pLevels, 256) );
    }
    else
    {
        Mat nppLut3[3];
        const Npp32s* pValues3[3];
        if (nppLut.channels() == 1)
            pValues3[0] = pValues3[1] = pValues3[2] = nppLut.ptr<Npp32s>();
        else
        {
            cv::split(nppLut, nppLut3);
            pValues3[0] = nppLut3[0].ptr<Npp32s>();
            pValues3[1] = nppLut3[1].ptr<Npp32s>();
            pValues3[2] = nppLut3[2].ptr<Npp32s>();
        }
        nppSafeCall( nppiLUT_Linear_8u_C3R(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz,
            pValues3, lvls.pLevels3, lvls.nValues3) );
    }
}

////////////////////////////////////////////////////////////////////////
// exp

void cv::gpu::exp(const GpuMat& src, GpuMat& dst)
{
    CV_Assert(src.type() == CV_32FC1);

    dst.create(src.size(), src.type());

    NppiSize sz;
    sz.width = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiExp_32f_C1R(src.ptr<Npp32f>(), src.step, dst.ptr<Npp32f>(), dst.step, sz) );
}

////////////////////////////////////////////////////////////////////////
// log

void cv::gpu::log(const GpuMat& src, GpuMat& dst)
{
    CV_Assert(src.type() == CV_32FC1);

    dst.create(src.size(), src.type());

    NppiSize sz;
    sz.width = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiLn_32f_C1R(src.ptr<Npp32f>(), src.step, dst.ptr<Npp32f>(), dst.step, sz) );
}

////////////////////////////////////////////////////////////////////////
// NPP magnitide

namespace
{
    typedef NppStatus (*nppMagnitude_t)(const Npp32fc* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep, NppiSize oSizeROI);

    inline void npp_magnitude(const GpuMat& src, GpuMat& dst, nppMagnitude_t func)
    {
        CV_Assert(src.type() == CV_32FC2);

        dst.create(src.size(), CV_32FC1);

        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        nppSafeCall( func(src.ptr<Npp32fc>(), src.step, dst.ptr<Npp32f>(), dst.step, sz) );
    }
}

void cv::gpu::magnitude(const GpuMat& src, GpuMat& dst)
{
    ::npp_magnitude(src, dst, nppiMagnitude_32fc32f_C1R);
}

void cv::gpu::magnitudeSqr(const GpuMat& src, GpuMat& dst)
{
    ::npp_magnitude(src, dst, nppiMagnitudeSqr_32fc32f_C1R);
}

////////////////////////////////////////////////////////////////////////
// Polar <-> Cart

namespace cv { namespace gpu { namespace mathfunc
{
    void cartToPolar_gpu(const DevMem2Df& x, const DevMem2Df& y, const DevMem2Df& mag, bool magSqr, const DevMem2Df& angle, bool angleInDegrees, cudaStream_t stream);
    void polarToCart_gpu(const DevMem2Df& mag, const DevMem2Df& angle, const DevMem2Df& x, const DevMem2Df& y, bool angleInDegrees, cudaStream_t stream);
}}}

namespace
{
    inline void cartToPolar_caller(const GpuMat& x, const GpuMat& y, GpuMat* mag, bool magSqr, GpuMat* angle, bool angleInDegrees, cudaStream_t stream)
    {
        CV_DbgAssert(x.size() == y.size() && x.type() == y.type());
        CV_Assert(x.depth() == CV_32F);

        if (mag)
            mag->create(x.size(), x.type());
        if (angle)
            angle->create(x.size(), x.type());

        GpuMat x1cn = x.reshape(1);
        GpuMat y1cn = y.reshape(1);
        GpuMat mag1cn = mag ? mag->reshape(1) : GpuMat();
        GpuMat angle1cn = angle ? angle->reshape(1) : GpuMat();

        mathfunc::cartToPolar_gpu(x1cn, y1cn, mag1cn, magSqr, angle1cn, angleInDegrees, stream);
    }

    inline void polarToCart_caller(const GpuMat& mag, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, cudaStream_t stream)
    {
        CV_DbgAssert((mag.empty() || mag.size() == angle.size()) && mag.type() == angle.type());
        CV_Assert(mag.depth() == CV_32F);

        x.create(mag.size(), mag.type());
        y.create(mag.size(), mag.type());

        GpuMat mag1cn = mag.reshape(1);
        GpuMat angle1cn = angle.reshape(1);
        GpuMat x1cn = x.reshape(1);
        GpuMat y1cn = y.reshape(1);

        mathfunc::polarToCart_gpu(mag1cn, angle1cn, x1cn, y1cn, angleInDegrees, stream);
    }
}

void cv::gpu::magnitude(const GpuMat& x, const GpuMat& y, GpuMat& dst)
{
    ::cartToPolar_caller(x, y, &dst, false, 0, false, 0);
}

void cv::gpu::magnitude(const GpuMat& x, const GpuMat& y, GpuMat& dst, const Stream& stream)
{
    ::cartToPolar_caller(x, y, &dst, false, 0, false, StreamAccessor::getStream(stream));
}

void cv::gpu::magnitudeSqr(const GpuMat& x, const GpuMat& y, GpuMat& dst)
{
    ::cartToPolar_caller(x, y, &dst, true, 0, false, 0);
}

void cv::gpu::magnitudeSqr(const GpuMat& x, const GpuMat& y, GpuMat& dst, const Stream& stream)
{
    ::cartToPolar_caller(x, y, &dst, true, 0, false, StreamAccessor::getStream(stream));
}

void cv::gpu::phase(const GpuMat& x, const GpuMat& y, GpuMat& angle, bool angleInDegrees)
{
    ::cartToPolar_caller(x, y, 0, false, &angle, angleInDegrees, 0);
}

void cv::gpu::phase(const GpuMat& x, const GpuMat& y, GpuMat& angle, bool angleInDegrees, const Stream& stream)
{
    ::cartToPolar_caller(x, y, 0, false, &angle, angleInDegrees, StreamAccessor::getStream(stream));
}

void cv::gpu::cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& mag, GpuMat& angle, bool angleInDegrees)
{
    ::cartToPolar_caller(x, y, &mag, false, &angle, angleInDegrees, 0);
}

void cv::gpu::cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& mag, GpuMat& angle, bool angleInDegrees, const Stream& stream)
{
    ::cartToPolar_caller(x, y, &mag, false, &angle, angleInDegrees, StreamAccessor::getStream(stream));
}

void cv::gpu::polarToCart(const GpuMat& magnitude, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees)
{
    ::polarToCart_caller(magnitude, angle, x, y, angleInDegrees, 0);
}

void cv::gpu::polarToCart(const GpuMat& magnitude, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, const Stream& stream)
{
    ::polarToCart_caller(magnitude, angle, x, y, angleInDegrees, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// Per-element bit-wise logical matrix operations

namespace cv { namespace gpu { namespace mathfunc
{
    void bitwise_not_caller(int rows, int cols, const PtrStep src, int elemSize, PtrStep dst, cudaStream_t stream);
    void bitwise_not_caller(int rows, int cols, const PtrStep src, int elemSize, PtrStep dst, const PtrStep mask, cudaStream_t stream);
    void bitwise_or_caller(int rows, int cols, const PtrStep src1, const PtrStep src2, int elemSize, PtrStep dst, cudaStream_t stream);
    void bitwise_or_caller(int rows, int cols, const PtrStep src1, const PtrStep src2, int elemSize, PtrStep dst, const PtrStep mask, cudaStream_t stream);
    void bitwise_and_caller(int rows, int cols, const PtrStep src1, const PtrStep src2, int elemSize, PtrStep dst, cudaStream_t stream);
    void bitwise_and_caller(int rows, int cols, const PtrStep src1, const PtrStep src2, int elemSize, PtrStep dst, const PtrStep mask, cudaStream_t stream);
    void bitwise_xor_caller(int rows, int cols, const PtrStep src1, const PtrStep src2, int elemSize, PtrStep dst, cudaStream_t stream);
    void bitwise_xor_caller(int rows, int cols, const PtrStep src1, const PtrStep src2, int elemSize, PtrStep dst, const PtrStep mask, cudaStream_t stream);


    template <int opid, typename Mask>
    void bitwise_bin_op(int rows, int cols, const PtrStep src1, const PtrStep src2, PtrStep dst, int elem_size, Mask mask, cudaStream_t stream);
}}}

namespace
{
    void bitwise_not_caller(const GpuMat& src, GpuMat& dst, cudaStream_t stream)
    {
        dst.create(src.size(), src.type());
        mathfunc::bitwise_not_caller(src.rows, src.cols, src, src.elemSize(), dst, stream);
    }

    void bitwise_not_caller(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream)
    {
        CV_Assert(mask.type() == CV_8U && mask.size() == src.size());
        dst.create(src.size(), src.type());
        mathfunc::bitwise_not_caller(src.rows, src.cols, src, src.elemSize(), dst, mask, stream);
    }

    void bitwise_or_caller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        dst.create(src1.size(), src1.type());
        mathfunc::bitwise_or_caller(dst.rows, dst.cols, src1, src2, dst.elemSize(), dst, stream);
    }

    void bitwise_or_caller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        CV_Assert(mask.type() == CV_8U && mask.size() == src1.size());
        dst.create(src1.size(), src1.type());
        mathfunc::bitwise_or_caller(dst.rows, dst.cols, src1, src2, dst.elemSize(), dst, mask, stream);
    }

    void bitwise_and_caller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        dst.create(src1.size(), src1.type());
        mathfunc::bitwise_and_caller(dst.rows, dst.cols, src1, src2, dst.elemSize(), dst, stream);
    }

    void bitwise_and_caller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        CV_Assert(mask.type() == CV_8U && mask.size() == src1.size());
        dst.create(src1.size(), src1.type());
        mathfunc::bitwise_and_caller(dst.rows, dst.cols, src1, src2, dst.elemSize(), dst, mask, stream);
    }

    void bitwise_xor_caller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size());
        CV_Assert(src1.type() == src2.type());
        dst.create(src1.size(), src1.type());
        mathfunc::bitwise_xor_caller(dst.rows, dst.cols, src1, src2, dst.elemSize(), dst, stream);
    }

    void bitwise_xor_caller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        CV_Assert(mask.type() == CV_8U && mask.size() == src1.size());
        dst.create(src1.size(), src1.type());
        mathfunc::bitwise_xor_caller(dst.rows, dst.cols, src1, src2, dst.elemSize(), dst, mask, stream);
    }
}

void cv::gpu::bitwise_not(const GpuMat& src, GpuMat& dst, const GpuMat& mask)
{
    if (mask.empty())
        ::bitwise_not_caller(src, dst, 0);
    else
        ::bitwise_not_caller(src, dst, mask, 0);
}

void cv::gpu::bitwise_not(const GpuMat& src, GpuMat& dst, const GpuMat& mask, const Stream& stream)
{
    if (mask.empty())
        ::bitwise_not_caller(src, dst, StreamAccessor::getStream(stream));
    else
        ::bitwise_not_caller(src, dst, mask, StreamAccessor::getStream(stream));
}

void cv::gpu::bitwise_or(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask)
{
    if (mask.empty())
        ::bitwise_or_caller(src1, src2, dst, 0);
    else
        ::bitwise_or_caller(src1, src2, dst, mask, 0);
}

void cv::gpu::bitwise_or(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, const Stream& stream)
{
    if (mask.empty())
        ::bitwise_or_caller(src1, src2, dst, StreamAccessor::getStream(stream));
    else
        ::bitwise_or_caller(src1, src2, dst, mask, StreamAccessor::getStream(stream));
}

void cv::gpu::bitwise_and(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask)
{
    if (mask.empty())
        ::bitwise_and_caller(src1, src2, dst, 0);
    else
        ::bitwise_and_caller(src1, src2, dst, mask, 0);
}

void cv::gpu::bitwise_and(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, const Stream& stream)
{
    if (mask.empty())
        ::bitwise_and_caller(src1, src2, dst, StreamAccessor::getStream(stream));
    else
        ::bitwise_and_caller(src1, src2, dst, mask, StreamAccessor::getStream(stream));
}

void cv::gpu::bitwise_xor(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask)
{
    if (mask.empty())
        ::bitwise_xor_caller(src1, src2, dst, 0);
    else
        ::bitwise_xor_caller(src1, src2, dst, mask, 0);
}

void cv::gpu::bitwise_xor(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, const Stream& stream)
{
    if (mask.empty())
        ::bitwise_xor_caller(src1, src2, dst, StreamAccessor::getStream(stream));
    else
        ::bitwise_xor_caller(src1, src2, dst, mask, StreamAccessor::getStream(stream));

}

cv::gpu::GpuMat cv::gpu::operator ~ (const GpuMat& src)
{
    GpuMat dst;
    bitwise_not(src, dst);
    return dst;
}

cv::gpu::GpuMat cv::gpu::operator | (const GpuMat& src1, const GpuMat& src2)
{
    GpuMat dst;
    bitwise_or(src1, src2, dst);
    return dst;
}

cv::gpu::GpuMat cv::gpu::operator & (const GpuMat& src1, const GpuMat& src2)
{
    GpuMat dst;
    bitwise_and(src1, src2, dst);
    return dst;
}

cv::gpu::GpuMat cv::gpu::operator ^ (const GpuMat& src1, const GpuMat& src2)
{
    GpuMat dst;
    bitwise_xor(src1, src2, dst);
    return dst;
}


#endif /* !defined (HAVE_CUDA) */
