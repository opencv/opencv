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

#if !defined (HAVE_CUDA)

void cv::gpu::meanStdDev(const GpuMat&, Scalar&, Scalar&) { throw_nogpu(); }
double cv::gpu::norm(const GpuMat&, int) { throw_nogpu(); return 0.0; }
double cv::gpu::norm(const GpuMat&, const GpuMat&, int) { throw_nogpu(); return 0.0; }
Scalar cv::gpu::sum(const GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::sum(const GpuMat&, GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::absSum(const GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::absSum(const GpuMat&, GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::sqrSum(const GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::sqrSum(const GpuMat&, GpuMat&) { throw_nogpu(); return Scalar(); }
void cv::gpu::minMax(const GpuMat&, double*, double*, const GpuMat&) { throw_nogpu(); }
void cv::gpu::minMax(const GpuMat&, double*, double*, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::minMaxLoc(const GpuMat&, double*, double*, Point*, Point*, const GpuMat&) { throw_nogpu(); }
void cv::gpu::minMaxLoc(const GpuMat&, double*, double*, Point*, Point*, const GpuMat&, GpuMat&, GpuMat&) { throw_nogpu(); }
int cv::gpu::countNonZero(const GpuMat&) { throw_nogpu(); return 0; }
int cv::gpu::countNonZero(const GpuMat&, GpuMat&) { throw_nogpu(); return 0; }

#else


////////////////////////////////////////////////////////////////////////
// meanStdDev

void cv::gpu::meanStdDev(const GpuMat& src, Scalar& mean, Scalar& stddev)
{
    CV_Assert(src.type() == CV_8UC1);

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    nppSafeCall( nppiMean_StdDev_8u_C1R(src.ptr<Npp8u>(), src.step, sz, mean.val, stddev.val) );

    cudaSafeCall( cudaThreadSynchronize() );
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

    cudaSafeCall( cudaThreadSynchronize() );

    return retVal;
}

////////////////////////////////////////////////////////////////////////
// Sum

namespace cv { namespace gpu { namespace mathfunc
{
    template <typename T>
    void sumCaller(const DevMem2D src, PtrStep buf, double* sum, int cn);

    template <typename T>
    void sumMultipassCaller(const DevMem2D src, PtrStep buf, double* sum, int cn);

    template <typename T>
    void absSumCaller(const DevMem2D src, PtrStep buf, double* sum, int cn);

    template <typename T>
    void absSumMultipassCaller(const DevMem2D src, PtrStep buf, double* sum, int cn);

    template <typename T>
    void sqrSumCaller(const DevMem2D src, PtrStep buf, double* sum, int cn);

    template <typename T>
    void sqrSumMultipassCaller(const DevMem2D src, PtrStep buf, double* sum, int cn);

    namespace sums
    {
        void getBufSizeRequired(int cols, int rows, int cn, int& bufcols, int& bufrows);
    }
}}}


Scalar cv::gpu::sum(const GpuMat& src) 
{
    GpuMat buf;
    return sum(src, buf);
}


Scalar cv::gpu::sum(const GpuMat& src, GpuMat& buf) 
{
    using namespace mathfunc;

    typedef void (*Caller)(const DevMem2D, PtrStep, double*, int);

    static Caller multipass_callers[7] = { 
            sumMultipassCaller<unsigned char>, sumMultipassCaller<char>, 
            sumMultipassCaller<unsigned short>, sumMultipassCaller<short>, 
            sumMultipassCaller<int>, sumMultipassCaller<float>, 0 };

    static Caller singlepass_callers[7] = { 
            sumCaller<unsigned char>, sumCaller<char>, 
            sumCaller<unsigned short>, sumCaller<short>, 
            sumCaller<int>, sumCaller<float>, 0 };

    Size buf_size;
    sums::getBufSizeRequired(src.cols, src.rows, src.channels(), 
                             buf_size.width, buf_size.height); 
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    Caller* callers = multipass_callers;
    if (TargetArchs::builtWith(ATOMICS) && DeviceInfo().has(ATOMICS))
        callers = singlepass_callers;

    Caller caller = callers[src.depth()];
    if (!caller) CV_Error(CV_StsBadArg, "sum: unsupported type");

    double result[4];
    caller(src, buf, result, src.channels());
    return Scalar(result[0], result[1], result[2], result[3]);
}


Scalar cv::gpu::absSum(const GpuMat& src) 
{
    GpuMat buf;
    return absSum(src, buf);
}


Scalar cv::gpu::absSum(const GpuMat& src, GpuMat& buf) 
{
    using namespace mathfunc;

    typedef void (*Caller)(const DevMem2D, PtrStep, double*, int);

    static Caller multipass_callers[7] = { 
            absSumMultipassCaller<unsigned char>, absSumMultipassCaller<char>, 
            absSumMultipassCaller<unsigned short>, absSumMultipassCaller<short>, 
            absSumMultipassCaller<int>, absSumMultipassCaller<float>, 0 };

    static Caller singlepass_callers[7] = { 
            absSumCaller<unsigned char>, absSumCaller<char>, 
            absSumCaller<unsigned short>, absSumCaller<short>, 
            absSumCaller<int>, absSumCaller<float>, 0 };

    Size buf_size;
    sums::getBufSizeRequired(src.cols, src.rows, src.channels(), 
                             buf_size.width, buf_size.height); 
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    Caller* callers = multipass_callers;
    if (TargetArchs::builtWith(ATOMICS) && DeviceInfo().has(ATOMICS))
        callers = singlepass_callers;

    Caller caller = callers[src.depth()];
    if (!caller) CV_Error(CV_StsBadArg, "absSum: unsupported type");

    double result[4];
    caller(src, buf, result, src.channels());
    return Scalar(result[0], result[1], result[2], result[3]);
}


Scalar cv::gpu::sqrSum(const GpuMat& src) 
{
    GpuMat buf;
    return sqrSum(src, buf);
}


Scalar cv::gpu::sqrSum(const GpuMat& src, GpuMat& buf) 
{
    using namespace mathfunc;

    typedef void (*Caller)(const DevMem2D, PtrStep, double*, int);

    static Caller multipass_callers[7] = { 
            sqrSumMultipassCaller<unsigned char>, sqrSumMultipassCaller<char>, 
            sqrSumMultipassCaller<unsigned short>, sqrSumMultipassCaller<short>, 
            sqrSumMultipassCaller<int>, sqrSumMultipassCaller<float>, 0 };

    static Caller singlepass_callers[7] = { 
            sqrSumCaller<unsigned char>, sqrSumCaller<char>, 
            sqrSumCaller<unsigned short>, sqrSumCaller<short>, 
            sqrSumCaller<int>, sqrSumCaller<float>, 0 };

    Caller* callers = multipass_callers;
    if (TargetArchs::builtWith(ATOMICS) && DeviceInfo().has(ATOMICS))
        callers = singlepass_callers;

    Size buf_size;
    sums::getBufSizeRequired(src.cols, src.rows, src.channels(), 
                             buf_size.width, buf_size.height); 
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    Caller caller = callers[src.depth()];
    if (!caller) CV_Error(CV_StsBadArg, "sqrSum: unsupported type");

    double result[4];
    caller(src, buf, result, src.channels());
    return Scalar(result[0], result[1], result[2], result[3]);
}




////////////////////////////////////////////////////////////////////////
// Find min or max

namespace cv { namespace gpu { namespace mathfunc { namespace minmax {

    void getBufSizeRequired(int cols, int rows, int elem_size, int& bufcols, int& bufrows);
    
    template <typename T> 
    void minMaxCaller(const DevMem2D src, double* minval, double* maxval, PtrStep buf);

    template <typename T> 
    void minMaxMaskCaller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, PtrStep buf);

    template <typename T> 
    void minMaxMultipassCaller(const DevMem2D src, double* minval, double* maxval, PtrStep buf);

    template <typename T> 
    void minMaxMaskMultipassCaller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, PtrStep buf);

}}}}


void cv::gpu::minMax(const GpuMat& src, double* minVal, double* maxVal, const GpuMat& mask)
{
    GpuMat buf;
    minMax(src, minVal, maxVal, mask, buf);
}


void cv::gpu::minMax(const GpuMat& src, double* minVal, double* maxVal, const GpuMat& mask, GpuMat& buf)
{
    using namespace mathfunc::minmax;

    typedef void (*Caller)(const DevMem2D, double*, double*, PtrStep);
    typedef void (*MaskedCaller)(const DevMem2D, const PtrStep, double*, double*, PtrStep);

    static Caller multipass_callers[7] = { 
            minMaxMultipassCaller<unsigned char>, minMaxMultipassCaller<char>, 
            minMaxMultipassCaller<unsigned short>, minMaxMultipassCaller<short>, 
            minMaxMultipassCaller<int>, minMaxMultipassCaller<float>, 0 };

    static Caller singlepass_callers[7] = { 
            minMaxCaller<unsigned char>, minMaxCaller<char>, 
            minMaxCaller<unsigned short>, minMaxCaller<short>, 
            minMaxCaller<int>, minMaxCaller<float>, minMaxCaller<double> };

    static MaskedCaller masked_multipass_callers[7] = { 
            minMaxMaskMultipassCaller<unsigned char>, minMaxMaskMultipassCaller<char>, 
            minMaxMaskMultipassCaller<unsigned short>, minMaxMaskMultipassCaller<short>,
            minMaxMaskMultipassCaller<int>, minMaxMaskMultipassCaller<float>, 0 };

    static MaskedCaller masked_singlepass_callers[7] = { 
            minMaxMaskCaller<unsigned char>, minMaxMaskCaller<char>, 
            minMaxMaskCaller<unsigned short>, minMaxMaskCaller<short>, 
            minMaxMaskCaller<int>, minMaxMaskCaller<float>, 
            minMaxMaskCaller<double> };

    CV_Assert(src.channels() == 1);

    CV_Assert(mask.empty() || (mask.type() == CV_8U && src.size() == mask.size()));

    CV_Assert(src.type() != CV_64F || (TargetArchs::builtWith(NATIVE_DOUBLE) && 
                                       DeviceInfo().has(NATIVE_DOUBLE)));

    double minVal_; if (!minVal) minVal = &minVal_;
    double maxVal_; if (!maxVal) maxVal = &maxVal_;
    
    Size buf_size;
    getBufSizeRequired(src.cols, src.rows, src.elemSize(), buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    if (mask.empty())
    {
        Caller* callers = multipass_callers;
        if (TargetArchs::builtWith(ATOMICS) && DeviceInfo().has(ATOMICS))
            callers = singlepass_callers;

        Caller caller = callers[src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMax: unsupported type");
        caller(src, minVal, maxVal, buf);
    }
    else
    {
        MaskedCaller* callers = masked_multipass_callers;
        if (TargetArchs::builtWith(ATOMICS) && DeviceInfo().has(ATOMICS))
            callers = masked_singlepass_callers;

        MaskedCaller caller = callers[src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMax: unsupported type");
        caller(src, mask, minVal, maxVal, buf);
    }
}


////////////////////////////////////////////////////////////////////////
// Locate min and max

namespace cv { namespace gpu { namespace mathfunc { namespace minmaxloc {

    void getBufSizeRequired(int cols, int rows, int elem_size, int& b1cols, 
                               int& b1rows, int& b2cols, int& b2rows);

    template <typename T> 
    void minMaxLocCaller(const DevMem2D src, double* minval, double* maxval, 
                            int minloc[2], int maxloc[2], PtrStep valBuf, PtrStep locBuf);

    template <typename T> 
    void minMaxLocMaskCaller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, 
                                 int minloc[2], int maxloc[2], PtrStep valBuf, PtrStep locBuf);

    template <typename T> 
    void minMaxLocMultipassCaller(const DevMem2D src, double* minval, double* maxval, 
                                     int minloc[2], int maxloc[2], PtrStep valBuf, PtrStep locBuf);

    template <typename T> 
    void minMaxLocMaskMultipassCaller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, 
                                           int minloc[2], int maxloc[2], PtrStep valBuf, PtrStep locBuf);
}}}}


void cv::gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc, const GpuMat& mask)
{    
    GpuMat valBuf, locBuf;
    minMaxLoc(src, minVal, maxVal, minLoc, maxLoc, mask, valBuf, locBuf);
}


void cv::gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc,
                        const GpuMat& mask, GpuMat& valBuf, GpuMat& locBuf)
{
    using namespace mathfunc::minmaxloc;

    typedef void (*Caller)(const DevMem2D, double*, double*, int[2], int[2], PtrStep, PtrStep);
    typedef void (*MaskedCaller)(const DevMem2D, const PtrStep, double*, double*, int[2], int[2], PtrStep, PtrStep);

    static Caller multipass_callers[7] = { 
            minMaxLocMultipassCaller<unsigned char>, minMaxLocMultipassCaller<char>, 
            minMaxLocMultipassCaller<unsigned short>, minMaxLocMultipassCaller<short>, 
            minMaxLocMultipassCaller<int>, minMaxLocMultipassCaller<float>, 0 };

    static Caller singlepass_callers[7] = { 
            minMaxLocCaller<unsigned char>, minMaxLocCaller<char>, 
            minMaxLocCaller<unsigned short>, minMaxLocCaller<short>, 
            minMaxLocCaller<int>, minMaxLocCaller<float>, minMaxLocCaller<double> };

    static MaskedCaller masked_multipass_callers[7] = { 
            minMaxLocMaskMultipassCaller<unsigned char>, minMaxLocMaskMultipassCaller<char>, 
            minMaxLocMaskMultipassCaller<unsigned short>, minMaxLocMaskMultipassCaller<short>, 
            minMaxLocMaskMultipassCaller<int>, minMaxLocMaskMultipassCaller<float>, 0 };

    static MaskedCaller masked_singlepass_callers[7] = { 
            minMaxLocMaskCaller<unsigned char>, minMaxLocMaskCaller<char>, 
            minMaxLocMaskCaller<unsigned short>, minMaxLocMaskCaller<short>, 
            minMaxLocMaskCaller<int>, minMaxLocMaskCaller<float>, 
            minMaxLocMaskCaller<double> };

    CV_Assert(src.channels() == 1);

    CV_Assert(mask.empty() || (mask.type() == CV_8U && src.size() == mask.size()));

    CV_Assert(src.type() != CV_64F || (TargetArchs::builtWith(NATIVE_DOUBLE) && 
                                       DeviceInfo().has(NATIVE_DOUBLE)));

    double minVal_; if (!minVal) minVal = &minVal_;
    double maxVal_; if (!maxVal) maxVal = &maxVal_;
    int minLoc_[2];
    int maxLoc_[2];

    Size valbuf_size, locbuf_size;
    getBufSizeRequired(src.cols, src.rows, src.elemSize(), valbuf_size.width, 
                       valbuf_size.height, locbuf_size.width, locbuf_size.height);
    ensureSizeIsEnough(valbuf_size, CV_8U, valBuf);
    ensureSizeIsEnough(locbuf_size, CV_8U, locBuf);

    if (mask.empty())
    {
        Caller* callers = multipass_callers;
        if (TargetArchs::builtWith(ATOMICS) && DeviceInfo().has(ATOMICS))
            callers = singlepass_callers;

        Caller caller = callers[src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMaxLoc: unsupported type");
        caller(src, minVal, maxVal, minLoc_, maxLoc_, valBuf, locBuf);
    }
    else
    {
        MaskedCaller* callers = masked_multipass_callers;
        if (TargetArchs::builtWith(ATOMICS) && DeviceInfo().has(ATOMICS))
            callers = masked_singlepass_callers;

        MaskedCaller caller = callers[src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMaxLoc: unsupported type");
        caller(src, mask, minVal, maxVal, minLoc_, maxLoc_, valBuf, locBuf);
    }

    if (minLoc) { minLoc->x = minLoc_[0]; minLoc->y = minLoc_[1]; }
    if (maxLoc) { maxLoc->x = maxLoc_[0]; maxLoc->y = maxLoc_[1]; }
}

//////////////////////////////////////////////////////////////////////////////
// Count non-zero elements

namespace cv { namespace gpu { namespace mathfunc { namespace countnonzero {

    void getBufSizeRequired(int cols, int rows, int& bufcols, int& bufrows);

    template <typename T> 
    int countNonZeroCaller(const DevMem2D src, PtrStep buf);

    template <typename T> 
    int countNonZeroMultipassCaller(const DevMem2D src, PtrStep buf);

}}}}


int cv::gpu::countNonZero(const GpuMat& src)
{
    GpuMat buf;
    return countNonZero(src, buf);
}


int cv::gpu::countNonZero(const GpuMat& src, GpuMat& buf)
{
    using namespace mathfunc::countnonzero;

    typedef int (*Caller)(const DevMem2D src, PtrStep buf);

    static Caller multipass_callers[7] = { 
            countNonZeroMultipassCaller<unsigned char>, countNonZeroMultipassCaller<char>,
            countNonZeroMultipassCaller<unsigned short>, countNonZeroMultipassCaller<short>,
            countNonZeroMultipassCaller<int>, countNonZeroMultipassCaller<float>, 0 };

    static Caller singlepass_callers[7] = { 
            countNonZeroCaller<unsigned char>, countNonZeroCaller<char>,
            countNonZeroCaller<unsigned short>, countNonZeroCaller<short>,
            countNonZeroCaller<int>, countNonZeroCaller<float>, 
            countNonZeroCaller<double> };

    CV_Assert(src.channels() == 1);

    CV_Assert(src.type() != CV_64F || (TargetArchs::builtWith(NATIVE_DOUBLE) && 
                                       DeviceInfo().has(NATIVE_DOUBLE)));

    Size buf_size;
    getBufSizeRequired(src.cols, src.rows, buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    Caller* callers = multipass_callers;
    if (TargetArchs::builtWith(ATOMICS) && DeviceInfo().has(ATOMICS))
        callers = singlepass_callers;

    Caller caller = callers[src.type()];
    if (!caller) CV_Error(CV_StsBadArg, "countNonZero: unsupported type");
    return caller(src, buf);
}

#endif
