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
// Sum

namespace cv { namespace gpu { namespace mathfunc
{
    template <typename T>
    void sum_caller(const DevMem2D src, PtrStep buf, double* sum, int cn);

    template <typename T>
    void sum_multipass_caller(const DevMem2D src, PtrStep buf, double* sum, int cn);

    template <typename T>
    void sqsum_caller(const DevMem2D src, PtrStep buf, double* sum, int cn);

    template <typename T>
    void sqsum_multipass_caller(const DevMem2D src, PtrStep buf, double* sum, int cn);

    namespace sum
    {
        void get_buf_size_required(int cols, int rows, int cn, int& bufcols, int& bufrows);
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
    static const Caller callers[2][7] = 
        { { sum_multipass_caller<unsigned char>, sum_multipass_caller<char>, 
            sum_multipass_caller<unsigned short>, sum_multipass_caller<short>, 
            sum_multipass_caller<int>, sum_multipass_caller<float>, 0 },
          { sum_caller<unsigned char>, sum_caller<char>, 
            sum_caller<unsigned short>, sum_caller<short>, 
            sum_caller<int>, sum_caller<float>, 0 } };

    Size bufSize;
    sum::get_buf_size_required(src.cols, src.rows, src.channels(), bufSize.width, bufSize.height); 
    ensureSizeIsEnough(bufSize, CV_8U, buf);

    Caller caller = callers[hasAtomicsSupport(getDevice())][src.depth()];
    if (!caller) CV_Error(CV_StsBadArg, "sum: unsupported type");

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
    static const Caller callers[2][7] = 
        { { sqsum_multipass_caller<unsigned char>, sqsum_multipass_caller<char>, 
            sqsum_multipass_caller<unsigned short>, sqsum_multipass_caller<short>, 
            sqsum_multipass_caller<int>, sqsum_multipass_caller<float>, 0 },
          { sqsum_caller<unsigned char>, sqsum_caller<char>, 
            sqsum_caller<unsigned short>, sqsum_caller<short>, 
            sqsum_caller<int>, sqsum_caller<float>, 0 } };

    Size bufSize;
    sum::get_buf_size_required(src.cols, src.rows, src.channels(), bufSize.width, bufSize.height); 
    ensureSizeIsEnough(bufSize, CV_8U, buf);

    Caller caller = callers[hasAtomicsSupport(getDevice())][src.depth()];
    if (!caller) CV_Error(CV_StsBadArg, "sqrSum: unsupported type");

    double result[4];
    caller(src, buf, result, src.channels());
    return Scalar(result[0], result[1], result[2], result[3]);
}

////////////////////////////////////////////////////////////////////////
// Find min or max

namespace cv { namespace gpu { namespace mathfunc { namespace minmax {

    void get_buf_size_required(int cols, int rows, int elem_size, int& bufcols, int& bufrows);
    
    template <typename T> 
    void min_max_caller(const DevMem2D src, double* minval, double* maxval, PtrStep buf);

    template <typename T> 
    void min_max_mask_caller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, PtrStep buf);

    template <typename T> 
    void min_max_multipass_caller(const DevMem2D src, double* minval, double* maxval, PtrStep buf);

    template <typename T> 
    void min_max_mask_multipass_caller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, PtrStep buf);

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

    static const Caller callers[2][7] = 
    { { min_max_multipass_caller<unsigned char>, min_max_multipass_caller<char>, 
        min_max_multipass_caller<unsigned short>, min_max_multipass_caller<short>, 
        min_max_multipass_caller<int>, min_max_multipass_caller<float>, 0 },
      { min_max_caller<unsigned char>, min_max_caller<char>, 
        min_max_caller<unsigned short>, min_max_caller<short>, 
        min_max_caller<int>, min_max_caller<float>, min_max_caller<double> } };

    static const MaskedCaller masked_callers[2][7] = 
    { { min_max_mask_multipass_caller<unsigned char>, min_max_mask_multipass_caller<char>, 
        min_max_mask_multipass_caller<unsigned short>, min_max_mask_multipass_caller<short>, 
        min_max_mask_multipass_caller<int>, min_max_mask_multipass_caller<float>, 0 },
      { min_max_mask_caller<unsigned char>, min_max_mask_caller<char>, 
        min_max_mask_caller<unsigned short>, min_max_mask_caller<short>, 
        min_max_mask_caller<int>, min_max_mask_caller<float>, 
        min_max_mask_caller<double> } };


    CV_Assert(src.channels() == 1);
    CV_Assert(mask.empty() || (mask.type() == CV_8U && src.size() == mask.size()));
    CV_Assert(src.type() != CV_64F || hasNativeDoubleSupport(getDevice()));

    double minVal_; if (!minVal) minVal = &minVal_;
    double maxVal_; if (!maxVal) maxVal = &maxVal_;
    
    Size bufSize;
    get_buf_size_required(src.cols, src.rows, src.elemSize(), bufSize.width, bufSize.height);
    ensureSizeIsEnough(bufSize, CV_8U, buf);

    if (mask.empty())
    {
        Caller caller = callers[hasAtomicsSupport(getDevice())][src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMax: unsupported type");
        caller(src, minVal, maxVal, buf);
    }
    else
    {
        MaskedCaller caller = masked_callers[hasAtomicsSupport(getDevice())][src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMax: unsupported type");
        caller(src, mask, minVal, maxVal, buf);
    }
}


////////////////////////////////////////////////////////////////////////
// Locate min and max

namespace cv { namespace gpu { namespace mathfunc { namespace minmaxloc {

    void get_buf_size_required(int cols, int rows, int elem_size, int& b1cols, 
                               int& b1rows, int& b2cols, int& b2rows);

    template <typename T> 
    void min_max_loc_caller(const DevMem2D src, double* minval, double* maxval, 
                            int minloc[2], int maxloc[2], PtrStep valBuf, PtrStep locBuf);

    template <typename T> 
    void min_max_loc_mask_caller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, 
                                 int minloc[2], int maxloc[2], PtrStep valBuf, PtrStep locBuf);

    template <typename T> 
    void min_max_loc_multipass_caller(const DevMem2D src, double* minval, double* maxval, 
                                     int minloc[2], int maxloc[2], PtrStep valBuf, PtrStep locBuf);

    template <typename T> 
    void min_max_loc_mask_multipass_caller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, 
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

    static const Caller callers[2][7] = 
    { { min_max_loc_multipass_caller<unsigned char>, min_max_loc_multipass_caller<char>, 
        min_max_loc_multipass_caller<unsigned short>, min_max_loc_multipass_caller<short>, 
        min_max_loc_multipass_caller<int>, min_max_loc_multipass_caller<float>, 0 },
      { min_max_loc_caller<unsigned char>, min_max_loc_caller<char>, 
        min_max_loc_caller<unsigned short>, min_max_loc_caller<short>, 
        min_max_loc_caller<int>, min_max_loc_caller<float>, min_max_loc_caller<double> } };

    static const MaskedCaller masked_callers[2][7] = 
    { { min_max_loc_mask_multipass_caller<unsigned char>, min_max_loc_mask_multipass_caller<char>, 
        min_max_loc_mask_multipass_caller<unsigned short>, min_max_loc_mask_multipass_caller<short>, 
        min_max_loc_mask_multipass_caller<int>, min_max_loc_mask_multipass_caller<float>, 0 },
      { min_max_loc_mask_caller<unsigned char>, min_max_loc_mask_caller<char>, 
        min_max_loc_mask_caller<unsigned short>, min_max_loc_mask_caller<short>, 
        min_max_loc_mask_caller<int>, min_max_loc_mask_caller<float>, min_max_loc_mask_caller<double> } };

    CV_Assert(src.channels() == 1);
    CV_Assert(mask.empty() || (mask.type() == CV_8U && src.size() == mask.size()));
    CV_Assert(src.type() != CV_64F || hasNativeDoubleSupport(getDevice()));

    double minVal_; if (!minVal) minVal = &minVal_;
    double maxVal_; if (!maxVal) maxVal = &maxVal_;
    int minLoc_[2];
    int maxLoc_[2];

    Size valBufSize, locBufSize;
    get_buf_size_required(src.cols, src.rows, src.elemSize(), valBufSize.width, 
                          valBufSize.height, locBufSize.width, locBufSize.height);
    ensureSizeIsEnough(valBufSize, CV_8U, valBuf);
    ensureSizeIsEnough(locBufSize, CV_8U, locBuf);

    if (mask.empty())
    {
        Caller caller = callers[hasAtomicsSupport(getDevice())][src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMaxLoc: unsupported type");
        caller(src, minVal, maxVal, minLoc_, maxLoc_, valBuf, locBuf);
    }
    else
    {
        MaskedCaller caller = masked_callers[hasAtomicsSupport(getDevice())][src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMaxLoc: unsupported type");
        caller(src, mask, minVal, maxVal, minLoc_, maxLoc_, valBuf, locBuf);
    }

    if (minLoc) { minLoc->x = minLoc_[0]; minLoc->y = minLoc_[1]; }
    if (maxLoc) { maxLoc->x = maxLoc_[0]; maxLoc->y = maxLoc_[1]; }
}

//////////////////////////////////////////////////////////////////////////////
// Count non-zero elements

namespace cv { namespace gpu { namespace mathfunc { namespace countnonzero {

    void get_buf_size_required(int cols, int rows, int& bufcols, int& bufrows);

    template <typename T> 
    int count_non_zero_caller(const DevMem2D src, PtrStep buf);

    template <typename T> 
    int count_non_zero_multipass_caller(const DevMem2D src, PtrStep buf);

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

    static const Caller callers[2][7] = 
    { { count_non_zero_multipass_caller<unsigned char>, count_non_zero_multipass_caller<char>,
        count_non_zero_multipass_caller<unsigned short>, count_non_zero_multipass_caller<short>,
        count_non_zero_multipass_caller<int>, count_non_zero_multipass_caller<float>, 0},
      { count_non_zero_caller<unsigned char>, count_non_zero_caller<char>,
        count_non_zero_caller<unsigned short>, count_non_zero_caller<short>,
        count_non_zero_caller<int>, count_non_zero_caller<float>, count_non_zero_caller<double> } };

    CV_Assert(src.channels() == 1);
    CV_Assert(src.type() != CV_64F || hasNativeDoubleSupport(getDevice()));

    Size bufSize;
    get_buf_size_required(src.cols, src.rows, bufSize.width, bufSize.height);
    ensureSizeIsEnough(bufSize, CV_8U, buf);

    Caller caller = callers[hasAtomicsSupport(getDevice())][src.type()];
    if (!caller) CV_Error(CV_StsBadArg, "countNonZero: unsupported type");
    return caller(src, buf);
}

#endif
