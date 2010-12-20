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

void cv::gpu::transpose(const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::meanStdDev(const GpuMat&, Scalar&, Scalar&) { throw_nogpu(); }
double cv::gpu::norm(const GpuMat&, int) { throw_nogpu(); return 0.0; }
double cv::gpu::norm(const GpuMat&, const GpuMat&, int) { throw_nogpu(); return 0.0; }
void cv::gpu::flip(const GpuMat&, GpuMat&, int) { throw_nogpu(); }
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
void cv::gpu::min(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::min(const GpuMat&, const GpuMat&, GpuMat&, const Stream&) { throw_nogpu(); }
void cv::gpu::min(const GpuMat&, double, GpuMat&) { throw_nogpu(); }
void cv::gpu::min(const GpuMat&, double, GpuMat&, const Stream&) { throw_nogpu(); }
void cv::gpu::max(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::max(const GpuMat&, const GpuMat&, GpuMat&, const Stream&) { throw_nogpu(); }
void cv::gpu::max(const GpuMat&, double, GpuMat&) { throw_nogpu(); }
void cv::gpu::max(const GpuMat&, double, GpuMat&, const Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// transpose

namespace cv { namespace gpu { namespace mathfunc
{
    void transpose_gpu(const DevMem2Di& src, const DevMem2Di& dst);
}}}

void cv::gpu::transpose(const GpuMat& src, GpuMat& dst)
{
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8SC1 || src.type() == CV_8UC4 || src.type() == CV_8SC4 
        || src.type() == CV_16UC2 || src.type() == CV_16SC2 || src.type() == CV_32SC1 || src.type() == CV_32FC1);

    dst.create( src.cols, src.rows, src.type() );

    if (src.type() == CV_8UC1 || src.type() == CV_8SC1)
    {
        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        nppSafeCall( nppiTranspose_8u_C1R(src.ptr<Npp8u>(), src.step, dst.ptr<Npp8u>(), dst.step, sz) );
    }
    else
    {
        mathfunc::transpose_gpu(src, dst);
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
    buf.create(bufSize, CV_8U);

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
    buf.create(bufSize, CV_8U);

    Caller caller = callers[hasAtomicsSupport(getDevice())][src.depth()];
    if (!caller) CV_Error(CV_StsBadArg, "sqrSum: unsupported type");

    double result[4];
    caller(src, buf, result, src.channels());
    return Scalar(result[0], result[1], result[2], result[3]);
}

////////////////////////////////////////////////////////////////////////
// minMax

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
    buf.create(bufSize, CV_8U);

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
// minMaxLoc

namespace cv { namespace gpu { namespace mathfunc { namespace minmaxloc {

    void get_buf_size_required(int cols, int rows, int elem_size, int& b1cols, 
                               int& b1rows, int& b2cols, int& b2rows);

    template <typename T> 
    void min_max_loc_caller(const DevMem2D src, double* minval, double* maxval, 
                            int minloc[2], int maxloc[2], PtrStep valbuf, PtrStep locbuf);

    template <typename T> 
    void min_max_loc_mask_caller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, 
                                 int minloc[2], int maxloc[2], PtrStep valbuf, PtrStep locbuf);

    template <typename T> 
    void min_max_loc_multipass_caller(const DevMem2D src, double* minval, double* maxval, 
                                     int minloc[2], int maxloc[2], PtrStep valbuf, PtrStep locbuf);

    template <typename T> 
    void min_max_loc_mask_multipass_caller(const DevMem2D src, const PtrStep mask, double* minval, double* maxval, 
                                           int minloc[2], int maxloc[2], PtrStep valbuf, PtrStep locbuf);


}}}}

void cv::gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc, const GpuMat& mask)
{    
    GpuMat valbuf, locbuf;
    minMaxLoc(src, minVal, maxVal, minLoc, maxLoc, mask, valbuf, locbuf);
}

void cv::gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc,
                        const GpuMat& mask, GpuMat& valbuf, GpuMat& locbuf)
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

    Size valbuf_size, locbuf_size;
    get_buf_size_required(src.cols, src.rows, src.elemSize(), valbuf_size.width, 
                          valbuf_size.height, locbuf_size.width, locbuf_size.height);
    valbuf.create(valbuf_size, CV_8U);
    locbuf.create(locbuf_size, CV_8U);

    if (mask.empty())
    {
        Caller caller = callers[hasAtomicsSupport(getDevice())][src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMaxLoc: unsupported type");
        caller(src, minVal, maxVal, minLoc_, maxLoc_, valbuf, locbuf);
    }
    else
    {
        MaskedCaller caller = masked_callers[hasAtomicsSupport(getDevice())][src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMaxLoc: unsupported type");
        caller(src, mask, minVal, maxVal, minLoc_, maxLoc_, valbuf, locbuf);
    }

    if (minLoc) { minLoc->x = minLoc_[0]; minLoc->y = minLoc_[1]; }
    if (maxLoc) { maxLoc->x = maxLoc_[0]; maxLoc->y = maxLoc_[1]; }
}

////////////////////////////////////////////////////////////////////////
// Count non zero

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

    Size buf_size;
    get_buf_size_required(src.cols, src.rows, buf_size.width, buf_size.height);
    buf.create(buf_size, CV_8U);

    Caller caller = callers[hasAtomicsSupport(getDevice())][src.type()];
    if (!caller) CV_Error(CV_StsBadArg, "countNonZero: unsupported type");
    return caller(src, buf);
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
// min/max

namespace cv { namespace gpu { namespace mathfunc
{
    template <typename T>
    void min_gpu(const DevMem2D_<T>& src1, const DevMem2D_<T>& src2, const DevMem2D_<T>& dst, cudaStream_t stream);

    template <typename T>
    void max_gpu(const DevMem2D_<T>& src1, const DevMem2D_<T>& src2, const DevMem2D_<T>& dst, cudaStream_t stream);

    template <typename T>
    void min_gpu(const DevMem2D_<T>& src1, double src2, const DevMem2D_<T>& dst, cudaStream_t stream);

    template <typename T>
    void max_gpu(const DevMem2D_<T>& src1, double src2, const DevMem2D_<T>& dst, cudaStream_t stream);
}}}

namespace
{
    template <typename T>
    void min_caller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        dst.create(src1.size(), src1.type());
        mathfunc::min_gpu<T>(src1.reshape(1), src2.reshape(1), dst.reshape(1), stream);
    }

    template <typename T>
    void min_caller(const GpuMat& src1, double src2, GpuMat& dst, cudaStream_t stream)
    {
        dst.create(src1.size(), src1.type());
        mathfunc::min_gpu<T>(src1.reshape(1), src2, dst.reshape(1), stream);
    }
    
    template <typename T>
    void max_caller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        dst.create(src1.size(), src1.type());
        mathfunc::max_gpu<T>(src1.reshape(1), src2.reshape(1), dst.reshape(1), stream);
    }

    template <typename T>
    void max_caller(const GpuMat& src1, double src2, GpuMat& dst, cudaStream_t stream)
    {
        dst.create(src1.size(), src1.type());
        mathfunc::max_gpu<T>(src1.reshape(1), src2, dst.reshape(1), stream);
    }
}

void cv::gpu::min(const GpuMat& src1, const GpuMat& src2, GpuMat& dst) 
{ 
    typedef void (*func_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[] = 
    {
        min_caller<uchar>, min_caller<char>, min_caller<ushort>, min_caller<short>, min_caller<int>, 
        min_caller<float>, min_caller<double>
    };
    funcs[src1.depth()](src1, src2, dst, 0);
}

void cv::gpu::min(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const Stream& stream) 
{ 
    typedef void (*func_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[] = 
    {
        min_caller<uchar>, min_caller<char>, min_caller<ushort>, min_caller<short>, min_caller<int>, 
        min_caller<float>, min_caller<double>
    };
    funcs[src1.depth()](src1, src2, dst, StreamAccessor::getStream(stream));
}

void cv::gpu::min(const GpuMat& src1, double src2, GpuMat& dst) 
{
    typedef void (*func_t)(const GpuMat& src1, double src2, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[] = 
    {
        min_caller<uchar>, min_caller<char>, min_caller<ushort>, min_caller<short>, min_caller<int>, 
        min_caller<float>, min_caller<double>
    };
    funcs[src1.depth()](src1, src2, dst, 0);
}

void cv::gpu::min(const GpuMat& src1, double src2, GpuMat& dst, const Stream& stream) 
{
    typedef void (*func_t)(const GpuMat& src1, double src2, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[] = 
    {
        min_caller<uchar>, min_caller<char>, min_caller<ushort>, min_caller<short>, min_caller<int>, 
        min_caller<float>, min_caller<double>
    };
    funcs[src1.depth()](src1, src2, dst, StreamAccessor::getStream(stream));
}

void cv::gpu::max(const GpuMat& src1, const GpuMat& src2, GpuMat& dst) 
{ 
    typedef void (*func_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[] = 
    {
        max_caller<uchar>, max_caller<char>, max_caller<ushort>, max_caller<short>, max_caller<int>, 
        max_caller<float>, max_caller<double>
    };
    funcs[src1.depth()](src1, src2, dst, 0);
}

void cv::gpu::max(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const Stream& stream) 
{ 
    typedef void (*func_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[] = 
    {
        max_caller<uchar>, max_caller<char>, max_caller<ushort>, max_caller<short>, max_caller<int>, 
        max_caller<float>, max_caller<double>
    };
    funcs[src1.depth()](src1, src2, dst, StreamAccessor::getStream(stream));
}

void cv::gpu::max(const GpuMat& src1, double src2, GpuMat& dst) 
{
    typedef void (*func_t)(const GpuMat& src1, double src2, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[] = 
    {
        max_caller<uchar>, max_caller<char>, max_caller<ushort>, max_caller<short>, max_caller<int>, 
        max_caller<float>, max_caller<double>
    };
    funcs[src1.depth()](src1, src2, dst, 0);
}

void cv::gpu::max(const GpuMat& src1, double src2, GpuMat& dst, const Stream& stream) 
{
    typedef void (*func_t)(const GpuMat& src1, double src2, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[] = 
    {
        max_caller<uchar>, max_caller<char>, max_caller<ushort>, max_caller<short>, max_caller<int>, 
        max_caller<float>, max_caller<double>
    };
    funcs[src1.depth()](src1, src2, dst, StreamAccessor::getStream(stream));
}


#endif /* !defined (HAVE_CUDA) */
