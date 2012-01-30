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
double cv::gpu::norm(const GpuMat&, int, GpuMat&) { throw_nogpu(); return 0.0; }
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
void cv::gpu::reduce(const GpuMat&, GpuMat&, int, int, int, Stream&) { throw_nogpu(); }

#else

namespace
{
    class DeviceBuffer
    {
    public:
        explicit DeviceBuffer(int count_ = 1) : count(count_)
        {
            cudaSafeCall( cudaMalloc(&pdev, count * sizeof(double)) );
        }
        ~DeviceBuffer()
        {
            cudaSafeCall( cudaFree(pdev) );
        }

        operator double*() {return pdev;}

        void download(double* hptr)
        {
            double hbuf;
            cudaSafeCall( cudaMemcpy(&hbuf, pdev, sizeof(double), cudaMemcpyDeviceToHost) );
            *hptr = hbuf;
        }
        void download(double** hptrs)
        {
            AutoBuffer<double, 2 * sizeof(double)> hbuf(count);
            cudaSafeCall( cudaMemcpy((void*)hbuf, pdev, count * sizeof(double), cudaMemcpyDeviceToHost) );
            for (int i = 0; i < count; ++i)
                *hptrs[i] = hbuf[i];
        }

    private:
        double* pdev;
        int count;
    };
}


////////////////////////////////////////////////////////////////////////
// meanStdDev

void cv::gpu::meanStdDev(const GpuMat& src, Scalar& mean, Scalar& stddev)
{
    CV_Assert(src.type() == CV_8UC1);

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    DeviceBuffer dbuf(2);

#if CUDART_VERSION > 4000 
    int bufSize;
    nppSafeCall( nppiMeanStdDev8uC1RGetBufferHostSize(sz, &bufSize) );

    GpuMat buf(1, bufSize, CV_8UC1);
    nppSafeCall( nppiMean_StdDev_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step), sz, buf.ptr<Npp8u>(), dbuf, (double*)dbuf + 1) );
#else
    nppSafeCall( nppiMean_StdDev_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step), sz, dbuf, (double*)dbuf + 1) );
#endif

    cudaSafeCall( cudaDeviceSynchronize() );
    
    double* ptrs[2] = {mean.val, stddev.val};
    dbuf.download(ptrs);
}


////////////////////////////////////////////////////////////////////////
// norm

double cv::gpu::norm(const GpuMat& src, int normType)
{
    GpuMat buf;
    return norm(src, normType, buf);
}

double cv::gpu::norm(const GpuMat& src, int normType, GpuMat& buf)
{
    GpuMat src_single_channel = src.reshape(1);

    if (normType == NORM_L1)
        return absSum(src_single_channel, buf)[0];

    if (normType == NORM_L2)
        return sqrt(sqrSum(src_single_channel, buf)[0]);

    if (normType == NORM_INF)
    {
        double min_val, max_val;
        minMax(src_single_channel, &min_val, &max_val, GpuMat(), buf);
        return std::max(std::abs(min_val), std::abs(max_val));
    }

    CV_Error(CV_StsBadArg, "norm: unsupported norm type");
    return 0;
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

    DeviceBuffer dbuf;

    nppSafeCall( npp_norm_diff_func[funcIdx](src1.ptr<Npp8u>(), static_cast<int>(src1.step), src2.ptr<Npp8u>(), static_cast<int>(src2.step), sz, dbuf) );

    cudaSafeCall( cudaDeviceSynchronize() );
    
    dbuf.download(&retVal);

    return retVal;
}

////////////////////////////////////////////////////////////////////////
// Sum

namespace cv { namespace gpu { namespace device 
{
    namespace matrix_reductions 
    {
        namespace sum
        {
            template <typename T>
            void sumCaller(const DevMem2Db src, PtrStepb buf, double* sum, int cn);

            template <typename T>
            void sumMultipassCaller(const DevMem2Db src, PtrStepb buf, double* sum, int cn);

            template <typename T>
            void absSumCaller(const DevMem2Db src, PtrStepb buf, double* sum, int cn);

            template <typename T>
            void absSumMultipassCaller(const DevMem2Db src, PtrStepb buf, double* sum, int cn);

            template <typename T>
            void sqrSumCaller(const DevMem2Db src, PtrStepb buf, double* sum, int cn);

            template <typename T>
            void sqrSumMultipassCaller(const DevMem2Db src, PtrStepb buf, double* sum, int cn);

            void getBufSizeRequired(int cols, int rows, int cn, int& bufcols, int& bufrows);
        }
    }
}}}

Scalar cv::gpu::sum(const GpuMat& src) 
{
    GpuMat buf;
    return sum(src, buf);
}


Scalar cv::gpu::sum(const GpuMat& src, GpuMat& buf) 
{
    using namespace ::cv::gpu::device::matrix_reductions::sum;

    typedef void (*Caller)(const DevMem2Db, PtrStepb, double*, int);

    static Caller multipass_callers[7] = 
    { 
        sumMultipassCaller<unsigned char>, sumMultipassCaller<char>, 
        sumMultipassCaller<unsigned short>, sumMultipassCaller<short>, 
        sumMultipassCaller<int>, sumMultipassCaller<float>, 0 
    };

    static Caller singlepass_callers[7] = { 
        sumCaller<unsigned char>, sumCaller<char>, 
        sumCaller<unsigned short>, sumCaller<short>, 
        sumCaller<int>, sumCaller<float>, 0 
    };

    Size buf_size;
    getBufSizeRequired(src.cols, src.rows, src.channels(), buf_size.width, buf_size.height); 
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    Caller* callers = multipass_callers;
    if (TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS))
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
    using namespace ::cv::gpu::device::matrix_reductions::sum;

    typedef void (*Caller)(const DevMem2Db, PtrStepb, double*, int);

    static Caller multipass_callers[7] = 
    { 
        absSumMultipassCaller<unsigned char>, absSumMultipassCaller<char>, 
        absSumMultipassCaller<unsigned short>, absSumMultipassCaller<short>, 
        absSumMultipassCaller<int>, absSumMultipassCaller<float>, 0 
    };

    static Caller singlepass_callers[7] = 
    {        
        absSumCaller<unsigned char>, absSumCaller<char>, 
        absSumCaller<unsigned short>, absSumCaller<short>, 
        absSumCaller<int>, absSumCaller<float>, 0 
    };

    Size buf_size;
    getBufSizeRequired(src.cols, src.rows, src.channels(), buf_size.width, buf_size.height); 
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    Caller* callers = multipass_callers;
    if (TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS))
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
    using namespace ::cv::gpu::device::matrix_reductions::sum;

    typedef void (*Caller)(const DevMem2Db, PtrStepb, double*, int);

    static Caller multipass_callers[7] = 
    { 
        sqrSumMultipassCaller<unsigned char>, sqrSumMultipassCaller<char>, 
        sqrSumMultipassCaller<unsigned short>, sqrSumMultipassCaller<short>, 
        sqrSumMultipassCaller<int>, sqrSumMultipassCaller<float>, 0 
    };

    static Caller singlepass_callers[7] = 
    { 
        sqrSumCaller<unsigned char>, sqrSumCaller<char>, 
        sqrSumCaller<unsigned short>, sqrSumCaller<short>, 
        sqrSumCaller<int>, sqrSumCaller<float>, 0 
    };

    Caller* callers = multipass_callers;
    if (TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS))
        callers = singlepass_callers;

    Size buf_size;
    getBufSizeRequired(src.cols, src.rows, src.channels(), buf_size.width, buf_size.height); 
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    Caller caller = callers[src.depth()];
    if (!caller) CV_Error(CV_StsBadArg, "sqrSum: unsupported type");

    double result[4];
    caller(src, buf, result, src.channels());
    return Scalar(result[0], result[1], result[2], result[3]);
}

////////////////////////////////////////////////////////////////////////
// Find min or max

namespace cv { namespace gpu { namespace device 
{
    namespace matrix_reductions 
    {
        namespace minmax 
        {
            void getBufSizeRequired(int cols, int rows, int elem_size, int& bufcols, int& bufrows);
            
            template <typename T> 
            void minMaxCaller(const DevMem2Db src, double* minval, double* maxval, PtrStepb buf);

            template <typename T> 
            void minMaxMaskCaller(const DevMem2Db src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);

            template <typename T> 
            void minMaxMultipassCaller(const DevMem2Db src, double* minval, double* maxval, PtrStepb buf);

            template <typename T> 
            void minMaxMaskMultipassCaller(const DevMem2Db src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
        }
    }
}}}


void cv::gpu::minMax(const GpuMat& src, double* minVal, double* maxVal, const GpuMat& mask)
{
    GpuMat buf;
    minMax(src, minVal, maxVal, mask, buf);
}


void cv::gpu::minMax(const GpuMat& src, double* minVal, double* maxVal, const GpuMat& mask, GpuMat& buf)
{
    using namespace ::cv::gpu::device::matrix_reductions::minmax;

    typedef void (*Caller)(const DevMem2Db, double*, double*, PtrStepb);
    typedef void (*MaskedCaller)(const DevMem2Db, const PtrStepb, double*, double*, PtrStepb);

    static Caller multipass_callers[7] = 
    { 
        minMaxMultipassCaller<unsigned char>, minMaxMultipassCaller<char>, 
        minMaxMultipassCaller<unsigned short>, minMaxMultipassCaller<short>, 
        minMaxMultipassCaller<int>, minMaxMultipassCaller<float>, 0 
    };

    static Caller singlepass_callers[7] = 
    { 
        minMaxCaller<unsigned char>, minMaxCaller<char>, 
        minMaxCaller<unsigned short>, minMaxCaller<short>, 
        minMaxCaller<int>, minMaxCaller<float>, minMaxCaller<double> 
    };

    static MaskedCaller masked_multipass_callers[7] = 
    { 
        minMaxMaskMultipassCaller<unsigned char>, minMaxMaskMultipassCaller<char>, 
        minMaxMaskMultipassCaller<unsigned short>, minMaxMaskMultipassCaller<short>,
        minMaxMaskMultipassCaller<int>, minMaxMaskMultipassCaller<float>, 0
    };

    static MaskedCaller masked_singlepass_callers[7] =
    { 
        minMaxMaskCaller<unsigned char>, minMaxMaskCaller<char>, 
        minMaxMaskCaller<unsigned short>, minMaxMaskCaller<short>, 
        minMaxMaskCaller<int>, minMaxMaskCaller<float>, minMaxMaskCaller<double> 
    };

    CV_Assert(src.channels() == 1);

    CV_Assert(mask.empty() || (mask.type() == CV_8U && src.size() == mask.size()));

    double minVal_; if (!minVal) minVal = &minVal_;
    double maxVal_; if (!maxVal) maxVal = &maxVal_;
    
    Size buf_size;
    getBufSizeRequired(src.cols, src.rows, static_cast<int>(src.elemSize()), buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    if (mask.empty())
    {
        Caller* callers = multipass_callers;
        if (TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS))
            callers = singlepass_callers;

        Caller caller = callers[src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMax: unsupported type");
        caller(src, minVal, maxVal, buf);
    }
    else
    {
        MaskedCaller* callers = masked_multipass_callers;
        if (TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS))
            callers = masked_singlepass_callers;

        MaskedCaller caller = callers[src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMax: unsupported type");
        caller(src, mask, minVal, maxVal, buf);
    }
}


////////////////////////////////////////////////////////////////////////
// Locate min and max

namespace cv { namespace gpu { namespace device 
{
    namespace matrix_reductions 
    {
        namespace minmaxloc 
        {
            void getBufSizeRequired(int cols, int rows, int elem_size, int& b1cols, 
                                    int& b1rows, int& b2cols, int& b2rows);

            template <typename T> 
            void minMaxLocCaller(const DevMem2Db src, double* minval, double* maxval, 
                                 int minloc[2], int maxloc[2], PtrStepb valBuf, PtrStepb locBuf);

            template <typename T> 
            void minMaxLocMaskCaller(const DevMem2Db src, const PtrStepb mask, double* minval, double* maxval, 
                                     int minloc[2], int maxloc[2], PtrStepb valBuf, PtrStepb locBuf);

            template <typename T> 
            void minMaxLocMultipassCaller(const DevMem2Db src, double* minval, double* maxval, 
                                          int minloc[2], int maxloc[2], PtrStepb valBuf, PtrStepb locBuf);

            template <typename T> 
            void minMaxLocMaskMultipassCaller(const DevMem2Db src, const PtrStepb mask, double* minval, double* maxval, 
                                              int minloc[2], int maxloc[2], PtrStepb valBuf, PtrStepb locBuf);
        }
    }
}}}

void cv::gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc, const GpuMat& mask)
{    
    GpuMat valBuf, locBuf;
    minMaxLoc(src, minVal, maxVal, minLoc, maxLoc, mask, valBuf, locBuf);
}

void cv::gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc,
                        const GpuMat& mask, GpuMat& valBuf, GpuMat& locBuf)
{
    using namespace ::cv::gpu::device::matrix_reductions::minmaxloc;

    typedef void (*Caller)(const DevMem2Db, double*, double*, int[2], int[2], PtrStepb, PtrStepb);
    typedef void (*MaskedCaller)(const DevMem2Db, const PtrStepb, double*, double*, int[2], int[2], PtrStepb, PtrStepb);

    static Caller multipass_callers[7] = 
    {
        minMaxLocMultipassCaller<unsigned char>, minMaxLocMultipassCaller<char>, 
        minMaxLocMultipassCaller<unsigned short>, minMaxLocMultipassCaller<short>, 
        minMaxLocMultipassCaller<int>, minMaxLocMultipassCaller<float>, 0 
    };

    static Caller singlepass_callers[7] = 
    {
        minMaxLocCaller<unsigned char>, minMaxLocCaller<char>, 
        minMaxLocCaller<unsigned short>, minMaxLocCaller<short>, 
        minMaxLocCaller<int>, minMaxLocCaller<float>, minMaxLocCaller<double> 
    };

    static MaskedCaller masked_multipass_callers[7] = 
    {
        minMaxLocMaskMultipassCaller<unsigned char>, minMaxLocMaskMultipassCaller<char>,
        minMaxLocMaskMultipassCaller<unsigned short>, minMaxLocMaskMultipassCaller<short>, 
        minMaxLocMaskMultipassCaller<int>, minMaxLocMaskMultipassCaller<float>, 0 
    };

    static MaskedCaller masked_singlepass_callers[7] = 
    { 
        minMaxLocMaskCaller<unsigned char>, minMaxLocMaskCaller<char>, 
        minMaxLocMaskCaller<unsigned short>, minMaxLocMaskCaller<short>, 
        minMaxLocMaskCaller<int>, minMaxLocMaskCaller<float>, minMaxLocMaskCaller<double> 
    };

    CV_Assert(src.channels() == 1);

    CV_Assert(mask.empty() || (mask.type() == CV_8U && src.size() == mask.size()));

    double minVal_; if (!minVal) minVal = &minVal_;
    double maxVal_; if (!maxVal) maxVal = &maxVal_;
    int minLoc_[2];
    int maxLoc_[2];

    Size valbuf_size, locbuf_size;
    getBufSizeRequired(src.cols, src.rows, static_cast<int>(src.elemSize()), valbuf_size.width, 
                       valbuf_size.height, locbuf_size.width, locbuf_size.height);
    ensureSizeIsEnough(valbuf_size, CV_8U, valBuf);
    ensureSizeIsEnough(locbuf_size, CV_8U, locBuf);

    if (mask.empty())
    {
        Caller* callers = multipass_callers;
        if (TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS))
            callers = singlepass_callers;

        Caller caller = callers[src.type()];
        if (!caller) CV_Error(CV_StsBadArg, "minMaxLoc: unsupported type");
        caller(src, minVal, maxVal, minLoc_, maxLoc_, valBuf, locBuf);
    }
    else
    {
        MaskedCaller* callers = masked_multipass_callers;
        if (TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS))
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

namespace cv { namespace gpu { namespace device 
{
    namespace matrix_reductions 
    {
        namespace countnonzero 
        {
            void getBufSizeRequired(int cols, int rows, int& bufcols, int& bufrows);

            template <typename T> 
            int countNonZeroCaller(const DevMem2Db src, PtrStepb buf);

            template <typename T> 
            int countNonZeroMultipassCaller(const DevMem2Db src, PtrStepb buf);
        }
    }
}}}

int cv::gpu::countNonZero(const GpuMat& src)
{
    GpuMat buf;
    return countNonZero(src, buf);
}


int cv::gpu::countNonZero(const GpuMat& src, GpuMat& buf)
{
    using namespace ::cv::gpu::device::matrix_reductions::countnonzero;

    typedef int (*Caller)(const DevMem2Db src, PtrStepb buf);

    static Caller multipass_callers[7] = 
    {
        countNonZeroMultipassCaller<unsigned char>, countNonZeroMultipassCaller<char>,
        countNonZeroMultipassCaller<unsigned short>, countNonZeroMultipassCaller<short>,
        countNonZeroMultipassCaller<int>, countNonZeroMultipassCaller<float>, 0 
    };

    static Caller singlepass_callers[7] = 
    {
        countNonZeroCaller<unsigned char>, countNonZeroCaller<char>,
        countNonZeroCaller<unsigned short>, countNonZeroCaller<short>,
        countNonZeroCaller<int>, countNonZeroCaller<float>, countNonZeroCaller<double> };

    CV_Assert(src.channels() == 1);

    Size buf_size;
    getBufSizeRequired(src.cols, src.rows, buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    Caller* callers = multipass_callers;
    if (TargetArchs::builtWith(GLOBAL_ATOMICS) && DeviceInfo().supports(GLOBAL_ATOMICS))
        callers = singlepass_callers;

    Caller caller = callers[src.type()];
    if (!caller) CV_Error(CV_StsBadArg, "countNonZero: unsupported type");
    return caller(src, buf);
}

//////////////////////////////////////////////////////////////////////////////
// reduce

namespace cv { namespace gpu { namespace device 
{
    namespace matrix_reductions 
    {
        template <typename T, typename S, typename D> void reduceRows_gpu(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);
        template <typename T, typename S, typename D> void reduceCols_gpu(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);
    }
}}}

void cv::gpu::reduce(const GpuMat& src, GpuMat& dst, int dim, int reduceOp, int dtype, Stream& stream)
{
    using namespace ::cv::gpu::device::matrix_reductions;

    CV_Assert(src.depth() <= CV_32F && src.channels() <= 4 && dtype <= CV_32F);
    CV_Assert(dim == 0 || dim == 1);
    CV_Assert(reduceOp == CV_REDUCE_SUM || reduceOp == CV_REDUCE_AVG || reduceOp == CV_REDUCE_MAX || reduceOp == CV_REDUCE_MIN);

    if (dtype < 0)
        dtype = src.depth();

    dst.create(1, dim == 0 ? src.cols : src.rows, CV_MAKETYPE(dtype, src.channels()));

    if (dim == 0)
    {
        typedef void (*caller_t)(const DevMem2Db& src, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);

        static const caller_t callers[6][6] = 
        {
            {
                reduceRows_gpu<unsigned char, int, unsigned char>,
                0/*reduceRows_gpu<unsigned char, int, signed char>*/,
                0/*reduceRows_gpu<unsigned char, int, unsigned short>*/,
                0/*reduceRows_gpu<unsigned char, int, short>*/,
                reduceRows_gpu<unsigned char, int, int>,
                reduceRows_gpu<unsigned char, int, float>
            },
            {
                0/*reduceRows_gpu<signed char, int, unsigned char>*/,
                0/*reduceRows_gpu<signed char, int, signed char>*/,
                0/*reduceRows_gpu<signed char, int, unsigned short>*/,
                0/*reduceRows_gpu<signed char, int, short>*/,
                0/*reduceRows_gpu<signed char, int, int>*/,
                0/*reduceRows_gpu<signed char, int, float>*/
            },
            {
                0/*reduceRows_gpu<unsigned short, int, unsigned char>*/,
                0/*reduceRows_gpu<unsigned short, int, signed char>*/,
                reduceRows_gpu<unsigned short, int, unsigned short>,
                0/*reduceRows_gpu<unsigned short, int, short>*/,
                reduceRows_gpu<unsigned short, int, int>,
                reduceRows_gpu<unsigned short, int, float>
            },
            {
                0/*reduceRows_gpu<short, int, unsigned char>*/,
                0/*reduceRows_gpu<short, int, signed char>*/,
                0/*reduceRows_gpu<short, int, unsigned short>*/,
                reduceRows_gpu<short, int, short>,
                reduceRows_gpu<short, int, int>,
                reduceRows_gpu<short, int, float>
            },
            {
                0/*reduceRows_gpu<int, int, unsigned char>*/,
                0/*reduceRows_gpu<int, int, signed char>*/,
                0/*reduceRows_gpu<int, int, unsigned short>*/,
                0/*reduceRows_gpu<int, int, short>*/,
                reduceRows_gpu<int, int, int>,
                reduceRows_gpu<int, int, float>
            },
            {
                0/*reduceRows_gpu<float, float, unsigned char>*/,
                0/*reduceRows_gpu<float, float, signed char>*/,
                0/*reduceRows_gpu<float, float, unsigned short>*/,
                0/*reduceRows_gpu<float, float, short>*/,
                0/*reduceRows_gpu<float, float, int>*/,
                reduceRows_gpu<float, float, float>
            }
        };

        const caller_t func = callers[src.depth()][dst.depth()];
        if (!func)
            CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of input and output array formats");

        func(src.reshape(1), dst.reshape(1), reduceOp, StreamAccessor::getStream(stream));
    }
    else
    {
        typedef void (*caller_t)(const DevMem2Db& src, int cn, const DevMem2Db& dst, int reduceOp, cudaStream_t stream);

        static const caller_t callers[6][6] = 
        {
            {
                reduceCols_gpu<unsigned char, int, unsigned char>,
                0/*reduceCols_gpu<unsigned char, int, signed char>*/,
                0/*reduceCols_gpu<unsigned char, int, unsigned short>*/,
                0/*reduceCols_gpu<unsigned char, int, short>*/,
                reduceCols_gpu<unsigned char, int, int>,
                reduceCols_gpu<unsigned char, int, float>
            },
            {
                0/*reduceCols_gpu<signed char, int, unsigned char>*/,
                0/*reduceCols_gpu<signed char, int, signed char>*/,
                0/*reduceCols_gpu<signed char, int, unsigned short>*/,
                0/*reduceCols_gpu<signed char, int, short>*/,
                0/*reduceCols_gpu<signed char, int, int>*/,
                0/*reduceCols_gpu<signed char, int, float>*/
            },
            {
                0/*reduceCols_gpu<unsigned short, int, unsigned char>*/,
                0/*reduceCols_gpu<unsigned short, int, signed char>*/,
                reduceCols_gpu<unsigned short, int, unsigned short>,
                0/*reduceCols_gpu<unsigned short, int, short>*/,
                reduceCols_gpu<unsigned short, int, int>,
                reduceCols_gpu<unsigned short, int, float>
            },
            {
                0/*reduceCols_gpu<short, int, unsigned char>*/,
                0/*reduceCols_gpu<short, int, signed char>*/,
                0/*reduceCols_gpu<short, int, unsigned short>*/,
                reduceCols_gpu<short, int, short>,
                reduceCols_gpu<short, int, int>,
                reduceCols_gpu<short, int, float>
            },
            {
                0/*reduceCols_gpu<int, int, unsigned char>*/,
                0/*reduceCols_gpu<int, int, signed char>*/,
                0/*reduceCols_gpu<int, int, unsigned short>*/,
                0/*reduceCols_gpu<int, int, short>*/,
                reduceCols_gpu<int, int, int>,
                reduceCols_gpu<int, int, float>
            },
            {
                0/*reduceCols_gpu<float, unsigned char>*/,
                0/*reduceCols_gpu<float, signed char>*/,
                0/*reduceCols_gpu<float, unsigned short>*/,
                0/*reduceCols_gpu<float, short>*/,
                0/*reduceCols_gpu<float, int>*/,
                reduceCols_gpu<float, float, float>
            }
        };

        const caller_t func = callers[src.depth()][dst.depth()];
        if (!func)
            CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of input and output array formats");

        func(src, src.channels(), dst, reduceOp, StreamAccessor::getStream(stream));        
    }
}

#endif
