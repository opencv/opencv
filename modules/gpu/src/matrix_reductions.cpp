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

void cv::gpu::meanStdDev(const GpuMat&, Scalar&, Scalar&) { throw_nogpu(); }
void cv::gpu::meanStdDev(const GpuMat&, Scalar&, Scalar&, GpuMat&) { throw_nogpu(); }
double cv::gpu::norm(const GpuMat&, int) { throw_nogpu(); return 0.0; }
double cv::gpu::norm(const GpuMat&, int, GpuMat&) { throw_nogpu(); return 0.0; }
double cv::gpu::norm(const GpuMat&, int, const GpuMat&, GpuMat&) { throw_nogpu(); return 0.0; }
double cv::gpu::norm(const GpuMat&, const GpuMat&, int) { throw_nogpu(); return 0.0; }
Scalar cv::gpu::sum(const GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::sum(const GpuMat&, GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::sum(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::absSum(const GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::absSum(const GpuMat&, GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::absSum(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::sqrSum(const GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::sqrSum(const GpuMat&, GpuMat&) { throw_nogpu(); return Scalar(); }
Scalar cv::gpu::sqrSum(const GpuMat&, const GpuMat&, GpuMat&) { throw_nogpu(); return Scalar(); }
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
    GpuMat buf;
    meanStdDev(src, mean, stddev, buf);
}

void cv::gpu::meanStdDev(const GpuMat& src, Scalar& mean, Scalar& stddev, GpuMat& buf)
{
    CV_Assert(src.type() == CV_8UC1);

    if (!deviceSupports(FEATURE_SET_COMPUTE_13))
        CV_Error(CV_StsNotImplemented, "Not sufficient compute capebility");

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    DeviceBuffer dbuf(2);

    int bufSize;
#if (CUDART_VERSION <= 4020)
    nppSafeCall( nppiMeanStdDev8uC1RGetBufferHostSize(sz, &bufSize) );
#else
    nppSafeCall( nppiMeanStdDevGetBufferHostSize_8u_C1R(sz, &bufSize) );
#endif

    ensureSizeIsEnough(1, bufSize, CV_8UC1, buf);

    nppSafeCall( nppiMean_StdDev_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step), sz, buf.ptr<Npp8u>(), dbuf, (double*)dbuf + 1) );

    cudaSafeCall( cudaDeviceSynchronize() );

    double* ptrs[2] = {mean.val, stddev.val};
    dbuf.download(ptrs);
}

////////////////////////////////////////////////////////////////////////
// norm

double cv::gpu::norm(const GpuMat& src, int normType)
{
    GpuMat buf;
    return norm(src, normType, GpuMat(), buf);
}

double cv::gpu::norm(const GpuMat& src, int normType, GpuMat& buf)
{
    return norm(src, normType, GpuMat(), buf);
}

double cv::gpu::norm(const GpuMat& src, int normType, const GpuMat& mask, GpuMat& buf)
{
    CV_Assert(normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2);
    CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size() && src.channels() == 1));

    GpuMat src_single_channel = src.reshape(1);

    if (normType == NORM_L1)
        return absSum(src_single_channel, mask, buf)[0];

    if (normType == NORM_L2)
        return std::sqrt(sqrSum(src_single_channel, mask, buf)[0]);

    // NORM_INF
    double min_val, max_val;
    minMax(src_single_channel, &min_val, &max_val, mask, buf);
    return std::max(std::abs(min_val), std::abs(max_val));
}

double cv::gpu::norm(const GpuMat& src1, const GpuMat& src2, int normType)
{
    CV_Assert(src1.type() == CV_8UC1);
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    CV_Assert(normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2);

#if CUDART_VERSION < 5050
    typedef NppStatus (*func_t)(const Npp8u* pSrc1, int nSrcStep1, const Npp8u* pSrc2, int nSrcStep2, NppiSize oSizeROI, Npp64f* pRetVal);

    static const func_t funcs[] = {nppiNormDiff_Inf_8u_C1R, nppiNormDiff_L1_8u_C1R, nppiNormDiff_L2_8u_C1R};
#else
    typedef NppStatus (*func_t)(const Npp8u* pSrc1, int nSrcStep1, const Npp8u* pSrc2, int nSrcStep2,
        NppiSize oSizeROI, Npp64f* pRetVal, Npp8u * pDeviceBuffer);

    typedef NppStatus (*buf_size_func_t)(NppiSize oSizeROI, int* hpBufferSize);

    static const func_t funcs[] = {nppiNormDiff_Inf_8u_C1R, nppiNormDiff_L1_8u_C1R, nppiNormDiff_L2_8u_C1R};

    static const buf_size_func_t buf_size_funcs[] = {nppiNormDiffInfGetBufferHostSize_8u_C1R, nppiNormDiffL1GetBufferHostSize_8u_C1R, nppiNormDiffL2GetBufferHostSize_8u_C1R};
#endif

    NppiSize sz;
    sz.width  = src1.cols;
    sz.height = src1.rows;

    int funcIdx = normType >> 1;

    double retVal;

    DeviceBuffer dbuf;

#if CUDART_VERSION < 5050
    nppSafeCall( funcs[funcIdx](src1.ptr<Npp8u>(), static_cast<int>(src1.step), src2.ptr<Npp8u>(), static_cast<int>(src2.step), sz, dbuf) );
#else
    int bufSize;
    buf_size_funcs[funcIdx](sz, &bufSize);

    GpuMat buf(1, bufSize, CV_8UC1);

    nppSafeCall( funcs[funcIdx](src1.ptr<Npp8u>(), static_cast<int>(src1.step), src2.ptr<Npp8u>(), static_cast<int>(src2.step), sz, dbuf, buf.data) );
#endif

    cudaSafeCall( cudaDeviceSynchronize() );

    dbuf.download(&retVal);

    return retVal;
}

////////////////////////////////////////////////////////////////////////
// Sum

namespace sum
{
    void getBufSize(int cols, int rows, int cn, int& bufcols, int& bufrows);

    template <typename T, int cn>
    void run(PtrStepSzb src, void* buf, double* sum, PtrStepSzb mask);

    template <typename T, int cn>
    void runAbs(PtrStepSzb src, void* buf, double* sum, PtrStepSzb mask);

    template <typename T, int cn>
    void runSqr(PtrStepSzb src, void* buf, double* sum, PtrStepSzb mask);
}

Scalar cv::gpu::sum(const GpuMat& src)
{
    GpuMat buf;
    return sum(src, GpuMat(), buf);
}

Scalar cv::gpu::sum(const GpuMat& src, GpuMat& buf)
{
    return sum(src, GpuMat(), buf);
}

Scalar cv::gpu::sum(const GpuMat& src, const GpuMat& mask, GpuMat& buf)
{
    typedef void (*func_t)(PtrStepSzb src, void* buf, double* sum, PtrStepSzb mask);
#ifdef OPENCV_TINY_GPU_MODULE
    static const func_t funcs[7][5] =
    {
        {0, ::sum::run<uchar , 1>, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, ::sum::run<float , 1>, 0, 0, 0},
        {0, 0, 0, 0, 0},
    };
#else
    static const func_t funcs[7][5] =
    {
        {0, ::sum::run<uchar , 1>, ::sum::run<uchar , 2>, ::sum::run<uchar , 3>, ::sum::run<uchar , 4>},
        {0, ::sum::run<schar , 1>, ::sum::run<schar , 2>, ::sum::run<schar , 3>, ::sum::run<schar , 4>},
        {0, ::sum::run<ushort, 1>, ::sum::run<ushort, 2>, ::sum::run<ushort, 3>, ::sum::run<ushort, 4>},
        {0, ::sum::run<short , 1>, ::sum::run<short , 2>, ::sum::run<short , 3>, ::sum::run<short , 4>},
        {0, ::sum::run<int   , 1>, ::sum::run<int   , 2>, ::sum::run<int   , 3>, ::sum::run<int   , 4>},
        {0, ::sum::run<float , 1>, ::sum::run<float , 2>, ::sum::run<float , 3>, ::sum::run<float , 4>},
        {0, ::sum::run<double, 1>, ::sum::run<double, 2>, ::sum::run<double, 3>, ::sum::run<double, 4>}
    };
#endif

    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    if (src.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    Size buf_size;
    ::sum::getBufSize(src.cols, src.rows, src.channels(), buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);
    buf.setTo(Scalar::all(0));

    const func_t func = funcs[src.depth()][src.channels()];
    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    double result[4];
    func(src, buf.data, result, mask);

    return Scalar(result[0], result[1], result[2], result[3]);
}

Scalar cv::gpu::absSum(const GpuMat& src)
{
    GpuMat buf;
    return absSum(src, GpuMat(), buf);
}

Scalar cv::gpu::absSum(const GpuMat& src, GpuMat& buf)
{
    return absSum(src, GpuMat(), buf);
}

Scalar cv::gpu::absSum(const GpuMat& src, const GpuMat& mask, GpuMat& buf)
{
    typedef void (*func_t)(PtrStepSzb src, void* buf, double* sum, PtrStepSzb mask);
#ifdef OPENCV_TINY_GPU_MODULE
    static const func_t funcs[7][5] =
    {
        {0, ::sum::runAbs<uchar , 1>, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, ::sum::runAbs<float , 1>, 0, 0, 0},
        {0, 0, 0, 0, 0},
    };
#else
    static const func_t funcs[7][5] =
    {
        {0, ::sum::runAbs<uchar , 1>, ::sum::runAbs<uchar , 2>, ::sum::runAbs<uchar , 3>, ::sum::runAbs<uchar , 4>},
        {0, ::sum::runAbs<schar , 1>, ::sum::runAbs<schar , 2>, ::sum::runAbs<schar , 3>, ::sum::runAbs<schar , 4>},
        {0, ::sum::runAbs<ushort, 1>, ::sum::runAbs<ushort, 2>, ::sum::runAbs<ushort, 3>, ::sum::runAbs<ushort, 4>},
        {0, ::sum::runAbs<short , 1>, ::sum::runAbs<short , 2>, ::sum::runAbs<short , 3>, ::sum::runAbs<short , 4>},
        {0, ::sum::runAbs<int   , 1>, ::sum::runAbs<int   , 2>, ::sum::runAbs<int   , 3>, ::sum::runAbs<int   , 4>},
        {0, ::sum::runAbs<float , 1>, ::sum::runAbs<float , 2>, ::sum::runAbs<float , 3>, ::sum::runAbs<float , 4>},
        {0, ::sum::runAbs<double, 1>, ::sum::runAbs<double, 2>, ::sum::runAbs<double, 3>, ::sum::runAbs<double, 4>}
    };
#endif

    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    if (src.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    Size buf_size;
    ::sum::getBufSize(src.cols, src.rows, src.channels(), buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);
    buf.setTo(Scalar::all(0));

    const func_t func = funcs[src.depth()][src.channels()];
    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    double result[4];
    func(src, buf.data, result, mask);

    return Scalar(result[0], result[1], result[2], result[3]);
}

Scalar cv::gpu::sqrSum(const GpuMat& src)
{
    GpuMat buf;
    return sqrSum(src, GpuMat(), buf);
}

Scalar cv::gpu::sqrSum(const GpuMat& src, GpuMat& buf)
{
    return sqrSum(src, GpuMat(), buf);
}

Scalar cv::gpu::sqrSum(const GpuMat& src, const GpuMat& mask, GpuMat& buf)
{
    typedef void (*func_t)(PtrStepSzb src, void* buf, double* sum, PtrStepSzb mask);
#ifdef OPENCV_TINY_GPU_MODULE
    static const func_t funcs[7][5] =
    {
        {0, ::sum::runSqr<uchar , 1>, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, ::sum::runSqr<float , 1>, 0, 0, 0},
        {0, 0, 0, 0, 0},
    };
#else
    static const func_t funcs[7][5] =
    {
        {0, ::sum::runSqr<uchar , 1>, ::sum::runSqr<uchar , 2>, ::sum::runSqr<uchar , 3>, ::sum::runSqr<uchar , 4>},
        {0, ::sum::runSqr<schar , 1>, ::sum::runSqr<schar , 2>, ::sum::runSqr<schar , 3>, ::sum::runSqr<schar , 4>},
        {0, ::sum::runSqr<ushort, 1>, ::sum::runSqr<ushort, 2>, ::sum::runSqr<ushort, 3>, ::sum::runSqr<ushort, 4>},
        {0, ::sum::runSqr<short , 1>, ::sum::runSqr<short , 2>, ::sum::runSqr<short , 3>, ::sum::runSqr<short , 4>},
        {0, ::sum::runSqr<int   , 1>, ::sum::runSqr<int   , 2>, ::sum::runSqr<int   , 3>, ::sum::runSqr<int   , 4>},
        {0, ::sum::runSqr<float , 1>, ::sum::runSqr<float , 2>, ::sum::runSqr<float , 3>, ::sum::runSqr<float , 4>},
        {0, ::sum::runSqr<double, 1>, ::sum::runSqr<double, 2>, ::sum::runSqr<double, 3>, ::sum::runSqr<double, 4>}
    };
#endif

    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    if (src.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    Size buf_size;
    ::sum::getBufSize(src.cols, src.rows, src.channels(), buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);
    buf.setTo(Scalar::all(0));

    const func_t func = funcs[src.depth()][src.channels()];
    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    double result[4];
    func(src, buf.data, result, mask);

    return Scalar(result[0], result[1], result[2], result[3]);
}

////////////////////////////////////////////////////////////////////////
// minMax

namespace minMax
{
    void getBufSize(int cols, int rows, int& bufcols, int& bufrows);

    template <typename T>
    void run(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
}

void cv::gpu::minMax(const GpuMat& src, double* minVal, double* maxVal, const GpuMat& mask)
{
    GpuMat buf;
    minMax(src, minVal, maxVal, mask, buf);
}

void cv::gpu::minMax(const GpuMat& src, double* minVal, double* maxVal, const GpuMat& mask, GpuMat& buf)
{
    typedef void (*func_t)(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
#ifdef OPENCV_TINY_GPU_MODULE
    static const func_t funcs[] =
    {
        ::minMax::run<uchar>,
        0/*::minMax::run<schar>*/,
        0/*::minMax::run<ushort>*/,
        0/*::minMax::run<short>*/,
        0/*::minMax::run<int>*/,
        ::minMax::run<float>,
        0/*::minMax::run<double>*/,
    };
#else
    static const func_t funcs[] =
    {
        ::minMax::run<uchar>,
        ::minMax::run<schar>,
        ::minMax::run<ushort>,
        ::minMax::run<short>,
        ::minMax::run<int>,
        ::minMax::run<float>,
        ::minMax::run<double>,
    };
#endif

    CV_Assert( src.channels() == 1 );
    CV_Assert( mask.empty() || (mask.size() == src.size() && mask.type() == CV_8U) );

    if (src.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    Size buf_size;
    ::minMax::getBufSize(src.cols, src.rows, buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    const func_t func = funcs[src.depth()];
    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    double temp1, temp2;
    func(src, mask, minVal ? minVal : &temp1, maxVal ? maxVal : &temp2, buf);
}

////////////////////////////////////////////////////////////////////////
// minMaxLoc

namespace minMaxLoc
{
    void getBufSize(int cols, int rows, size_t elem_size, int& b1cols, int& b1rows, int& b2cols, int& b2rows);

    template <typename T>
    void run(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
}

void cv::gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc, const GpuMat& mask)
{
    GpuMat valBuf, locBuf;
    minMaxLoc(src, minVal, maxVal, minLoc, maxLoc, mask, valBuf, locBuf);
}

void cv::gpu::minMaxLoc(const GpuMat& src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc,
                        const GpuMat& mask, GpuMat& valBuf, GpuMat& locBuf)
{
    typedef void (*func_t)(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
#ifdef OPENCV_TINY_GPU_MODULE
    static const func_t funcs[] =
    {
        ::minMaxLoc::run<uchar>,
        0/*::minMaxLoc::run<schar>*/,
        0/*::minMaxLoc::run<ushort>*/,
        0/*::minMaxLoc::run<short>*/,
        ::minMaxLoc::run<int>,
        ::minMaxLoc::run<float>,
        0/*::minMaxLoc::run<double>*/,
    };
#else
    static const func_t funcs[] =
    {
        ::minMaxLoc::run<uchar>,
        ::minMaxLoc::run<schar>,
        ::minMaxLoc::run<ushort>,
        ::minMaxLoc::run<short>,
        ::minMaxLoc::run<int>,
        ::minMaxLoc::run<float>,
        ::minMaxLoc::run<double>,
    };
#endif

    CV_Assert( src.channels() == 1 );
    CV_Assert( mask.empty() || (mask.size() == src.size() && mask.type() == CV_8U) );

    if (src.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    Size valbuf_size, locbuf_size;
    ::minMaxLoc::getBufSize(src.cols, src.rows, src.elemSize(), valbuf_size.width, valbuf_size.height, locbuf_size.width, locbuf_size.height);
    ensureSizeIsEnough(valbuf_size, CV_8U, valBuf);
    ensureSizeIsEnough(locbuf_size, CV_8U, locBuf);

    const func_t func = funcs[src.depth()];
    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    double temp1, temp2;
    Point temp3, temp4;
    func(src, mask, minVal ? minVal : &temp1, maxVal ? maxVal : &temp2, minLoc ? &minLoc->x : &temp3.x, maxLoc ? &maxLoc->x : &temp4.x, valBuf, locBuf);
}

//////////////////////////////////////////////////////////////////////////////
// countNonZero

namespace countNonZero
{
    void getBufSize(int cols, int rows, int& bufcols, int& bufrows);

    template <typename T>
    int run(const PtrStepSzb src, PtrStep<unsigned int> buf);
}

int cv::gpu::countNonZero(const GpuMat& src)
{
    GpuMat buf;
    return countNonZero(src, buf);
}

int cv::gpu::countNonZero(const GpuMat& src, GpuMat& buf)
{
    typedef int (*func_t)(const PtrStepSzb src, PtrStep<unsigned int> buf);
#ifdef OPENCV_TINY_GPU_MODULE
    static const func_t funcs[] =
    {
        ::countNonZero::run<uchar>,
        0/*::countNonZero::run<schar>*/,
        0/*::countNonZero::run<ushort>*/,
        0/*::countNonZero::run<short>*/,
        0/*::countNonZero::run<int>*/,
        ::countNonZero::run<float>,
        0/*::countNonZero::run<double>*/,
    };
#else
    static const func_t funcs[] =
    {
        ::countNonZero::run<uchar>,
        ::countNonZero::run<schar>,
        ::countNonZero::run<ushort>,
        ::countNonZero::run<short>,
        ::countNonZero::run<int>,
        ::countNonZero::run<float>,
        ::countNonZero::run<double>,
    };
#endif

    CV_Assert(src.channels() == 1);

    if (src.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    Size buf_size;
    ::countNonZero::getBufSize(src.cols, src.rows, buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    const func_t func = funcs[src.depth()];
    if (!func)
        CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of source and destination types");

    return func(src, buf);
}

//////////////////////////////////////////////////////////////////////////////
// reduce

namespace reduce
{
    template <typename T, typename S, typename D>
    void rows(PtrStepSzb src, void* dst, int op, cudaStream_t stream);

    template <typename T, typename S, typename D>
    void cols(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
}

void cv::gpu::reduce(const GpuMat& src, GpuMat& dst, int dim, int reduceOp, int dtype, Stream& stream)
{
    CV_Assert( src.channels() <= 4 );
    CV_Assert( dim == 0 || dim == 1 );
    CV_Assert( reduceOp == CV_REDUCE_SUM || reduceOp == CV_REDUCE_AVG || reduceOp == CV_REDUCE_MAX || reduceOp == CV_REDUCE_MIN );

    if (dtype < 0)
        dtype = src.depth();

    dst.create(1, dim == 0 ? src.cols : src.rows, CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()));

    if (dim == 0)
    {
        typedef void (*func_t)(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
#ifdef OPENCV_TINY_GPU_MODULE
        static const func_t funcs[7][7] =
        {
            {
                ::reduce::rows<unsigned char, int, unsigned char>,
                0/*::reduce::rows<unsigned char, int, signed char>*/,
                0/*::reduce::rows<unsigned char, int, unsigned short>*/,
                0/*::reduce::rows<unsigned char, int, short>*/,
                0/*::reduce::rows<unsigned char, int, int>*/,
                ::reduce::rows<unsigned char, float, float>,
                0/*::reduce::rows<unsigned char, double, double>*/,
            },
            {
                0/*::reduce::rows<signed char, int, unsigned char>*/,
                0/*::reduce::rows<signed char, int, signed char>*/,
                0/*::reduce::rows<signed char, int, unsigned short>*/,
                0/*::reduce::rows<signed char, int, short>*/,
                0/*::reduce::rows<signed char, int, int>*/,
                0/*::reduce::rows<signed char, float, float>*/,
                0/*::reduce::rows<signed char, double, double>*/,
            },
            {
                0/*::reduce::rows<unsigned short, int, unsigned char>*/,
                0/*::reduce::rows<unsigned short, int, signed char>*/,
                0/*::reduce::rows<unsigned short, int, unsigned short>*/,
                0/*::reduce::rows<unsigned short, int, short>*/,
                0/*::reduce::rows<unsigned short, int, int>*/,
                0/*::reduce::rows<unsigned short, float, float>*/,
                0/*::reduce::rows<unsigned short, double, double>*/,
            },
            {
                0/*::reduce::rows<short, int, unsigned char>*/,
                0/*::reduce::rows<short, int, signed char>*/,
                0/*::reduce::rows<short, int, unsigned short>*/,
                0/*::reduce::rows<short, int, short>*/,
                0/*::reduce::rows<short, int, int>*/,
                0/*::reduce::rows<short, float, float>*/,
                0/*::reduce::rows<short, double, double>*/,
            },
            {
                0/*::reduce::rows<int, int, unsigned char>*/,
                0/*::reduce::rows<int, int, signed char>*/,
                0/*::reduce::rows<int, int, unsigned short>*/,
                0/*::reduce::rows<int, int, short>*/,
                0/*::reduce::rows<int, int, int>*/,
                0/*::reduce::rows<int, float, float>*/,
                0/*::reduce::rows<int, double, double>*/,
            },
            {
                0/*::reduce::rows<float, float, unsigned char>*/,
                0/*::reduce::rows<float, float, signed char>*/,
                0/*::reduce::rows<float, float, unsigned short>*/,
                0/*::reduce::rows<float, float, short>*/,
                0/*::reduce::rows<float, float, int>*/,
                ::reduce::rows<float, float, float>,
                0/*::reduce::rows<float, double, double>*/,
            },
            {
                0/*::reduce::rows<double, double, unsigned char>*/,
                0/*::reduce::rows<double, double, signed char>*/,
                0/*::reduce::rows<double, double, unsigned short>*/,
                0/*::reduce::rows<double, double, short>*/,
                0/*::reduce::rows<double, double, int>*/,
                0/*::reduce::rows<double, double, float>*/,
                0/*::reduce::rows<double, double, double>*/,
            }
        };
#else
        static const func_t funcs[7][7] =
        {
            {
                ::reduce::rows<unsigned char, int, unsigned char>,
                0/*::reduce::rows<unsigned char, int, signed char>*/,
                0/*::reduce::rows<unsigned char, int, unsigned short>*/,
                0/*::reduce::rows<unsigned char, int, short>*/,
                ::reduce::rows<unsigned char, int, int>,
                ::reduce::rows<unsigned char, float, float>,
                ::reduce::rows<unsigned char, double, double>,
            },
            {
                0/*::reduce::rows<signed char, int, unsigned char>*/,
                0/*::reduce::rows<signed char, int, signed char>*/,
                0/*::reduce::rows<signed char, int, unsigned short>*/,
                0/*::reduce::rows<signed char, int, short>*/,
                0/*::reduce::rows<signed char, int, int>*/,
                0/*::reduce::rows<signed char, float, float>*/,
                0/*::reduce::rows<signed char, double, double>*/,
            },
            {
                0/*::reduce::rows<unsigned short, int, unsigned char>*/,
                0/*::reduce::rows<unsigned short, int, signed char>*/,
                ::reduce::rows<unsigned short, int, unsigned short>,
                0/*::reduce::rows<unsigned short, int, short>*/,
                ::reduce::rows<unsigned short, int, int>,
                ::reduce::rows<unsigned short, float, float>,
                ::reduce::rows<unsigned short, double, double>,
            },
            {
                0/*::reduce::rows<short, int, unsigned char>*/,
                0/*::reduce::rows<short, int, signed char>*/,
                0/*::reduce::rows<short, int, unsigned short>*/,
                ::reduce::rows<short, int, short>,
                ::reduce::rows<short, int, int>,
                ::reduce::rows<short, float, float>,
                ::reduce::rows<short, double, double>,
            },
            {
                0/*::reduce::rows<int, int, unsigned char>*/,
                0/*::reduce::rows<int, int, signed char>*/,
                0/*::reduce::rows<int, int, unsigned short>*/,
                0/*::reduce::rows<int, int, short>*/,
                ::reduce::rows<int, int, int>,
                ::reduce::rows<int, float, float>,
                ::reduce::rows<int, double, double>,
            },
            {
                0/*::reduce::rows<float, float, unsigned char>*/,
                0/*::reduce::rows<float, float, signed char>*/,
                0/*::reduce::rows<float, float, unsigned short>*/,
                0/*::reduce::rows<float, float, short>*/,
                0/*::reduce::rows<float, float, int>*/,
                ::reduce::rows<float, float, float>,
                ::reduce::rows<float, double, double>,
            },
            {
                0/*::reduce::rows<double, double, unsigned char>*/,
                0/*::reduce::rows<double, double, signed char>*/,
                0/*::reduce::rows<double, double, unsigned short>*/,
                0/*::reduce::rows<double, double, short>*/,
                0/*::reduce::rows<double, double, int>*/,
                0/*::reduce::rows<double, double, float>*/,
                ::reduce::rows<double, double, double>,
            }
        };
#endif

        const func_t func = funcs[src.depth()][dst.depth()];

        if (!func)
            CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of input and output array formats");

        func(src.reshape(1), dst.data, reduceOp, StreamAccessor::getStream(stream));
    }
    else
    {
        typedef void (*func_t)(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
#ifdef OPENCV_TINY_GPU_MODULE
        static const func_t funcs[7][7] =
        {
            {
                ::reduce::cols<unsigned char, int, unsigned char>,
                0/*::reduce::cols<unsigned char, int, signed char>*/,
                0/*::reduce::cols<unsigned char, int, unsigned short>*/,
                0/*::reduce::cols<unsigned char, int, short>*/,
                0/*::reduce::cols<unsigned char, int, int>*/,
                ::reduce::cols<unsigned char, float, float>,
                0/*::reduce::cols<unsigned char, double, double>*/,
            },
            {
                0/*::reduce::cols<signed char, int, unsigned char>*/,
                0/*::reduce::cols<signed char, int, signed char>*/,
                0/*::reduce::cols<signed char, int, unsigned short>*/,
                0/*::reduce::cols<signed char, int, short>*/,
                0/*::reduce::cols<signed char, int, int>*/,
                0/*::reduce::cols<signed char, float, float>*/,
                0/*::reduce::cols<signed char, double, double>*/,
            },
            {
                0/*::reduce::cols<unsigned short, int, unsigned char>*/,
                0/*::reduce::cols<unsigned short, int, signed char>*/,
                0/*::reduce::cols<unsigned short, int, unsigned short>*/,
                0/*::reduce::cols<unsigned short, int, short>*/,
                0/*::reduce::cols<unsigned short, int, int>*/,
                0/*::reduce::cols<unsigned short, float, float>*/,
                0/*::reduce::cols<unsigned short, double, double>*/,
            },
            {
                0/*::reduce::cols<short, int, unsigned char>*/,
                0/*::reduce::cols<short, int, signed char>*/,
                0/*::reduce::cols<short, int, unsigned short>*/,
                0/*::reduce::cols<short, int, short>*/,
                0/*::reduce::cols<short, int, int>*/,
                0/*::reduce::cols<short, float, float>*/,
                0/*::reduce::cols<short, double, double>*/,
            },
            {
                0/*::reduce::cols<int, int, unsigned char>*/,
                0/*::reduce::cols<int, int, signed char>*/,
                0/*::reduce::cols<int, int, unsigned short>*/,
                0/*::reduce::cols<int, int, short>*/,
                0/*::reduce::cols<int, int, int>*/,
                0/*::reduce::cols<int, float, float>*/,
                0/*::reduce::cols<int, double, double>*/,
            },
            {
                0/*::reduce::cols<float, float, unsigned char>*/,
                0/*::reduce::cols<float, float, signed char>*/,
                0/*::reduce::cols<float, float, unsigned short>*/,
                0/*::reduce::cols<float, float, short>*/,
                0/*::reduce::cols<float, float, int>*/,
                ::reduce::cols<float, float, float>,
                0/*::reduce::cols<float, double, double>*/,
            },
            {
                0/*::reduce::cols<double, double, unsigned char>*/,
                0/*::reduce::cols<double, double, signed char>*/,
                0/*::reduce::cols<double, double, unsigned short>*/,
                0/*::reduce::cols<double, double, short>*/,
                0/*::reduce::cols<double, double, int>*/,
                0/*::reduce::cols<double, double, float>*/,
                0/*::reduce::cols<double, double, double>*/,
            }
        };
#else
        static const func_t funcs[7][7] =
        {
            {
                ::reduce::cols<unsigned char, int, unsigned char>,
                0/*::reduce::cols<unsigned char, int, signed char>*/,
                0/*::reduce::cols<unsigned char, int, unsigned short>*/,
                0/*::reduce::cols<unsigned char, int, short>*/,
                ::reduce::cols<unsigned char, int, int>,
                ::reduce::cols<unsigned char, float, float>,
                ::reduce::cols<unsigned char, double, double>,
            },
            {
                0/*::reduce::cols<signed char, int, unsigned char>*/,
                0/*::reduce::cols<signed char, int, signed char>*/,
                0/*::reduce::cols<signed char, int, unsigned short>*/,
                0/*::reduce::cols<signed char, int, short>*/,
                0/*::reduce::cols<signed char, int, int>*/,
                0/*::reduce::cols<signed char, float, float>*/,
                0/*::reduce::cols<signed char, double, double>*/,
            },
            {
                0/*::reduce::cols<unsigned short, int, unsigned char>*/,
                0/*::reduce::cols<unsigned short, int, signed char>*/,
                ::reduce::cols<unsigned short, int, unsigned short>,
                0/*::reduce::cols<unsigned short, int, short>*/,
                ::reduce::cols<unsigned short, int, int>,
                ::reduce::cols<unsigned short, float, float>,
                ::reduce::cols<unsigned short, double, double>,
            },
            {
                0/*::reduce::cols<short, int, unsigned char>*/,
                0/*::reduce::cols<short, int, signed char>*/,
                0/*::reduce::cols<short, int, unsigned short>*/,
                ::reduce::cols<short, int, short>,
                ::reduce::cols<short, int, int>,
                ::reduce::cols<short, float, float>,
                ::reduce::cols<short, double, double>,
            },
            {
                0/*::reduce::cols<int, int, unsigned char>*/,
                0/*::reduce::cols<int, int, signed char>*/,
                0/*::reduce::cols<int, int, unsigned short>*/,
                0/*::reduce::cols<int, int, short>*/,
                ::reduce::cols<int, int, int>,
                ::reduce::cols<int, float, float>,
                ::reduce::cols<int, double, double>,
            },
            {
                0/*::reduce::cols<float, float, unsigned char>*/,
                0/*::reduce::cols<float, float, signed char>*/,
                0/*::reduce::cols<float, float, unsigned short>*/,
                0/*::reduce::cols<float, float, short>*/,
                0/*::reduce::cols<float, float, int>*/,
                ::reduce::cols<float, float, float>,
                ::reduce::cols<float, double, double>,
            },
            {
                0/*::reduce::cols<double, double, unsigned char>*/,
                0/*::reduce::cols<double, double, signed char>*/,
                0/*::reduce::cols<double, double, unsigned short>*/,
                0/*::reduce::cols<double, double, short>*/,
                0/*::reduce::cols<double, double, int>*/,
                0/*::reduce::cols<double, double, float>*/,
                ::reduce::cols<double, double, double>,
            }
        };
#endif

        const func_t func = funcs[src.depth()][dst.depth()];

        if (!func)
            CV_Error(CV_StsUnsupportedFormat, "Unsupported combination of input and output array formats");

        func(src, dst.data, src.channels(), reduceOp, StreamAccessor::getStream(stream));
    }
}

#endif
