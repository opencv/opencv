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

double cv::cuda::norm(InputArray, int, InputArray, GpuMat&) { throw_no_cuda(); return 0.0; }
double cv::cuda::norm(InputArray, InputArray, GpuMat&, int) { throw_no_cuda(); return 0.0; }

Scalar cv::cuda::sum(InputArray, InputArray, GpuMat&) { throw_no_cuda(); return Scalar(); }
Scalar cv::cuda::absSum(InputArray, InputArray, GpuMat&) { throw_no_cuda(); return Scalar(); }
Scalar cv::cuda::sqrSum(InputArray, InputArray, GpuMat&) { throw_no_cuda(); return Scalar(); }

void cv::cuda::minMax(InputArray, double*, double*, InputArray, GpuMat&) { throw_no_cuda(); }
void cv::cuda::minMaxLoc(InputArray, double*, double*, Point*, Point*, InputArray, GpuMat&, GpuMat&) { throw_no_cuda(); }

int cv::cuda::countNonZero(InputArray, GpuMat&) { throw_no_cuda(); return 0; }

void cv::cuda::reduce(InputArray, OutputArray, int, int, int, Stream&) { throw_no_cuda(); }

void cv::cuda::meanStdDev(InputArray, Scalar&, Scalar&, GpuMat&) { throw_no_cuda(); }

void cv::cuda::rectStdDev(InputArray, InputArray, OutputArray, Rect, Stream&) { throw_no_cuda(); }

void cv::cuda::normalize(InputArray, OutputArray, double, double, int, int, InputArray, GpuMat&, GpuMat&) { throw_no_cuda(); }

void cv::cuda::integral(InputArray, OutputArray, GpuMat&, Stream&) { throw_no_cuda(); }
void cv::cuda::sqrIntegral(InputArray, OutputArray, GpuMat&, Stream&) { throw_no_cuda(); }

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
// norm

double cv::cuda::norm(InputArray _src, int normType, InputArray _mask, GpuMat& buf)
{
    GpuMat src = _src.getGpuMat();
    GpuMat mask = _mask.getGpuMat();

    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );
    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size() && src.channels() == 1) );

    GpuMat src_single_channel = src.reshape(1);

    if (normType == NORM_L1)
        return cuda::absSum(src_single_channel, mask, buf)[0];

    if (normType == NORM_L2)
        return std::sqrt(cuda::sqrSum(src_single_channel, mask, buf)[0]);

    // NORM_INF
    double min_val, max_val;
    cuda::minMax(src_single_channel, &min_val, &max_val, mask, buf);
    return std::max(std::abs(min_val), std::abs(max_val));
}

double cv::cuda::norm(InputArray _src1, InputArray _src2, GpuMat& buf, int normType)
{
#if CUDA_VERSION < 5050
    (void) buf;

    typedef NppStatus (*func_t)(const Npp8u* pSrc1, int nSrcStep1, const Npp8u* pSrc2, int nSrcStep2, NppiSize oSizeROI, Npp64f* pRetVal);

    static const func_t funcs[] = {nppiNormDiff_Inf_8u_C1R, nppiNormDiff_L1_8u_C1R, nppiNormDiff_L2_8u_C1R};
#else
    typedef NppStatus (*func_t)(const Npp8u* pSrc1, int nSrcStep1, const Npp8u* pSrc2, int nSrcStep2,
        NppiSize oSizeROI, Npp64f* pRetVal, Npp8u * pDeviceBuffer);

    typedef NppStatus (*buf_size_func_t)(NppiSize oSizeROI, int* hpBufferSize);

    static const func_t funcs[] = {nppiNormDiff_Inf_8u_C1R, nppiNormDiff_L1_8u_C1R, nppiNormDiff_L2_8u_C1R};

    static const buf_size_func_t buf_size_funcs[] = {nppiNormDiffInfGetBufferHostSize_8u_C1R, nppiNormDiffL1GetBufferHostSize_8u_C1R, nppiNormDiffL2GetBufferHostSize_8u_C1R};
#endif

    GpuMat src1 = _src1.getGpuMat();
    GpuMat src2 = _src2.getGpuMat();

    CV_Assert( src1.type() == CV_8UC1 );
    CV_Assert( src1.size() == src2.size() && src1.type() == src2.type() );
    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );

    NppiSize sz;
    sz.width  = src1.cols;
    sz.height = src1.rows;

    const int funcIdx = normType >> 1;

    DeviceBuffer dbuf;

#if CUDA_VERSION < 5050
    nppSafeCall( funcs[funcIdx](src1.ptr<Npp8u>(), static_cast<int>(src1.step), src2.ptr<Npp8u>(), static_cast<int>(src2.step), sz, dbuf) );
#else
    int bufSize;
    buf_size_funcs[funcIdx](sz, &bufSize);

    ensureSizeIsEnough(1, bufSize, CV_8UC1, buf);

    nppSafeCall( funcs[funcIdx](src1.ptr<Npp8u>(), static_cast<int>(src1.step), src2.ptr<Npp8u>(), static_cast<int>(src2.step), sz, dbuf, buf.data) );
#endif

    cudaSafeCall( cudaDeviceSynchronize() );

    double retVal;
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

Scalar cv::cuda::sum(InputArray _src, InputArray _mask, GpuMat& buf)
{
    GpuMat src = _src.getGpuMat();
    GpuMat mask = _mask.getGpuMat();

    typedef void (*func_t)(PtrStepSzb src, void* buf, double* sum, PtrStepSzb mask);
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

    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    if (src.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    Size buf_size;
    ::sum::getBufSize(src.cols, src.rows, src.channels(), buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);
    buf.setTo(Scalar::all(0));

    const func_t func = funcs[src.depth()][src.channels()];

    double result[4];
    func(src, buf.data, result, mask);

    return Scalar(result[0], result[1], result[2], result[3]);
}

Scalar cv::cuda::absSum(InputArray _src, InputArray _mask, GpuMat& buf)
{
    GpuMat src = _src.getGpuMat();
    GpuMat mask = _mask.getGpuMat();

    typedef void (*func_t)(PtrStepSzb src, void* buf, double* sum, PtrStepSzb mask);
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

    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    if (src.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    Size buf_size;
    ::sum::getBufSize(src.cols, src.rows, src.channels(), buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);
    buf.setTo(Scalar::all(0));

    const func_t func = funcs[src.depth()][src.channels()];

    double result[4];
    func(src, buf.data, result, mask);

    return Scalar(result[0], result[1], result[2], result[3]);
}

Scalar cv::cuda::sqrSum(InputArray _src, InputArray _mask, GpuMat& buf)
{
    GpuMat src = _src.getGpuMat();
    GpuMat mask = _mask.getGpuMat();

    typedef void (*func_t)(PtrStepSzb src, void* buf, double* sum, PtrStepSzb mask);
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

    CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == src.size()) );

    if (src.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    Size buf_size;
    ::sum::getBufSize(src.cols, src.rows, src.channels(), buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);
    buf.setTo(Scalar::all(0));

    const func_t func = funcs[src.depth()][src.channels()];

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

void cv::cuda::minMax(InputArray _src, double* minVal, double* maxVal, InputArray _mask, GpuMat& buf)
{
    GpuMat src = _src.getGpuMat();
    GpuMat mask = _mask.getGpuMat();

    typedef void (*func_t)(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, PtrStepb buf);
    static const func_t funcs[] =
    {
        ::minMax::run<uchar>,
        ::minMax::run<schar>,
        ::minMax::run<ushort>,
        ::minMax::run<short>,
        ::minMax::run<int>,
        ::minMax::run<float>,
        ::minMax::run<double>
    };

    CV_Assert( src.channels() == 1 );
    CV_Assert( mask.empty() || (mask.size() == src.size() && mask.type() == CV_8U) );

    if (src.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    Size buf_size;
    ::minMax::getBufSize(src.cols, src.rows, buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    const func_t func = funcs[src.depth()];

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

void cv::cuda::minMaxLoc(InputArray _src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc,
                        InputArray _mask, GpuMat& valBuf, GpuMat& locBuf)
{
    GpuMat src = _src.getGpuMat();
    GpuMat mask = _mask.getGpuMat();

    typedef void (*func_t)(const PtrStepSzb src, const PtrStepb mask, double* minval, double* maxval, int* minloc, int* maxloc, PtrStepb valbuf, PtrStep<unsigned int> locbuf);
    static const func_t funcs[] =
    {
        ::minMaxLoc::run<uchar>,
        ::minMaxLoc::run<schar>,
        ::minMaxLoc::run<ushort>,
        ::minMaxLoc::run<short>,
        ::minMaxLoc::run<int>,
        ::minMaxLoc::run<float>,
        ::minMaxLoc::run<double>
    };

    CV_Assert( src.channels() == 1 );
    CV_Assert( mask.empty() || (mask.size() == src.size() && mask.type() == CV_8U) );

    if (src.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    Size valbuf_size, locbuf_size;
    ::minMaxLoc::getBufSize(src.cols, src.rows, src.elemSize(), valbuf_size.width, valbuf_size.height, locbuf_size.width, locbuf_size.height);
    ensureSizeIsEnough(valbuf_size, CV_8U, valBuf);
    ensureSizeIsEnough(locbuf_size, CV_8U, locBuf);

    const func_t func = funcs[src.depth()];

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

int cv::cuda::countNonZero(InputArray _src, GpuMat& buf)
{
    GpuMat src = _src.getGpuMat();

    typedef int (*func_t)(const PtrStepSzb src, PtrStep<unsigned int> buf);
    static const func_t funcs[] =
    {
        ::countNonZero::run<uchar>,
        ::countNonZero::run<schar>,
        ::countNonZero::run<ushort>,
        ::countNonZero::run<short>,
        ::countNonZero::run<int>,
        ::countNonZero::run<float>,
        ::countNonZero::run<double>
    };

    CV_Assert(src.channels() == 1);

    if (src.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    Size buf_size;
    ::countNonZero::getBufSize(src.cols, src.rows, buf_size.width, buf_size.height);
    ensureSizeIsEnough(buf_size, CV_8U, buf);

    const func_t func = funcs[src.depth()];

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

void cv::cuda::reduce(InputArray _src, OutputArray _dst, int dim, int reduceOp, int dtype, Stream& stream)
{
    GpuMat src = _src.getGpuMat();

    CV_Assert( src.channels() <= 4 );
    CV_Assert( dim == 0 || dim == 1 );
    CV_Assert( reduceOp == REDUCE_SUM || reduceOp == REDUCE_AVG || reduceOp == REDUCE_MAX || reduceOp == REDUCE_MIN );

    if (dtype < 0)
        dtype = src.depth();

    _dst.create(1, dim == 0 ? src.cols : src.rows, CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()));
    GpuMat dst = _dst.getGpuMat();

    if (dim == 0)
    {
        typedef void (*func_t)(PtrStepSzb src, void* dst, int op, cudaStream_t stream);
        static const func_t funcs[7][7] =
        {
            {
                ::reduce::rows<unsigned char, int, unsigned char>,
                0/*::reduce::rows<unsigned char, int, signed char>*/,
                0/*::reduce::rows<unsigned char, int, unsigned short>*/,
                0/*::reduce::rows<unsigned char, int, short>*/,
                ::reduce::rows<unsigned char, int, int>,
                ::reduce::rows<unsigned char, float, float>,
                ::reduce::rows<unsigned char, double, double>
            },
            {
                0/*::reduce::rows<signed char, int, unsigned char>*/,
                0/*::reduce::rows<signed char, int, signed char>*/,
                0/*::reduce::rows<signed char, int, unsigned short>*/,
                0/*::reduce::rows<signed char, int, short>*/,
                0/*::reduce::rows<signed char, int, int>*/,
                0/*::reduce::rows<signed char, float, float>*/,
                0/*::reduce::rows<signed char, double, double>*/
            },
            {
                0/*::reduce::rows<unsigned short, int, unsigned char>*/,
                0/*::reduce::rows<unsigned short, int, signed char>*/,
                ::reduce::rows<unsigned short, int, unsigned short>,
                0/*::reduce::rows<unsigned short, int, short>*/,
                ::reduce::rows<unsigned short, int, int>,
                ::reduce::rows<unsigned short, float, float>,
                ::reduce::rows<unsigned short, double, double>
            },
            {
                0/*::reduce::rows<short, int, unsigned char>*/,
                0/*::reduce::rows<short, int, signed char>*/,
                0/*::reduce::rows<short, int, unsigned short>*/,
                ::reduce::rows<short, int, short>,
                ::reduce::rows<short, int, int>,
                ::reduce::rows<short, float, float>,
                ::reduce::rows<short, double, double>
            },
            {
                0/*::reduce::rows<int, int, unsigned char>*/,
                0/*::reduce::rows<int, int, signed char>*/,
                0/*::reduce::rows<int, int, unsigned short>*/,
                0/*::reduce::rows<int, int, short>*/,
                ::reduce::rows<int, int, int>,
                ::reduce::rows<int, float, float>,
                ::reduce::rows<int, double, double>
            },
            {
                0/*::reduce::rows<float, float, unsigned char>*/,
                0/*::reduce::rows<float, float, signed char>*/,
                0/*::reduce::rows<float, float, unsigned short>*/,
                0/*::reduce::rows<float, float, short>*/,
                0/*::reduce::rows<float, float, int>*/,
                ::reduce::rows<float, float, float>,
                ::reduce::rows<float, double, double>
            },
            {
                0/*::reduce::rows<double, double, unsigned char>*/,
                0/*::reduce::rows<double, double, signed char>*/,
                0/*::reduce::rows<double, double, unsigned short>*/,
                0/*::reduce::rows<double, double, short>*/,
                0/*::reduce::rows<double, double, int>*/,
                0/*::reduce::rows<double, double, float>*/,
                ::reduce::rows<double, double, double>
            }
        };

        const func_t func = funcs[src.depth()][dst.depth()];

        if (!func)
            CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of input and output array formats");

        func(src.reshape(1), dst.data, reduceOp, StreamAccessor::getStream(stream));
    }
    else
    {
        typedef void (*func_t)(PtrStepSzb src, void* dst, int cn, int op, cudaStream_t stream);
        static const func_t funcs[7][7] =
        {
            {
                ::reduce::cols<unsigned char, int, unsigned char>,
                0/*::reduce::cols<unsigned char, int, signed char>*/,
                0/*::reduce::cols<unsigned char, int, unsigned short>*/,
                0/*::reduce::cols<unsigned char, int, short>*/,
                ::reduce::cols<unsigned char, int, int>,
                ::reduce::cols<unsigned char, float, float>,
                ::reduce::cols<unsigned char, double, double>
            },
            {
                0/*::reduce::cols<signed char, int, unsigned char>*/,
                0/*::reduce::cols<signed char, int, signed char>*/,
                0/*::reduce::cols<signed char, int, unsigned short>*/,
                0/*::reduce::cols<signed char, int, short>*/,
                0/*::reduce::cols<signed char, int, int>*/,
                0/*::reduce::cols<signed char, float, float>*/,
                0/*::reduce::cols<signed char, double, double>*/
            },
            {
                0/*::reduce::cols<unsigned short, int, unsigned char>*/,
                0/*::reduce::cols<unsigned short, int, signed char>*/,
                ::reduce::cols<unsigned short, int, unsigned short>,
                0/*::reduce::cols<unsigned short, int, short>*/,
                ::reduce::cols<unsigned short, int, int>,
                ::reduce::cols<unsigned short, float, float>,
                ::reduce::cols<unsigned short, double, double>
            },
            {
                0/*::reduce::cols<short, int, unsigned char>*/,
                0/*::reduce::cols<short, int, signed char>*/,
                0/*::reduce::cols<short, int, unsigned short>*/,
                ::reduce::cols<short, int, short>,
                ::reduce::cols<short, int, int>,
                ::reduce::cols<short, float, float>,
                ::reduce::cols<short, double, double>
            },
            {
                0/*::reduce::cols<int, int, unsigned char>*/,
                0/*::reduce::cols<int, int, signed char>*/,
                0/*::reduce::cols<int, int, unsigned short>*/,
                0/*::reduce::cols<int, int, short>*/,
                ::reduce::cols<int, int, int>,
                ::reduce::cols<int, float, float>,
                ::reduce::cols<int, double, double>
            },
            {
                0/*::reduce::cols<float, float, unsigned char>*/,
                0/*::reduce::cols<float, float, signed char>*/,
                0/*::reduce::cols<float, float, unsigned short>*/,
                0/*::reduce::cols<float, float, short>*/,
                0/*::reduce::cols<float, float, int>*/,
                ::reduce::cols<float, float, float>,
                ::reduce::cols<float, double, double>
            },
            {
                0/*::reduce::cols<double, double, unsigned char>*/,
                0/*::reduce::cols<double, double, signed char>*/,
                0/*::reduce::cols<double, double, unsigned short>*/,
                0/*::reduce::cols<double, double, short>*/,
                0/*::reduce::cols<double, double, int>*/,
                0/*::reduce::cols<double, double, float>*/,
                ::reduce::cols<double, double, double>
            }
        };

        const func_t func = funcs[src.depth()][dst.depth()];

        if (!func)
            CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported combination of input and output array formats");

        func(src, dst.data, src.channels(), reduceOp, StreamAccessor::getStream(stream));
    }
}

////////////////////////////////////////////////////////////////////////
// meanStdDev

void cv::cuda::meanStdDev(InputArray _src, Scalar& mean, Scalar& stddev, GpuMat& buf)
{
    GpuMat src = _src.getGpuMat();

    CV_Assert( src.type() == CV_8UC1 );

    if (!deviceSupports(FEATURE_SET_COMPUTE_13))
        CV_Error(cv::Error::StsNotImplemented, "Not sufficient compute capebility");

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    DeviceBuffer dbuf(2);

    int bufSize;
#if (CUDA_VERSION <= 4020)
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

//////////////////////////////////////////////////////////////////////////////
// rectStdDev

void cv::cuda::rectStdDev(InputArray _src, InputArray _sqr, OutputArray _dst, Rect rect, Stream& _stream)
{
    GpuMat src = _src.getGpuMat();
    GpuMat sqr = _sqr.getGpuMat();

    CV_Assert( src.type() == CV_32SC1 && sqr.type() == CV_64FC1 );

    _dst.create(src.size(), CV_32FC1);
    GpuMat dst = _dst.getGpuMat();

    NppiSize sz;
    sz.width = src.cols;
    sz.height = src.rows;

    NppiRect nppRect;
    nppRect.height = rect.height;
    nppRect.width = rect.width;
    nppRect.x = rect.x;
    nppRect.y = rect.y;

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    NppStreamHandler h(stream);

    nppSafeCall( nppiRectStdDev_32s32f_C1R(src.ptr<Npp32s>(), static_cast<int>(src.step), sqr.ptr<Npp64f>(), static_cast<int>(sqr.step),
                dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz, nppRect) );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// normalize

void cv::cuda::normalize(InputArray _src, OutputArray dst, double a, double b, int norm_type, int dtype, InputArray mask, GpuMat& norm_buf, GpuMat& cvt_buf)
{
    GpuMat src = _src.getGpuMat();

    double scale = 1, shift = 0;

    if (norm_type == NORM_MINMAX)
    {
        double smin = 0, smax = 0;
        double dmin = std::min(a, b), dmax = std::max(a, b);
        cuda::minMax(src, &smin, &smax, mask, norm_buf);
        scale = (dmax - dmin) * (smax - smin > std::numeric_limits<double>::epsilon() ? 1.0 / (smax - smin) : 0.0);
        shift = dmin - smin * scale;
    }
    else if (norm_type == NORM_L2 || norm_type == NORM_L1 || norm_type == NORM_INF)
    {
        scale = cuda::norm(src, norm_type, mask, norm_buf);
        scale = scale > std::numeric_limits<double>::epsilon() ? a / scale : 0.0;
        shift = 0;
    }
    else
    {
        CV_Error(cv::Error::StsBadArg, "Unknown/unsupported norm type");
    }

    if (mask.empty())
    {
        src.convertTo(dst, dtype, scale, shift);
    }
    else
    {
        src.convertTo(cvt_buf, dtype, scale, shift);
        cvt_buf.copyTo(dst, mask);
    }
}

////////////////////////////////////////////////////////////////////////
// integral

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        void shfl_integral_gpu(const PtrStepSzb& img, PtrStepSz<unsigned int> integral, cudaStream_t stream);
    }
}}}

void cv::cuda::integral(InputArray _src, OutputArray _dst, GpuMat& buffer, Stream& _stream)
{
    GpuMat src = _src.getGpuMat();

    CV_Assert( src.type() == CV_8UC1 );

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    cv::Size whole;
    cv::Point offset;
    src.locateROI(whole, offset);

    if (deviceSupports(WARP_SHUFFLE_FUNCTIONS) && src.cols <= 2048
        && offset.x % 16 == 0 && ((src.cols + 63) / 64) * 64 <= (static_cast<int>(src.step) - offset.x))
    {
        ensureSizeIsEnough(((src.rows + 7) / 8) * 8, ((src.cols + 63) / 64) * 64, CV_32SC1, buffer);

        cv::cuda::device::imgproc::shfl_integral_gpu(src, buffer, stream);

        _dst.create(src.rows + 1, src.cols + 1, CV_32SC1);
        GpuMat dst = _dst.getGpuMat();

        dst.setTo(Scalar::all(0), _stream);

        GpuMat inner = dst(Rect(1, 1, src.cols, src.rows));
        GpuMat res = buffer(Rect(0, 0, src.cols, src.rows));

        res.copyTo(inner, _stream);
    }
    else
    {
    #ifndef HAVE_OPENCV_CUDALEGACY
        throw_no_cuda();
    #else
        _dst.create(src.rows + 1, src.cols + 1, CV_32SC1);
        GpuMat dst = _dst.getGpuMat();

        NcvSize32u roiSize;
        roiSize.width = src.cols;
        roiSize.height = src.rows;

        cudaDeviceProp prop;
        cudaSafeCall( cudaGetDeviceProperties(&prop, cv::cuda::getDevice()) );

        Ncv32u bufSize;
        ncvSafeCall( nppiStIntegralGetSize_8u32u(roiSize, &bufSize, prop) );
        ensureSizeIsEnough(1, bufSize, CV_8UC1, buffer);

        NppStStreamHandler h(stream);

        ncvSafeCall( nppiStIntegral_8u32u_C1R(const_cast<Ncv8u*>(src.ptr<Ncv8u>()), static_cast<int>(src.step),
            dst.ptr<Ncv32u>(), static_cast<int>(dst.step), roiSize, buffer.ptr<Ncv8u>(), bufSize, prop) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    #endif
    }
}

//////////////////////////////////////////////////////////////////////////////
// sqrIntegral

void cv::cuda::sqrIntegral(InputArray _src, OutputArray _dst, GpuMat& buf, Stream& _stream)
{
#ifndef HAVE_OPENCV_CUDALEGACY
    (void) _src;
    (void) _dst;
    (void) _stream;
    throw_no_cuda();
#else
    GpuMat src = _src.getGpuMat();

    CV_Assert( src.type() == CV_8U );

    NcvSize32u roiSize;
    roiSize.width = src.cols;
    roiSize.height = src.rows;

    cudaDeviceProp prop;
    cudaSafeCall( cudaGetDeviceProperties(&prop, cv::cuda::getDevice()) );

    Ncv32u bufSize;
    ncvSafeCall(nppiStSqrIntegralGetSize_8u64u(roiSize, &bufSize, prop));

    ensureSizeIsEnough(1, bufSize, CV_8U, buf);

    cudaStream_t stream = StreamAccessor::getStream(_stream);

    NppStStreamHandler h(stream);

    _dst.create(src.rows + 1, src.cols + 1, CV_64F);
    GpuMat dst = _dst.getGpuMat();

    ncvSafeCall(nppiStSqrIntegral_8u64u_C1R(const_cast<Ncv8u*>(src.ptr<Ncv8u>(0)), static_cast<int>(src.step),
            dst.ptr<Ncv64u>(0), static_cast<int>(dst.step), roiSize, buf.ptr<Ncv8u>(0), bufSize, prop));

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
#endif
}

#endif
