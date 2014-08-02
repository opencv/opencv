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

#endif
