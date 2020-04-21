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

double cv::cuda::norm(InputArray, int, InputArray) { throw_no_cuda(); return 0.0; }
void cv::cuda::calcNorm(InputArray, OutputArray, int, InputArray, Stream&) { throw_no_cuda(); }
double cv::cuda::norm(InputArray, InputArray, int) { throw_no_cuda(); return 0.0; }
void cv::cuda::calcNormDiff(InputArray, InputArray, OutputArray, int, Stream&) { throw_no_cuda(); }

Scalar cv::cuda::sum(InputArray, InputArray) { throw_no_cuda(); return Scalar(); }
void cv::cuda::calcSum(InputArray, OutputArray, InputArray, Stream&) { throw_no_cuda(); }
Scalar cv::cuda::absSum(InputArray, InputArray) { throw_no_cuda(); return Scalar(); }
void cv::cuda::calcAbsSum(InputArray, OutputArray, InputArray, Stream&) { throw_no_cuda(); }
Scalar cv::cuda::sqrSum(InputArray, InputArray) { throw_no_cuda(); return Scalar(); }
void cv::cuda::calcSqrSum(InputArray, OutputArray, InputArray, Stream&) { throw_no_cuda(); }

void cv::cuda::minMax(InputArray, double*, double*, InputArray) { throw_no_cuda(); }
void cv::cuda::findMinMax(InputArray, OutputArray, InputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::minMaxLoc(InputArray, double*, double*, Point*, Point*, InputArray) { throw_no_cuda(); }
void cv::cuda::findMinMaxLoc(InputArray, OutputArray, OutputArray, InputArray, Stream&) { throw_no_cuda(); }

int cv::cuda::countNonZero(InputArray) { throw_no_cuda(); return 0; }
void cv::cuda::countNonZero(InputArray, OutputArray, Stream&) { throw_no_cuda(); }

void cv::cuda::reduce(InputArray, OutputArray, int, int, int, Stream&) { throw_no_cuda(); }

void cv::cuda::meanStdDev(InputArray, Scalar&, Scalar&) { throw_no_cuda(); }
void cv::cuda::meanStdDev(InputArray, OutputArray, Stream&) { throw_no_cuda(); }

void cv::cuda::rectStdDev(InputArray, InputArray, OutputArray, Rect, Stream&) { throw_no_cuda(); }

void cv::cuda::normalize(InputArray, OutputArray, double, double, int, int, InputArray, Stream&) { throw_no_cuda(); }

void cv::cuda::integral(InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::sqrIntegral(InputArray, OutputArray, Stream&) { throw_no_cuda(); }

#else

////////////////////////////////////////////////////////////////////////
// norm

namespace cv { namespace cuda { namespace device {

void normL2(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _mask, Stream& stream);

void findMaxAbs(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _mask, Stream& stream);

}}}

void cv::cuda::calcNorm(InputArray _src, OutputArray dst, int normType, InputArray mask, Stream& stream)
{
    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );

    GpuMat src = getInputMat(_src, stream);

    GpuMat src_single_channel = src.reshape(1);

    if (normType == NORM_L1)
    {
        calcAbsSum(src_single_channel, dst, mask, stream);
    }
    else if (normType == NORM_L2)
    {
        cv::cuda::device::normL2(src_single_channel, dst, mask, stream);
    }
    else // NORM_INF
    {
        cv::cuda::device::findMaxAbs(src_single_channel, dst, mask, stream);
    }
}

double cv::cuda::norm(InputArray _src, int normType, InputArray _mask)
{
    Stream& stream = Stream::Null();

    HostMem dst;
    calcNorm(_src, dst, normType, _mask, stream);

    stream.waitForCompletion();

    double val;
    dst.createMatHeader().convertTo(Mat(1, 1, CV_64FC1, &val), CV_64F);

    return val;
}

////////////////////////////////////////////////////////////////////////
// meanStdDev

void cv::cuda::meanStdDev(InputArray _src, OutputArray _dst, Stream& stream)
{
    if (!deviceSupports(FEATURE_SET_COMPUTE_13))
        CV_Error(cv::Error::StsNotImplemented, "Not sufficient compute capebility");

    const GpuMat src = getInputMat(_src, stream);

    CV_Assert( src.type() == CV_8UC1 );

    GpuMat dst = getOutputMat(_dst, 1, 2, CV_64FC1, stream);

    NppiSize sz;
    sz.width  = src.cols;
    sz.height = src.rows;

    int bufSize;
#if (CUDA_VERSION <= 4020)
    nppSafeCall( nppiMeanStdDev8uC1RGetBufferHostSize(sz, &bufSize) );
#else
    nppSafeCall( nppiMeanStdDevGetBufferHostSize_8u_C1R(sz, &bufSize) );
#endif

    BufferPool pool(stream);
    GpuMat buf = pool.getBuffer(1, bufSize, CV_8UC1);

    // detail: https://github.com/opencv/opencv/issues/11063
    //NppStreamHandler h(StreamAccessor::getStream(stream));

    nppSafeCall( nppiMean_StdDev_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step), sz, buf.ptr<Npp8u>(), dst.ptr<Npp64f>(), dst.ptr<Npp64f>() + 1) );

    syncOutput(dst, _dst, stream);
}

void cv::cuda::meanStdDev(InputArray _src, Scalar& mean, Scalar& stddev)
{
    Stream& stream = Stream::Null();

    HostMem dst;
    meanStdDev(_src, dst, stream);

    stream.waitForCompletion();

    double vals[2];
    dst.createMatHeader().copyTo(Mat(1, 2, CV_64FC1, &vals[0]));

    mean = Scalar(vals[0]);
    stddev = Scalar(vals[1]);
}

//////////////////////////////////////////////////////////////////////////////
// rectStdDev

void cv::cuda::rectStdDev(InputArray _src, InputArray _sqr, OutputArray _dst, Rect rect, Stream& _stream)
{
    GpuMat src = getInputMat(_src, _stream);
    GpuMat sqr = getInputMat(_sqr, _stream);

    CV_Assert( src.type() == CV_32SC1 && sqr.type() == CV_64FC1 );

    GpuMat dst = getOutputMat(_dst, src.size(), CV_32FC1, _stream);

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

    syncOutput(dst, _dst, _stream);
}

#endif
