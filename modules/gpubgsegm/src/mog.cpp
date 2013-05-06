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

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

cv::gpu::MOG_GPU::MOG_GPU(int) { throw_no_cuda(); }
void cv::gpu::MOG_GPU::initialize(cv::Size, int) { throw_no_cuda(); }
void cv::gpu::MOG_GPU::operator()(const cv::gpu::GpuMat&, cv::gpu::GpuMat&, float, Stream&) { throw_no_cuda(); }
void cv::gpu::MOG_GPU::getBackgroundImage(GpuMat&, Stream&) const { throw_no_cuda(); }
void cv::gpu::MOG_GPU::release() {}

#else

namespace cv { namespace gpu { namespace cudev
{
    namespace mog
    {
        void mog_gpu(PtrStepSzb frame, int cn, PtrStepSzb fgmask, PtrStepSzf weight, PtrStepSzf sortKey, PtrStepSzb mean, PtrStepSzb var,
                     int nmixtures, float varThreshold, float learningRate, float backgroundRatio, float noiseSigma,
                     cudaStream_t stream);
        void getBackgroundImage_gpu(int cn, PtrStepSzf weight, PtrStepSzb mean, PtrStepSzb dst, int nmixtures, float backgroundRatio, cudaStream_t stream);
    }
}}}

namespace mog
{
    const int defaultNMixtures = 5;
    const int defaultHistory = 200;
    const float defaultBackgroundRatio = 0.7f;
    const float defaultVarThreshold = 2.5f * 2.5f;
    const float defaultNoiseSigma = 30.0f * 0.5f;
    const float defaultInitialWeight = 0.05f;
}

cv::gpu::MOG_GPU::MOG_GPU(int nmixtures) :
    frameSize_(0, 0), frameType_(0), nframes_(0)
{
    nmixtures_ = std::min(nmixtures > 0 ? nmixtures : mog::defaultNMixtures, 8);
    history = mog::defaultHistory;
    varThreshold = mog::defaultVarThreshold;
    backgroundRatio = mog::defaultBackgroundRatio;
    noiseSigma = mog::defaultNoiseSigma;
}

void cv::gpu::MOG_GPU::initialize(cv::Size frameSize, int frameType)
{
    CV_Assert(frameType == CV_8UC1 || frameType == CV_8UC3 || frameType == CV_8UC4);

    frameSize_ = frameSize;
    frameType_ = frameType;

    int ch = CV_MAT_CN(frameType);
    int work_ch = ch;

    // for each gaussian mixture of each pixel bg model we store
    // the mixture sort key (w/sum_of_variances), the mixture weight (w),
    // the mean (nchannels values) and
    // the diagonal covariance matrix (another nchannels values)

    weight_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC1);
    sortKey_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC1);
    mean_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC(work_ch));
    var_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC(work_ch));

    weight_.setTo(cv::Scalar::all(0));
    sortKey_.setTo(cv::Scalar::all(0));
    mean_.setTo(cv::Scalar::all(0));
    var_.setTo(cv::Scalar::all(0));

    nframes_ = 0;
}

void cv::gpu::MOG_GPU::operator()(const cv::gpu::GpuMat& frame, cv::gpu::GpuMat& fgmask, float learningRate, Stream& stream)
{
    using namespace cv::gpu::cudev::mog;

    CV_Assert(frame.depth() == CV_8U);

    int ch = frame.channels();
    int work_ch = ch;

    if (nframes_ == 0 || learningRate >= 1.0 || frame.size() != frameSize_ || work_ch != mean_.channels())
        initialize(frame.size(), frame.type());

    fgmask.create(frameSize_, CV_8UC1);

    ++nframes_;
    learningRate = learningRate >= 0.0f && nframes_ > 1 ? learningRate : 1.0f / std::min(nframes_, history);
    CV_Assert(learningRate >= 0.0f);

    mog_gpu(frame, ch, fgmask, weight_, sortKey_, mean_, var_, nmixtures_,
            varThreshold, learningRate, backgroundRatio, noiseSigma,
            StreamAccessor::getStream(stream));
}

void cv::gpu::MOG_GPU::getBackgroundImage(GpuMat& backgroundImage, Stream& stream) const
{
    using namespace cv::gpu::cudev::mog;

    backgroundImage.create(frameSize_, frameType_);

    getBackgroundImage_gpu(backgroundImage.channels(), weight_, mean_, backgroundImage, nmixtures_, backgroundRatio, StreamAccessor::getStream(stream));
}

void cv::gpu::MOG_GPU::release()
{
    frameSize_ = Size(0, 0);
    frameType_ = 0;
    nframes_ = 0;

    weight_.release();
    sortKey_.release();
    mean_.release();
    var_.release();
}

#endif
