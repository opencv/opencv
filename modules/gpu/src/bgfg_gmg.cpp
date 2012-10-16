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

cv::gpu::GMG_GPU::GMG_GPU() { throw_nogpu(); }
void cv::gpu::GMG_GPU::initialize(cv::Size, float, float) { throw_nogpu(); }
void cv::gpu::GMG_GPU::operator ()(const cv::gpu::GpuMat&, cv::gpu::GpuMat&, float, cv::gpu::Stream&) { throw_nogpu(); }
void cv::gpu::GMG_GPU::release() {}

#else

namespace cv { namespace gpu { namespace device {
    namespace bgfg_gmg
    {
        void loadConstants(int width, int height, float minVal, float maxVal, int quantizationLevels, float backgroundPrior,
                           float decisionThreshold, int maxFeatures, int numInitializationFrames);

        template <typename SrcT>
        void update_gpu(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures,
                        int frameNum,  float learningRate, bool updateBackgroundModel, cudaStream_t stream);
    }
}}}

cv::gpu::GMG_GPU::GMG_GPU()
{
    maxFeatures = 64;
    learningRate = 0.025f;
    numInitializationFrames = 120;
    quantizationLevels = 16;
    backgroundPrior = 0.8f;
    decisionThreshold = 0.8f;
    smoothingRadius = 7;
    updateBackgroundModel = true;
}

void cv::gpu::GMG_GPU::initialize(cv::Size frameSize, float min, float max)
{
    using namespace cv::gpu::device::bgfg_gmg;

    CV_Assert(min < max);
    CV_Assert(maxFeatures > 0);
    CV_Assert(learningRate >= 0.0f && learningRate <= 1.0f);
    CV_Assert(numInitializationFrames >= 1);
    CV_Assert(quantizationLevels >= 1 && quantizationLevels <= 255);
    CV_Assert(backgroundPrior >= 0.0f && backgroundPrior <= 1.0f);

    minVal_ = min;
    maxVal_ = max;

    frameSize_ = frameSize;

    frameNum_ = 0;

    nfeatures_.create(frameSize_, CV_32SC1);
    colors_.create(maxFeatures * frameSize_.height, frameSize_.width, CV_32SC1);
    weights_.create(maxFeatures * frameSize_.height, frameSize_.width, CV_32FC1);

    nfeatures_.setTo(cv::Scalar::all(0));

    if (smoothingRadius > 0)
        boxFilter_ = cv::gpu::createBoxFilter_GPU(CV_8UC1, CV_8UC1, cv::Size(smoothingRadius, smoothingRadius));

    loadConstants(frameSize_.width, frameSize_.height, minVal_, maxVal_, quantizationLevels, backgroundPrior, decisionThreshold, maxFeatures, numInitializationFrames);
}

void cv::gpu::GMG_GPU::operator ()(const cv::gpu::GpuMat& frame, cv::gpu::GpuMat& fgmask, float newLearningRate, cv::gpu::Stream& stream)
{
    using namespace cv::gpu::device::bgfg_gmg;

    typedef void (*func_t)(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures,
                           int frameNum, float learningRate, bool updateBackgroundModel, cudaStream_t stream);
    static const func_t funcs[6][4] =
    {
        {update_gpu<uchar>, 0, update_gpu<uchar3>, update_gpu<uchar4>},
        {0,0,0,0},
        {update_gpu<ushort>, 0, update_gpu<ushort3>, update_gpu<ushort4>},
        {0,0,0,0},
        {0,0,0,0},
        {update_gpu<float>, 0, update_gpu<float3>, update_gpu<float4>}
    };

    CV_Assert(frame.depth() == CV_8U || frame.depth() == CV_16U || frame.depth() == CV_32F);
    CV_Assert(frame.channels() == 1 || frame.channels() == 3 || frame.channels() == 4);

    if (newLearningRate != -1.0f)
    {
        CV_Assert(newLearningRate >= 0.0f && newLearningRate <= 1.0f);
        learningRate = newLearningRate;
    }

    if (frame.size() != frameSize_)
        initialize(frame.size(), 0.0f, frame.depth() == CV_8U ? 255.0f : frame.depth() == CV_16U ? std::numeric_limits<ushort>::max() : 1.0f);

    fgmask.create(frameSize_, CV_8UC1);
    if (stream)
        stream.enqueueMemSet(fgmask, cv::Scalar::all(0));
    else
        fgmask.setTo(cv::Scalar::all(0));

    funcs[frame.depth()][frame.channels() - 1](frame, fgmask, colors_, weights_, nfeatures_, frameNum_, learningRate, updateBackgroundModel, cv::gpu::StreamAccessor::getStream(stream));

    // medianBlur
    if (smoothingRadius > 0)
    {
        boxFilter_->apply(fgmask, buf_, cv::Rect(0,0,-1,-1), stream);
        int minCount = (smoothingRadius * smoothingRadius + 1) / 2;
        double thresh = 255.0 * minCount / (smoothingRadius * smoothingRadius);
        cv::gpu::threshold(buf_, fgmask, thresh, 255.0, cv::THRESH_BINARY, stream);
    }

    // keep track of how many frames we have processed
    ++frameNum_;
}

void cv::gpu::GMG_GPU::release()
{
    frameSize_ = Size();

    nfeatures_.release();
    colors_.release();
    weights_.release();
    boxFilter_.release();
    buf_.release();
}

#endif
