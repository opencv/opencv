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

#if defined(HAVE_OPENCV_GPU)

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

cv::gpu::VIBE_GPU::VIBE_GPU(unsigned long) { throw_nogpu(); }
void cv::gpu::VIBE_GPU::initialize(const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::VIBE_GPU::operator()(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::VIBE_GPU::release() {}

#else

namespace cv { namespace gpu { namespace cuda
{
    namespace vibe
    {
        void loadConstants(int nbSamples, int reqMatches, int radius, int subsamplingFactor);

        void init_gpu(PtrStepSzb frame, int cn, PtrStepSzb samples, PtrStepSz<unsigned int> randStates, cudaStream_t stream);

        void update_gpu(PtrStepSzb frame, int cn, PtrStepSzb fgmask, PtrStepSzb samples, PtrStepSz<unsigned int> randStates, cudaStream_t stream);
    }
}}}

namespace
{
    const int defaultNbSamples = 20;
    const int defaultReqMatches = 2;
    const int defaultRadius = 20;
    const int defaultSubsamplingFactor = 16;
}

cv::gpu::VIBE_GPU::VIBE_GPU(unsigned long rngSeed) :
    frameSize_(0, 0), rngSeed_(rngSeed)
{
    nbSamples = defaultNbSamples;
    reqMatches = defaultReqMatches;
    radius = defaultRadius;
    subsamplingFactor = defaultSubsamplingFactor;
}

void cv::gpu::VIBE_GPU::initialize(const GpuMat& firstFrame, Stream& s)
{
    using namespace cv::gpu::cuda::vibe;

    CV_Assert(firstFrame.type() == CV_8UC1 || firstFrame.type() == CV_8UC3 || firstFrame.type() == CV_8UC4);

    cudaStream_t stream = StreamAccessor::getStream(s);

    loadConstants(nbSamples, reqMatches, radius, subsamplingFactor);

    frameSize_ = firstFrame.size();

    if (randStates_.size() != frameSize_)
    {
        cv::RNG rng(rngSeed_);
        cv::Mat h_randStates(frameSize_, CV_8UC4);
        rng.fill(h_randStates, cv::RNG::UNIFORM, 0, 255);
        randStates_.upload(h_randStates);
    }

    int ch = firstFrame.channels();
    int sample_ch = ch == 1 ? 1 : 4;

    samples_.create(nbSamples * frameSize_.height, frameSize_.width, CV_8UC(sample_ch));

    init_gpu(firstFrame, ch, samples_, randStates_, stream);
}

void cv::gpu::VIBE_GPU::operator()(const GpuMat& frame, GpuMat& fgmask, Stream& s)
{
    using namespace cv::gpu::cuda::vibe;

    CV_Assert(frame.depth() == CV_8U);

    int ch = frame.channels();
    int sample_ch = ch == 1 ? 1 : 4;

    if (frame.size() != frameSize_ || sample_ch != samples_.channels())
        initialize(frame);

    fgmask.create(frameSize_, CV_8UC1);

    update_gpu(frame, ch, fgmask, samples_, randStates_, StreamAccessor::getStream(s));
}

void cv::gpu::VIBE_GPU::release()
{
    frameSize_ = Size(0, 0);

    randStates_.release();

    samples_.release();
}

#endif

#endif // defined(HAVE_OPENCV_GPU)
