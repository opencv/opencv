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

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

Ptr<gpu::BackgroundSubtractorMOG> cv::gpu::createBackgroundSubtractorMOG(int, int, double, double)  { throw_no_cuda(); return Ptr<gpu::BackgroundSubtractorMOG>(); }

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

namespace
{
    const int defaultNMixtures = 5;
    const int defaultHistory = 200;
    const float defaultBackgroundRatio = 0.7f;
    const float defaultVarThreshold = 2.5f * 2.5f;
    const float defaultNoiseSigma = 30.0f * 0.5f;
    const float defaultInitialWeight = 0.05f;

    class MOGImpl : public gpu::BackgroundSubtractorMOG
    {
    public:
        MOGImpl(int history, int nmixtures, double backgroundRatio, double noiseSigma);

        void apply(InputArray image, OutputArray fgmask, double learningRate=-1);
        void apply(InputArray image, OutputArray fgmask, double learningRate, Stream& stream);

        void getBackgroundImage(OutputArray backgroundImage) const;
        void getBackgroundImage(OutputArray backgroundImage, Stream& stream) const;

        int getHistory() const { return history_; }
        void setHistory(int nframes) { history_ = nframes; }

        int getNMixtures() const { return nmixtures_; }
        void setNMixtures(int nmix) { nmixtures_ = nmix; }

        double getBackgroundRatio() const { return backgroundRatio_; }
        void setBackgroundRatio(double backgroundRatio) { backgroundRatio_ = (float) backgroundRatio; }

        double getNoiseSigma() const { return noiseSigma_; }
        void setNoiseSigma(double noiseSigma) { noiseSigma_ = (float) noiseSigma; }

    private:
        //! re-initiaization method
        void initialize(Size frameSize, int frameType);

        int history_;
        int nmixtures_;
        float backgroundRatio_;
        float noiseSigma_;

        float varThreshold_;

        Size frameSize_;
        int frameType_;
        int nframes_;

        GpuMat weight_;
        GpuMat sortKey_;
        GpuMat mean_;
        GpuMat var_;
    };

    MOGImpl::MOGImpl(int history, int nmixtures, double backgroundRatio, double noiseSigma) :
        frameSize_(0, 0), frameType_(0), nframes_(0)
    {
        history_ = history > 0 ? history : defaultHistory;
        nmixtures_ = std::min(nmixtures > 0 ? nmixtures : defaultNMixtures, 8);
        backgroundRatio_ = backgroundRatio > 0 ? (float) backgroundRatio : defaultBackgroundRatio;
        noiseSigma_ = noiseSigma > 0 ? (float) noiseSigma : defaultNoiseSigma;

        varThreshold_ = defaultVarThreshold;
    }

    void MOGImpl::apply(InputArray image, OutputArray fgmask, double learningRate)
    {
        apply(image, fgmask, learningRate, Stream::Null());
    }

    void MOGImpl::apply(InputArray _frame, OutputArray _fgmask, double learningRate, Stream& stream)
    {
        using namespace cv::gpu::cudev::mog;

        GpuMat frame = _frame.getGpuMat();

        CV_Assert( frame.depth() == CV_8U );

        int ch = frame.channels();
        int work_ch = ch;

        if (nframes_ == 0 || learningRate >= 1.0 || frame.size() != frameSize_ || work_ch != mean_.channels())
            initialize(frame.size(), frame.type());

        _fgmask.create(frameSize_, CV_8UC1);
        GpuMat fgmask = _fgmask.getGpuMat();

        ++nframes_;
        learningRate = learningRate >= 0 && nframes_ > 1 ? learningRate : 1.0 / std::min(nframes_, history_);
        CV_Assert( learningRate >= 0 );

        mog_gpu(frame, ch, fgmask, weight_, sortKey_, mean_, var_, nmixtures_,
                varThreshold_, (float) learningRate, backgroundRatio_, noiseSigma_,
                StreamAccessor::getStream(stream));
    }

    void MOGImpl::getBackgroundImage(OutputArray backgroundImage) const
    {
        getBackgroundImage(backgroundImage, Stream::Null());
    }

    void MOGImpl::getBackgroundImage(OutputArray _backgroundImage, Stream& stream) const
    {
        using namespace cv::gpu::cudev::mog;

        _backgroundImage.create(frameSize_, frameType_);
        GpuMat backgroundImage = _backgroundImage.getGpuMat();

        getBackgroundImage_gpu(backgroundImage.channels(), weight_, mean_, backgroundImage, nmixtures_, backgroundRatio_, StreamAccessor::getStream(stream));
    }

    void MOGImpl::initialize(Size frameSize, int frameType)
    {
        CV_Assert( frameType == CV_8UC1 || frameType == CV_8UC3 || frameType == CV_8UC4 );

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
}

Ptr<gpu::BackgroundSubtractorMOG> cv::gpu::createBackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio, double noiseSigma)
{
    return new MOGImpl(history, nmixtures, backgroundRatio, noiseSigma);
}

#endif
