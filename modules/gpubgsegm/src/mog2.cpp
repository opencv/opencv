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

Ptr<gpu::BackgroundSubtractorMOG2> cv::gpu::createBackgroundSubtractorMOG2(int, double, bool) { throw_no_cuda(); return Ptr<gpu::BackgroundSubtractorMOG2>(); }

#else

namespace cv { namespace gpu { namespace cudev
{
    namespace mog2
    {
        void loadConstants(int nmixtures, float Tb, float TB, float Tg, float varInit, float varMin, float varMax, float tau, unsigned char shadowVal);
        void mog2_gpu(PtrStepSzb frame, int cn, PtrStepSzb fgmask, PtrStepSzb modesUsed, PtrStepSzf weight, PtrStepSzf variance, PtrStepSzb mean, float alphaT, float prune, bool detectShadows, cudaStream_t stream);
        void getBackgroundImage2_gpu(int cn, PtrStepSzb modesUsed, PtrStepSzf weight, PtrStepSzb mean, PtrStepSzb dst, cudaStream_t stream);
    }
}}}

namespace
{
    // default parameters of gaussian background detection algorithm
    const int defaultHistory = 500; // Learning rate; alpha = 1/defaultHistory2
    const float defaultVarThreshold = 4.0f * 4.0f;
    const int defaultNMixtures = 5; // maximal number of Gaussians in mixture
    const float defaultBackgroundRatio = 0.9f; // threshold sum of weights for background test
    const float defaultVarThresholdGen = 3.0f * 3.0f;
    const float defaultVarInit = 15.0f; // initial variance for new components
    const float defaultVarMax = 5.0f * defaultVarInit;
    const float defaultVarMin = 4.0f;

    // additional parameters
    const float defaultCT = 0.05f; // complexity reduction prior constant 0 - no reduction of number of components
    const unsigned char defaultShadowValue = 127; // value to use in the segmentation mask for shadows, set 0 not to do shadow detection
    const float defaultShadowThreshold = 0.5f; // Tau - shadow threshold, see the paper for explanation

    class MOG2Impl : public gpu::BackgroundSubtractorMOG2
    {
    public:
        MOG2Impl(int history, double varThreshold, bool detectShadows);

        void apply(InputArray image, OutputArray fgmask, double learningRate=-1);
        void apply(InputArray image, OutputArray fgmask, double learningRate, Stream& stream);

        void getBackgroundImage(OutputArray backgroundImage) const;
        void getBackgroundImage(OutputArray backgroundImage, Stream& stream) const;

        int getHistory() const { return history_; }
        void setHistory(int history) { history_ = history; }

        int getNMixtures() const { return nmixtures_; }
        void setNMixtures(int nmixtures) { nmixtures_ = nmixtures; }

        double getBackgroundRatio() const { return backgroundRatio_; }
        void setBackgroundRatio(double ratio) { backgroundRatio_ = (float) ratio; }

        double getVarThreshold() const { return varThreshold_; }
        void setVarThreshold(double varThreshold) { varThreshold_ = (float) varThreshold; }

        double getVarThresholdGen() const { return varThresholdGen_; }
        void setVarThresholdGen(double varThresholdGen) { varThresholdGen_ = (float) varThresholdGen; }

        double getVarInit() const { return varInit_; }
        void setVarInit(double varInit) { varInit_ = (float) varInit; }

        double getVarMin() const { return varMin_; }
        void setVarMin(double varMin) { varMin_ = (float) varMin; }

        double getVarMax() const { return varMax_; }
        void setVarMax(double varMax) { varMax_ = (float) varMax; }

        double getComplexityReductionThreshold() const { return ct_; }
        void setComplexityReductionThreshold(double ct) { ct_ = (float) ct; }

        bool getDetectShadows() const { return detectShadows_; }
        void setDetectShadows(bool detectShadows) { detectShadows_ = detectShadows; }

        int getShadowValue() const { return shadowValue_; }
        void setShadowValue(int value) { shadowValue_ = (uchar) value; }

        double getShadowThreshold() const { return shadowThreshold_; }
        void setShadowThreshold(double threshold) { shadowThreshold_ = (float) threshold; }

    private:
        void initialize(Size frameSize, int frameType);

        int history_;
        int nmixtures_;
        float backgroundRatio_;
        float varThreshold_;
        float varThresholdGen_;
        float varInit_;
        float varMin_;
        float varMax_;
        float ct_;
        bool detectShadows_;
        uchar shadowValue_;
        float shadowThreshold_;

        Size frameSize_;
        int frameType_;
        int nframes_;

        GpuMat weight_;
        GpuMat variance_;
        GpuMat mean_;

        //keep track of number of modes per pixel
        GpuMat bgmodelUsedModes_;
    };

    MOG2Impl::MOG2Impl(int history, double varThreshold, bool detectShadows) :
        frameSize_(0, 0), frameType_(0), nframes_(0)
    {
        history_ = history > 0 ? history : defaultHistory;
        varThreshold_ = varThreshold > 0 ? (float) varThreshold : defaultVarThreshold;
        detectShadows_ = detectShadows;

        nmixtures_ = defaultNMixtures;
        backgroundRatio_ = defaultBackgroundRatio;
        varInit_ = defaultVarInit;
        varMax_ = defaultVarMax;
        varMin_ = defaultVarMin;
        varThresholdGen_ = defaultVarThresholdGen;
        ct_ = defaultCT;
        shadowValue_ =  defaultShadowValue;
        shadowThreshold_ = defaultShadowThreshold;
    }

    void MOG2Impl::apply(InputArray image, OutputArray fgmask, double learningRate)
    {
        apply(image, fgmask, learningRate, Stream::Null());
    }

    void MOG2Impl::apply(InputArray _frame, OutputArray _fgmask, double learningRate, Stream& stream)
    {
        using namespace cv::gpu::cudev::mog2;

        GpuMat frame = _frame.getGpuMat();

        int ch = frame.channels();
        int work_ch = ch;

        if (nframes_ == 0 || learningRate >= 1.0 || frame.size() != frameSize_ || work_ch != mean_.channels())
            initialize(frame.size(), frame.type());

        _fgmask.create(frameSize_, CV_8UC1);
        GpuMat fgmask = _fgmask.getGpuMat();

        fgmask.setTo(Scalar::all(0), stream);

        ++nframes_;
        learningRate = learningRate >= 0 && nframes_ > 1 ? learningRate : 1.0 / std::min(2 * nframes_, history_);
        CV_Assert( learningRate >= 0 );

        mog2_gpu(frame, frame.channels(), fgmask, bgmodelUsedModes_, weight_, variance_, mean_,
                 (float) learningRate, static_cast<float>(-learningRate * ct_), detectShadows_, StreamAccessor::getStream(stream));
    }

    void MOG2Impl::getBackgroundImage(OutputArray backgroundImage) const
    {
        getBackgroundImage(backgroundImage, Stream::Null());
    }

    void MOG2Impl::getBackgroundImage(OutputArray _backgroundImage, Stream& stream) const
    {
        using namespace cv::gpu::cudev::mog2;

        _backgroundImage.create(frameSize_, frameType_);
        GpuMat backgroundImage = _backgroundImage.getGpuMat();

        getBackgroundImage2_gpu(backgroundImage.channels(), bgmodelUsedModes_, weight_, mean_, backgroundImage, StreamAccessor::getStream(stream));
    }

    void MOG2Impl::initialize(cv::Size frameSize, int frameType)
    {
        using namespace cv::gpu::cudev::mog2;

        CV_Assert( frameType == CV_8UC1 || frameType == CV_8UC3 || frameType == CV_8UC4 );

        frameSize_ = frameSize;
        frameType_ = frameType;
        nframes_ = 0;

        int ch = CV_MAT_CN(frameType);
        int work_ch = ch;

        // for each gaussian mixture of each pixel bg model we store ...
        // the mixture weight (w),
        // the mean (nchannels values) and
        // the covariance
        weight_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC1);
        variance_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC1);
        mean_.create(frameSize.height * nmixtures_, frameSize_.width, CV_32FC(work_ch));

        //make the array for keeping track of the used modes per pixel - all zeros at start
        bgmodelUsedModes_.create(frameSize_, CV_8UC1);
        bgmodelUsedModes_.setTo(Scalar::all(0));

        loadConstants(nmixtures_, varThreshold_, backgroundRatio_, varThresholdGen_, varInit_, varMin_, varMax_, shadowThreshold_, shadowValue_);
    }
}

Ptr<gpu::BackgroundSubtractorMOG2> cv::gpu::createBackgroundSubtractorMOG2(int history, double varThreshold, bool detectShadows)
{
    return new MOG2Impl(history, varThreshold, detectShadows);
}

#endif
