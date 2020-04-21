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
#include "cuda/mog2.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cuda::device::mog2;

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

Ptr<cuda::BackgroundSubtractorMOG2> cv::cuda::createBackgroundSubtractorMOG2(int, double, bool)
{
    throw_no_cuda();
    return Ptr<cuda::BackgroundSubtractorMOG2>();
}

#else

namespace
{
// default parameters of gaussian background detection algorithm
const int defaultHistory = 500; // Learning rate; alpha = 1/defaultHistory2
const float defaultVarThreshold = 4.0f * 4.0f;
const int defaultNMixtures = 5;            // maximal number of Gaussians in mixture
const float defaultBackgroundRatio = 0.9f; // threshold sum of weights for background test
const float defaultVarThresholdGen = 3.0f * 3.0f;
const float defaultVarInit = 15.0f; // initial variance for new components
const float defaultVarMax = 5.0f * defaultVarInit;
const float defaultVarMin = 4.0f;

// additional parameters
const float defaultCT = 0.05f;                // complexity reduction prior constant 0 - no reduction of number of components
const unsigned char defaultShadowValue = 127; // value to use in the segmentation mask for shadows, set 0 not to do shadow detection
const float defaultShadowThreshold = 0.5f;    // Tau - shadow threshold, see the paper for explanation

class MOG2Impl CV_FINAL : public cuda::BackgroundSubtractorMOG2
{
public:
    MOG2Impl(int history, double varThreshold, bool detectShadows);
    ~MOG2Impl();

    void apply(InputArray image, OutputArray fgmask, double learningRate = -1) CV_OVERRIDE;
    void apply(InputArray image, OutputArray fgmask, double learningRate, Stream &stream) CV_OVERRIDE;

    void getBackgroundImage(OutputArray backgroundImage) const CV_OVERRIDE;
    void getBackgroundImage(OutputArray backgroundImage, Stream &stream) const CV_OVERRIDE;

    int getHistory() const CV_OVERRIDE { return history_; }
    void setHistory(int history) CV_OVERRIDE { history_ = history; }

    int getNMixtures() const CV_OVERRIDE { return constantsHost_.nmixtures_; }
    void setNMixtures(int nmixtures) CV_OVERRIDE { constantsHost_.nmixtures_ = nmixtures; }

    double getBackgroundRatio() const CV_OVERRIDE { return constantsHost_.TB_; }
    void setBackgroundRatio(double ratio) CV_OVERRIDE { constantsHost_.TB_ = (float)ratio; }

    double getVarThreshold() const CV_OVERRIDE { return constantsHost_.Tb_; }
    void setVarThreshold(double varThreshold) CV_OVERRIDE { constantsHost_.Tb_ = (float)varThreshold; }

    double getVarThresholdGen() const CV_OVERRIDE { return constantsHost_.Tg_; }
    void setVarThresholdGen(double varThresholdGen) CV_OVERRIDE { constantsHost_.Tg_ = (float)varThresholdGen; }

    double getVarInit() const CV_OVERRIDE { return constantsHost_.varInit_; }
    void setVarInit(double varInit) CV_OVERRIDE { constantsHost_.varInit_ = (float)varInit; }

    double getVarMin() const CV_OVERRIDE { return constantsHost_.varMin_; }
    void setVarMin(double varMin) CV_OVERRIDE { constantsHost_.varMin_ = ::fminf((float)varMin, constantsHost_.varMax_); }

    double getVarMax() const CV_OVERRIDE { return constantsHost_.varMax_; }
    void setVarMax(double varMax) CV_OVERRIDE { constantsHost_.varMax_ = ::fmaxf(constantsHost_.varMin_, (float)varMax); }

    double getComplexityReductionThreshold() const CV_OVERRIDE { return ct_; }
    void setComplexityReductionThreshold(double ct) CV_OVERRIDE { ct_ = (float)ct; }

    bool getDetectShadows() const CV_OVERRIDE { return detectShadows_; }
    void setDetectShadows(bool detectShadows) CV_OVERRIDE { detectShadows_ = detectShadows; }

    int getShadowValue() const CV_OVERRIDE { return constantsHost_.shadowVal_; }
    void setShadowValue(int value) CV_OVERRIDE { constantsHost_.shadowVal_ = (uchar)value; }

    double getShadowThreshold() const CV_OVERRIDE { return constantsHost_.tau_; }
    void setShadowThreshold(double threshold) CV_OVERRIDE { constantsHost_.tau_ = (float)threshold; }

private:
    void initialize(Size frameSize, int frameType, Stream &stream);

    Constants constantsHost_;
    Constants *constantsDevice_;

    int history_;
    float ct_;
    bool detectShadows_;

    Size frameSize_;
    int frameType_;
    int nframes_;

    GpuMat weight_;
    GpuMat variance_;
    GpuMat mean_;

    //keep track of number of modes per pixel
    GpuMat bgmodelUsedModes_;
};

MOG2Impl::MOG2Impl(int history, double varThreshold, bool detectShadows) : frameSize_(0, 0), frameType_(0), nframes_(0)
{
    history_ = history > 0 ? history : defaultHistory;
    detectShadows_ = detectShadows;
    ct_ = defaultCT;

    setNMixtures(defaultNMixtures);
    setBackgroundRatio(defaultBackgroundRatio);
    setVarInit(defaultVarInit);
    setVarMin(defaultVarMin);
    setVarMax(defaultVarMax);
    setVarThreshold(varThreshold > 0 ? (float)varThreshold : defaultVarThreshold);
    setVarThresholdGen(defaultVarThresholdGen);

    setShadowValue(defaultShadowValue);
    setShadowThreshold(defaultShadowThreshold);

    cudaSafeCall(cudaMalloc((void **)&constantsDevice_, sizeof(Constants)));
}

MOG2Impl::~MOG2Impl()
{
    cudaFree(constantsDevice_);
}

void MOG2Impl::apply(InputArray image, OutputArray fgmask, double learningRate)
{
    apply(image, fgmask, learningRate, Stream::Null());
}

void MOG2Impl::apply(InputArray _frame, OutputArray _fgmask, double learningRate, Stream &stream)
{
    using namespace cv::cuda::device::mog2;

    GpuMat frame = _frame.getGpuMat();

    int ch = frame.channels();
    int work_ch = ch;

    if (nframes_ == 0 || learningRate >= 1.0 || frame.size() != frameSize_ || work_ch != mean_.channels())
        initialize(frame.size(), frame.type(), stream);

    _fgmask.create(frameSize_, CV_8UC1);
    GpuMat fgmask = _fgmask.getGpuMat();

    fgmask.setTo(Scalar::all(0), stream);

    ++nframes_;
    learningRate = learningRate >= 0 && nframes_ > 1 ? learningRate : 1.0 / std::min(2 * nframes_, history_);
    CV_Assert(learningRate >= 0);

    mog2_gpu(frame, frame.channels(), fgmask, bgmodelUsedModes_, weight_, variance_, mean_,
             (float)learningRate, static_cast<float>(-learningRate * ct_), detectShadows_, constantsDevice_, StreamAccessor::getStream(stream));
}

void MOG2Impl::getBackgroundImage(OutputArray backgroundImage) const
{
    getBackgroundImage(backgroundImage, Stream::Null());
}

void MOG2Impl::getBackgroundImage(OutputArray _backgroundImage, Stream &stream) const
{
    using namespace cv::cuda::device::mog2;

    _backgroundImage.create(frameSize_, frameType_);
    GpuMat backgroundImage = _backgroundImage.getGpuMat();

    getBackgroundImage2_gpu(backgroundImage.channels(), bgmodelUsedModes_, weight_, mean_, backgroundImage, constantsDevice_, StreamAccessor::getStream(stream));
}

void MOG2Impl::initialize(cv::Size frameSize, int frameType, Stream &stream)
{
    using namespace cv::cuda::device::mog2;

    CV_Assert(frameType == CV_8UC1 || frameType == CV_8UC3 || frameType == CV_8UC4);

    frameSize_ = frameSize;
    frameType_ = frameType;
    nframes_ = 0;

    const int ch = CV_MAT_CN(frameType);
    const int work_ch = ch;

    // for each gaussian mixture of each pixel bg model we store ...
    // the mixture weight (w),
    // the mean (nchannels values) and
    // the covariance
    weight_.create(frameSize.height * getNMixtures(), frameSize_.width, CV_32FC1);
    variance_.create(frameSize.height * getNMixtures(), frameSize_.width, CV_32FC1);
    mean_.create(frameSize.height * getNMixtures(), frameSize_.width, CV_32FC(work_ch));

    //make the array for keeping track of the used modes per pixel - all zeros at start
    bgmodelUsedModes_.create(frameSize_, CV_8UC1);
    bgmodelUsedModes_.setTo(Scalar::all(0));

    cudaSafeCall(cudaMemcpyAsync(constantsDevice_, &constantsHost_, sizeof(Constants), cudaMemcpyHostToDevice, StreamAccessor::getStream(stream)));
}
} // namespace

Ptr<cuda::BackgroundSubtractorMOG2> cv::cuda::createBackgroundSubtractorMOG2(int history, double varThreshold, bool detectShadows)
{
    return makePtr<MOG2Impl>(history, varThreshold, detectShadows);
}

#endif
