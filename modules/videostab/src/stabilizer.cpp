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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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
#include "opencv2/videostab/stabilizer.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

StabilizerBase::StabilizerBase()
{
    setLog(new NullLog());
    setFrameSource(new NullFrameSource());
    setMotionEstimator(new PyrLkRobustMotionEstimator());
    setDeblurer(new NullDeblurer());
    setInpainter(new NullInpainter());
    setRadius(15);
    setTrimRatio(0);
    setCorrectionForInclusion(false);
    setBorderMode(BORDER_REPLICATE);
}


void StabilizerBase::setUp(int cacheSize, const Mat &frame)
{
    InpainterBase *inpainter = static_cast<InpainterBase*>(inpainter_);
    doInpainting_ = dynamic_cast<NullInpainter*>(inpainter) == 0;
    if (doInpainting_)
    {
        inpainter_->setRadius(radius_);
        inpainter_->setFrames(frames_);
        inpainter_->setMotions(motions_);
        inpainter_->setStabilizedFrames(stabilizedFrames_);
        inpainter_->setStabilizationMotions(stabilizationMotions_);
        inpainter_->update();
    }

    DeblurerBase *deblurer = static_cast<DeblurerBase*>(deblurer_);
    doDeblurring_ = dynamic_cast<NullDeblurer*>(deblurer) == 0;
    if (doDeblurring_)
    {
        blurrinessRates_.resize(cacheSize);
        float blurriness = calcBlurriness(frame);
        for (int i  = -radius_; i <= 0; ++i)
            at(i, blurrinessRates_) = blurriness;
        deblurer_->setRadius(radius_);
        deblurer_->setFrames(frames_);
        deblurer_->setMotions(motions_);
        deblurer_->setBlurrinessRates(blurrinessRates_);
        deblurer_->update();
    }

    log_->print("processing frames");
}


Mat StabilizerBase::nextStabilizedFrame()
{
    if (curStabilizedPos_ == curPos_ && curStabilizedPos_ != -1)
        return Mat(); // we've processed all frames already

    bool processed;
    do processed = doOneIteration();
    while (processed && curStabilizedPos_ == -1);

    if (curStabilizedPos_ == -1)
        return Mat(); // frame source is empty

    const Mat &stabilizedFrame = at(curStabilizedPos_, stabilizedFrames_);
    int dx = static_cast<int>(floor(trimRatio_ * stabilizedFrame.cols));
    int dy = static_cast<int>(floor(trimRatio_ * stabilizedFrame.rows));
    return stabilizedFrame(Rect(dx, dy, stabilizedFrame.cols - 2*dx, stabilizedFrame.rows - 2*dy));
}


bool StabilizerBase::doOneIteration()
{
    Mat frame = frameSource_->nextFrame();
    if (!frame.empty())
    {
        curPos_++;

        if (curPos_ > 0)
        {
            at(curPos_, frames_) = frame;

            if (doDeblurring_)
                at(curPos_, blurrinessRates_) = calcBlurriness(frame);

            estimateMotion();

            if (curPos_ >= radius_)
            {
                curStabilizedPos_ = curPos_ - radius_;
                stabilizeFrame();
            }
        }
        else
            setUp(frame);

        log_->print(".");
        return true;
    }

    if (curStabilizedPos_ < curPos_)
    {
        curStabilizedPos_++;
        at(curStabilizedPos_ + radius_, frames_) = at(curPos_, frames_);
        at(curStabilizedPos_ + radius_ - 1, motions_) = at(curPos_ - 1, motions_);
        stabilizeFrame();

        log_->print(".");
        return true;
    }

    return false;
}


void StabilizerBase::stabilizeFrame(const Mat &stabilizationMotion)
{
    Mat stabilizationMotion_;
    if (doCorrectionForInclusion_)
        stabilizationMotion_ = ensureInclusionConstraint(stabilizationMotion, frameSize_, trimRatio_);
    else
        stabilizationMotion_ = stabilizationMotion.clone();

    at(curStabilizedPos_, stabilizationMotions_) = stabilizationMotion_;

    if (doDeblurring_)
    {
        at(curStabilizedPos_, frames_).copyTo(preProcessedFrame_);
        deblurer_->deblur(curStabilizedPos_, preProcessedFrame_);
    }
    else
        preProcessedFrame_ = at(curStabilizedPos_, frames_);

    // apply stabilization transformation
    warpAffine(
            preProcessedFrame_, at(curStabilizedPos_, stabilizedFrames_),
            stabilizationMotion_(Rect(0,0,3,2)), frameSize_, INTER_LINEAR, borderMode_);

    if (doInpainting_)
    {
        warpAffine(
                frameMask_, at(curStabilizedPos_, stabilizedMasks_),
                stabilizationMotion_(Rect(0,0,3,2)), frameSize_, INTER_NEAREST);

        erode(at(curStabilizedPos_, stabilizedMasks_), at(curStabilizedPos_, stabilizedMasks_),
              Mat());

        at(curStabilizedPos_, stabilizedMasks_).copyTo(inpaintingMask_);

        inpainter_->inpaint(
            curStabilizedPos_, at(curStabilizedPos_, stabilizedFrames_), inpaintingMask_);
    }
}


OnePassStabilizer::OnePassStabilizer()
{
    setMotionFilter(new GaussianMotionFilter());
    resetImpl();
}


void OnePassStabilizer::resetImpl()
{
    curPos_ = -1;
    curStabilizedPos_ = -1;
    frames_.clear();
    motions_.clear();
    stabilizedFrames_.clear();
    stabilizationMotions_.clear();
    doDeblurring_ = false;
    doInpainting_ = false;
}


void OnePassStabilizer::setUp(Mat &firstFrame)
{
    frameSize_ = firstFrame.size();
    frameMask_.create(frameSize_, CV_8U);
    frameMask_.setTo(255);

    int cacheSize = 2*radius_ + 1;

    frames_.resize(cacheSize);
    stabilizedFrames_.resize(cacheSize);
    stabilizedMasks_.resize(cacheSize);
    motions_.resize(cacheSize);
    stabilizationMotions_.resize(cacheSize);

    for (int i = -radius_; i < 0; ++i)
    {
        at(i, motions_) = Mat::eye(3, 3, CV_32F);
        at(i, frames_) = firstFrame;
    }

    at(0, frames_) = firstFrame;

    motionFilter_->setRadius(radius_);
    motionFilter_->update();

    StabilizerBase::setUp(cacheSize, firstFrame);
}


void OnePassStabilizer::estimateMotion()
{
    at(curPos_ - 1, motions_) = motionEstimator_->estimate(
            at(curPos_ - 1, frames_), at(curPos_, frames_));
}


void OnePassStabilizer::stabilizeFrame()
{
    Mat stabilizationMotion = motionFilter_->stabilize(curStabilizedPos_, &motions_[0], motions_.size());
    StabilizerBase::stabilizeFrame(stabilizationMotion);
}


TwoPassStabilizer::TwoPassStabilizer()
{
    setMotionStabilizer(new GaussianMotionFilter());
    setEstimateTrimRatio(true);
    resetImpl();
}


Mat TwoPassStabilizer::nextFrame()
{
    runPrePassIfNecessary();
    return StabilizerBase::nextStabilizedFrame();
}


void TwoPassStabilizer::resetImpl()
{
    isPrePassDone_ = false;
    frameCount_ = 0;
    curPos_ = -1;
    curStabilizedPos_ = -1;
    frames_.clear();
    motions_.clear();
    stabilizedFrames_.clear();
    stabilizationMotions_.clear();
    doDeblurring_ = false;
    doInpainting_ = false;
}


void TwoPassStabilizer::runPrePassIfNecessary()
{
    if (!isPrePassDone_)
    {
        log_->print("first pass: estimating motions");

        Mat prevFrame, frame;

        while (!(frame = frameSource_->nextFrame()).empty())
        {
            if (frameCount_ > 0)
                motions_.push_back(motionEstimator_->estimate(prevFrame, frame));
            else
            {
                frameSize_ = frame.size();
                frameMask_.create(frameSize_, CV_8U);
                frameMask_.setTo(255);
            }

            prevFrame = frame;
            frameCount_++;

            log_->print(".");
        }

        for (int i = 0; i < radius_; ++i)
            motions_.push_back(Mat::eye(3, 3, CV_32F));
        log_->print("\n");

        IMotionStabilizer *motionStabilizer = static_cast<IMotionStabilizer*>(motionStabilizer_);
        MotionFilterBase *motionFilterBase = dynamic_cast<MotionFilterBase*>(motionStabilizer);
        if (motionFilterBase)
        {
            motionFilterBase->setRadius(radius_);
            motionFilterBase->update();
        }

        stabilizationMotions_.resize(frameCount_);
        motionStabilizer_->stabilize(&motions_[0], frameCount_, &stabilizationMotions_[0]);

        if (mustEstTrimRatio_)
        {
            trimRatio_ = 0;
            for (int i = 0; i < frameCount_; ++i)
            {
                Mat S = stabilizationMotions_[i];
                trimRatio_ = std::max(trimRatio_, estimateOptimalTrimRatio(S, frameSize_));
            }
            log_->print("estimated trim ratio: %f\n", static_cast<double>(trimRatio_));
        }

        isPrePassDone_ = true;
        frameSource_->reset();
    }
}


void TwoPassStabilizer::setUp(Mat &firstFrame)
{
    int cacheSize = 2*radius_ + 1;

    frames_.resize(cacheSize);
    stabilizedFrames_.resize(cacheSize);
    stabilizedMasks_.resize(cacheSize);

    for (int i = -radius_; i <= 0; ++i)
        at(i, frames_) = firstFrame;

    StabilizerBase::setUp(cacheSize, firstFrame);
}


void TwoPassStabilizer::stabilizeFrame()
{
    StabilizerBase::stabilizeFrame(stabilizationMotions_[curStabilizedPos_]);
}

} // namespace videostab
} // namespace cv
