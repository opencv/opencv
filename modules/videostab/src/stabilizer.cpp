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
#include "opencv2/videostab/ring_buffer.hpp"

// for debug purposes
#define SAVE_MOTIONS 0

using namespace std;

namespace cv
{
namespace videostab
{

StabilizerBase::StabilizerBase()
{
    setLog(new LogToStdout());
    setFrameSource(new NullFrameSource());
    setMotionEstimator(new PyrLkRobustMotionEstimator());
    setDeblurer(new NullDeblurer());
    setInpainter(new NullInpainter());
    setRadius(15);
    setTrimRatio(0);
    setCorrectionForInclusion(false);
    setBorderMode(BORDER_REPLICATE);
}


void StabilizerBase::reset()
{
    frameSize_ = Size(0, 0);
    frameMask_ = Mat();
    curPos_ = -1;
    curStabilizedPos_ = -1;
    doDeblurring_ = false;
    preProcessedFrame_ = Mat();
    doInpainting_ = false;
    inpaintingMask_ = Mat();
    frames_.clear();
    motions_.clear();
    blurrinessRates_.clear();
    stabilizedFrames_.clear();
    stabilizedMasks_.clear();
    stabilizationMotions_.clear();
    processingStartTime_ = 0;
}


Mat StabilizerBase::nextStabilizedFrame()
{
    // check if we've processed all frames already
    if (curStabilizedPos_ == curPos_ && curStabilizedPos_ != -1)
    {
        logProcessingTime();
        return Mat();
    }

    bool processed;
    do processed = doOneIteration();
    while (processed && curStabilizedPos_ == -1);

    // check if the frame source is empty
    if (curStabilizedPos_ == -1)
    {
        logProcessingTime();
        return Mat();
    }

    return postProcessFrame(at(curStabilizedPos_, stabilizedFrames_));
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

            at(curPos_ - 1, motions_) = estimateMotion();

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
        at(curStabilizedPos_ + radius_ - 1, motions_) = Mat::eye(3, 3, CV_32F);
        stabilizeFrame();

        log_->print(".");
        return true;
    }

    return false;
}


void StabilizerBase::setUp(const Mat &firstFrame)
{
    InpainterBase *inpainter = static_cast<InpainterBase*>(inpainter_);
    doInpainting_ = dynamic_cast<NullInpainter*>(inpainter) == 0;
    if (doInpainting_)
    {
        inpainter_->setMotionModel(motionEstimator_->motionModel());
        inpainter_->setFrames(frames_);
        inpainter_->setMotions(motions_);
        inpainter_->setStabilizedFrames(stabilizedFrames_);
        inpainter_->setStabilizationMotions(stabilizationMotions_);
    }

    DeblurerBase *deblurer = static_cast<DeblurerBase*>(deblurer_);
    doDeblurring_ = dynamic_cast<NullDeblurer*>(deblurer) == 0;
    if (doDeblurring_)
    {
        blurrinessRates_.resize(2*radius_ + 1);
        float blurriness = calcBlurriness(firstFrame);
        for (int i  = -radius_; i <= 0; ++i)
            at(i, blurrinessRates_) = blurriness;
        deblurer_->setFrames(frames_);
        deblurer_->setMotions(motions_);
        deblurer_->setBlurrinessRates(blurrinessRates_);
    }

    log_->print("processing frames");
    processingStartTime_ = clock();
}


void StabilizerBase::stabilizeFrame()
{
    Mat stabilizationMotion = estimateStabilizationMotion();
    if (doCorrectionForInclusion_)
        stabilizationMotion = ensureInclusionConstraint(stabilizationMotion, frameSize_, trimRatio_);

    at(curStabilizedPos_, stabilizationMotions_) = stabilizationMotion;

    if (doDeblurring_)
    {
        at(curStabilizedPos_, frames_).copyTo(preProcessedFrame_);
        deblurer_->deblur(curStabilizedPos_, preProcessedFrame_);
    }
    else
        preProcessedFrame_ = at(curStabilizedPos_, frames_);

    // apply stabilization transformation

    if (motionEstimator_->motionModel() != MM_HOMOGRAPHY)
        warpAffine(
                preProcessedFrame_, at(curStabilizedPos_, stabilizedFrames_),
                stabilizationMotion(Rect(0,0,3,2)), frameSize_, INTER_LINEAR, borderMode_);
    else
        warpPerspective(
                preProcessedFrame_, at(curStabilizedPos_, stabilizedFrames_),
                stabilizationMotion, frameSize_, INTER_LINEAR, borderMode_);

    if (doInpainting_)
    {
        if (motionEstimator_->motionModel() != MM_HOMOGRAPHY)
            warpAffine(
                    frameMask_, at(curStabilizedPos_, stabilizedMasks_),
                    stabilizationMotion(Rect(0,0,3,2)), frameSize_, INTER_NEAREST);
        else
            warpPerspective(
                    frameMask_, at(curStabilizedPos_, stabilizedMasks_),
                    stabilizationMotion, frameSize_, INTER_NEAREST);

        erode(at(curStabilizedPos_, stabilizedMasks_), at(curStabilizedPos_, stabilizedMasks_),
              Mat());

        at(curStabilizedPos_, stabilizedMasks_).copyTo(inpaintingMask_);

        inpainter_->inpaint(
            curStabilizedPos_, at(curStabilizedPos_, stabilizedFrames_), inpaintingMask_);
    }
}


Mat StabilizerBase::postProcessFrame(const Mat &frame)
{
    // trim frame
    int dx = static_cast<int>(floor(trimRatio_ * frame.cols));
    int dy = static_cast<int>(floor(trimRatio_ * frame.rows));
    return frame(Rect(dx, dy, frame.cols - 2*dx, frame.rows - 2*dy));
}


void StabilizerBase::logProcessingTime()
{
    clock_t elapsedTime = clock() - processingStartTime_;
    log_->print("\nprocessing time: %.3f sec\n", static_cast<double>(elapsedTime) / CLOCKS_PER_SEC);
}


OnePassStabilizer::OnePassStabilizer()
{
    setMotionFilter(new GaussianMotionFilter());
    reset();
}


void OnePassStabilizer::reset()
{
    StabilizerBase::reset();
}


void OnePassStabilizer::setUp(const Mat &firstFrame)
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

    StabilizerBase::setUp(firstFrame);
}


Mat OnePassStabilizer::estimateMotion()
{
    return motionEstimator_->estimate(at(curPos_ - 1, frames_), at(curPos_, frames_));
}


Mat OnePassStabilizer::estimateStabilizationMotion()
{
    return motionFilter_->stabilize(curStabilizedPos_, motions_, make_pair(0, curPos_));
}


Mat OnePassStabilizer::postProcessFrame(const Mat &frame)
{
    return StabilizerBase::postProcessFrame(frame);
}


TwoPassStabilizer::TwoPassStabilizer()
{
    setMotionStabilizer(new GaussianMotionFilter());
    setWobbleSuppressor(new NullWobbleSuppressor());
    setEstimateTrimRatio(false);
    reset();
}


void TwoPassStabilizer::reset()
{
    StabilizerBase::reset();
    frameCount_ = 0;
    isPrePassDone_ = false;
    doWobbleSuppression_ = false;
    motions2_.clear();
    suppressedFrame_ = Mat();
}


Mat TwoPassStabilizer::nextFrame()
{
    runPrePassIfNecessary();
    return StabilizerBase::nextStabilizedFrame();
}


#if SAVE_MOTIONS
static void saveMotions(
        int frameCount, const vector<Mat> &motions, const vector<Mat> &stabilizationMotions)
{
    ofstream fm("log_motions.csv");
    for (int i = 0; i < frameCount - 1; ++i)
    {
        Mat_<float> M = at(i, motions);
        fm << M(0,0) << " " << M(0,1) << " " << M(0,2) << " "
           << M(1,0) << " " << M(1,1) << " " << M(1,2) << " "
           << M(2,0) << " " << M(2,1) << " " << M(2,2) << endl;
    }
    ofstream fo("log_orig.csv");
    for (int i = 0; i < frameCount; ++i)
    {
        Mat_<float> M = getMotion(0, i, motions);
        fo << M(0,0) << " " << M(0,1) << " " << M(0,2) << " "
           << M(1,0) << " " << M(1,1) << " " << M(1,2) << " "
           << M(2,0) << " " << M(2,1) << " " << M(2,2) << endl;
    }
    ofstream fs("log_stab.csv");
    for (int i = 0; i < frameCount; ++i)
    {
        Mat_<float> M = stabilizationMotions[i] * getMotion(0, i, motions);
        fs << M(0,0) << " " << M(0,1) << " " << M(0,2) << " "
           << M(1,0) << " " << M(1,1) << " " << M(1,2) << " "
           << M(2,0) << " " << M(2,1) << " " << M(2,2) << endl;
    }
}
#endif


void TwoPassStabilizer::runPrePassIfNecessary()
{
    if (!isPrePassDone_)
    {        
        // check if we must do wobble suppression

        WobbleSuppressorBase *wobbleSuppressor = static_cast<WobbleSuppressorBase*>(wobbleSuppressor_);
        doWobbleSuppression_ = dynamic_cast<NullWobbleSuppressor*>(wobbleSuppressor) == 0;

        // estimate motions

        clock_t startTime = clock();
        log_->print("first pass: estimating motions");

        Mat prevFrame, frame;
        bool ok = true, ok2 = true;

        while (!(frame = frameSource_->nextFrame()).empty())
        {
            if (frameCount_ > 0)
            {
                motions_.push_back(motionEstimator_->estimate(prevFrame, frame, &ok));

                if (doWobbleSuppression_)
                {
                    Mat M = wobbleSuppressor_->motionEstimator()->estimate(prevFrame, frame, &ok2);
                    if (ok2)
                        motions2_.push_back(M);
                    else
                        motions2_.push_back(motions_.back());
                }

                if (ok)
                {
                    if (ok2) log_->print(".");
                    else log_->print("?");
                }
                else log_->print("x");
            }
            else
            {
                frameSize_ = frame.size();
                frameMask_.create(frameSize_, CV_8U);
                frameMask_.setTo(255);
            }

            prevFrame = frame;
            frameCount_++;
        }

        clock_t elapsedTime = clock() - startTime;
        log_->print("\nmotion estimation time: %.3f sec\n",
                    static_cast<double>(elapsedTime) / CLOCKS_PER_SEC);

        // add aux. motions

        for (int i = 0; i < radius_; ++i)
            motions_.push_back(Mat::eye(3, 3, CV_32F));

        // stabilize

        startTime = clock();

        stabilizationMotions_.resize(frameCount_);
        motionStabilizer_->stabilize(
            frameCount_, motions_, make_pair(0, frameCount_ - 1), &stabilizationMotions_[0]);

        elapsedTime = clock() - startTime;
        log_->print("motion stabilization time: %.3f sec\n",
                    static_cast<double>(elapsedTime) / CLOCKS_PER_SEC);

        // estimate optimal trim ratio if necessary

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

#if SAVE_MOTIONS
        saveMotions(frameCount_, motions_, stabilizationMotions_);
#endif

        isPrePassDone_ = true;
        frameSource_->reset();
    }
}


void TwoPassStabilizer::setUp(const Mat &firstFrame)
{
    int cacheSize = 2*radius_ + 1;
    frames_.resize(cacheSize);
    stabilizedFrames_.resize(cacheSize);
    stabilizedMasks_.resize(cacheSize);

    for (int i = -radius_; i <= 0; ++i)
        at(i, frames_) = firstFrame;

    WobbleSuppressorBase *wobbleSuppressor = static_cast<WobbleSuppressorBase*>(wobbleSuppressor_);
    doWobbleSuppression_ = dynamic_cast<NullWobbleSuppressor*>(wobbleSuppressor) == 0;
    if (doWobbleSuppression_)
    {
        wobbleSuppressor_->setFrameCount(frameCount_);
        wobbleSuppressor_->setMotions(motions_);
        wobbleSuppressor_->setMotions2(motions2_);
        wobbleSuppressor_->setStabilizationMotions(stabilizationMotions_);
    }

    StabilizerBase::setUp(firstFrame);
}


Mat TwoPassStabilizer::estimateMotion()
{
    return motions_[curPos_ - 1].clone();
}


Mat TwoPassStabilizer::estimateStabilizationMotion()
{
    return stabilizationMotions_[curStabilizedPos_].clone();
}


Mat TwoPassStabilizer::postProcessFrame(const Mat &frame)
{
    wobbleSuppressor_->suppress(curStabilizedPos_, frame, suppressedFrame_);
    return StabilizerBase::postProcessFrame(suppressedFrame_);
}

} // namespace videostab
} // namespace cv
