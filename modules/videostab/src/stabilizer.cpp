#include "precomp.hpp"
#include "opencv2/videostab/stabilizer.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

Stabilizer::Stabilizer()
{
    setFrameSource(new NullFrameSource());
    setMotionEstimator(new PyrLkRobustMotionEstimator());
    setMotionFilter(new GaussianMotionFilter(15, sqrt(15)));
    setDeblurer(new NullDeblurer());
    setInpainter(new NullInpainter());
    setEstimateTrimRatio(true);
    setTrimRatio(0);
    setInclusionConstraint(false);
    setBorderMode(BORDER_REPLICATE);
    setLog(new NullLog());
}


void Stabilizer::reset()
{
    radius_ = 0;
    curPos_ = -1;
    curStabilizedPos_ = -1;
    auxPassWasDone_ = false;
    frames_.clear();
    motions_.clear();
    stabilizedFrames_.clear();
    stabilizationMotions_.clear();
    doDeblurring_ = false;
    doInpainting_ = false;
}


Mat Stabilizer::nextFrame()
{
    if (mustEstimateTrimRatio_ && !auxPassWasDone_)
    {
        estimateMotionsAndTrimRatio();
        auxPassWasDone_ = true;
        frameSource_->reset();
    }

    if (curStabilizedPos_ == curPos_ && curStabilizedPos_ != -1)
        return Mat(); // we've processed all frames already

    bool processed;
    do {
        processed = processNextFrame();
    } while (processed && curStabilizedPos_ == -1);

    if (curStabilizedPos_ == -1)
        return Mat(); // frame source is empty

    const Mat &stabilizedFrame = at(curStabilizedPos_, stabilizedFrames_);
    int dx = floor(trimRatio_ * stabilizedFrame.cols);
    int dy = floor(trimRatio_ * stabilizedFrame.rows);
    return stabilizedFrame(Rect(dx, dy, stabilizedFrame.cols - 2*dx, stabilizedFrame.rows - 2*dy));
}


void Stabilizer::estimateMotionsAndTrimRatio()
{
    log_->print("estimating motions and trim ratio");

    Size size;
    Mat prevFrame, frame;
    int frameCount = 0;

    while (!(frame = frameSource_->nextFrame()).empty())
    {
        if (frameCount > 0)
            motions_.push_back(motionEstimator_->estimate(prevFrame, frame));
        else
            size = frame.size();
        prevFrame = frame;
        frameCount++;

        log_->print(".");
    }

    radius_ = motionFilter_->radius();
    for (int i = 0; i < radius_; ++i)
        motions_.push_back(Mat::eye(3, 3, CV_32F));
    log_->print("\n");

    trimRatio_ = 0;
    for (int i = 0; i < frameCount; ++i)
    {
        Mat S = motionFilter_->apply(i, motions_);
        trimRatio_ = std::max(trimRatio_, estimateOptimalTrimRatio(S, size));
        stabilizationMotions_.push_back(S);
    }

    log_->print("estimated trim ratio: %f\n", static_cast<double>(trimRatio_));
}


void Stabilizer::processFirstFrame(Mat &frame)
{
    log_->print("processing frames");

    frameSize_ = frame.size();
    frameMask_.create(frameSize_, CV_8U);
    frameMask_.setTo(255);

    radius_ = motionFilter_->radius();
    int cacheSize = 2*radius_ + 1;

    frames_.resize(cacheSize);
    stabilizedFrames_.resize(cacheSize);
    stabilizedMasks_.resize(cacheSize);

    if (!auxPassWasDone_)
    {
        motions_.resize(cacheSize);
        stabilizationMotions_.resize(cacheSize);
    }

    for (int i = -radius_; i < 0; ++i)
    {
        at(i, motions_) = Mat::eye(3, 3, CV_32F);
        at(i, frames_) = frame;
    }

    at(0, frames_) = frame;

    IInpainter *inpainter = static_cast<IInpainter*>(inpainter_);
    doInpainting_ = dynamic_cast<NullInpainter*>(inpainter) == 0;
    if (doInpainting_)
    {
        inpainter_->setRadius(radius_);
        inpainter_->setFrames(frames_);
        inpainter_->setMotions(motions_);
        inpainter_->setStabilizedFrames(stabilizedFrames_);
        inpainter_->setStabilizationMotions(stabilizationMotions_);
    }

    IDeblurer *deblurer = static_cast<IDeblurer*>(deblurer_);
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
    }
}


bool Stabilizer::processNextFrame()
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

            if (!auxPassWasDone_)
            {
                Mat motionPrevToCur = motionEstimator_->estimate(
                        at(curPos_ - 1, frames_), at(curPos_, frames_));
                at(curPos_ - 1, motions_) = motionPrevToCur;
            }

            if (curPos_ >= radius_)
            {
                curStabilizedPos_ = curPos_ - radius_;
                stabilizeFrame(curStabilizedPos_);
            }
        }
        else
            processFirstFrame(frame);

        log_->print(".");
        return true;
    }

    if (curStabilizedPos_ < curPos_)
    {
        curStabilizedPos_++;
        at(curStabilizedPos_ + radius_, frames_) = at(curPos_, frames_);
        at(curStabilizedPos_ + radius_ - 1, motions_) = at(curPos_ - 1, motions_);
        stabilizeFrame(curStabilizedPos_);

        log_->print(".");
        return true;
    }

    return false;
}


void Stabilizer::stabilizeFrame(int idx)
{
    Mat stabMotion;
    if (!auxPassWasDone_)
        stabMotion = motionFilter_->apply(idx, motions_);
    else
        stabMotion = at(idx, stabilizationMotions_);

    if (inclusionConstraint_ && !mustEstimateTrimRatio_)
        stabMotion = ensureInclusionConstraint(stabMotion, frameSize_, trimRatio_);

    at(idx, stabilizationMotions_) = stabMotion;

    if (doDeblurring_)
    {
        at(idx, frames_).copyTo(preProcessedFrame_);
        deblurer_->deblur(idx, preProcessedFrame_);
    }
    else
        preProcessedFrame_ = at(idx, frames_);

    // apply stabilization transformation
    warpAffine(
            preProcessedFrame_, at(idx, stabilizedFrames_), stabMotion(Rect(0,0,3,2)),
            frameSize_, INTER_LINEAR, borderMode_);

    if (doInpainting_)
    {
        warpAffine(
                frameMask_, at(idx, stabilizedMasks_), stabMotion(Rect(0,0,3,2)), frameSize_,
                INTER_NEAREST);
        erode(at(idx, stabilizedMasks_), at(idx, stabilizedMasks_), Mat());
        at(idx, stabilizedMasks_).copyTo(inpaintingMask_);
        inpainter_->inpaint(idx, at(idx, stabilizedFrames_), inpaintingMask_);
    }
}

} // namespace videostab
} // namespace cv
