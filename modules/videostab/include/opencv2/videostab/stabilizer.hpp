#ifndef __OPENCV_VIDEOSTAB_STABILIZER_HPP__
#define __OPENCV_VIDEOSTAB_STABILIZER_HPP__

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videostab/global_motion.hpp"
#include "opencv2/videostab/motion_filtering.hpp"
#include "opencv2/videostab/frame_source.hpp"
#include "opencv2/videostab/log.hpp"
#include "opencv2/videostab/inpainting.hpp"
#include "opencv2/videostab/deblurring.hpp"

namespace cv
{
namespace videostab
{

class Stabilizer : public IFrameSource
{
public:
    Stabilizer();

    void setLog(Ptr<ILog> log) { log_ = log; }
    Ptr<ILog> log() const { return log_; }

    void setFrameSource(Ptr<IFrameSource> val) { frameSource_ = val; reset(); }
    Ptr<IFrameSource> frameSource() const { return frameSource_; }

    void setMotionEstimator(Ptr<IGlobalMotionEstimator> val) { motionEstimator_ = val; }
    Ptr<IGlobalMotionEstimator> motionEstimator() const { return motionEstimator_; }

    void setMotionFilter(Ptr<IMotionFilter> val) { motionFilter_ = val; reset(); }
    Ptr<IMotionFilter> motionFilter() const { return motionFilter_; }

    void setDeblurer(Ptr<IDeblurer> val) { deblurer_ = val; reset(); }
    Ptr<IDeblurer> deblurrer() const { return deblurer_; }

    void setEstimateTrimRatio(bool val) { mustEstimateTrimRatio_ = val; reset(); }
    bool mustEstimateTrimRatio() const { return mustEstimateTrimRatio_; }

    void setTrimRatio(float val) { trimRatio_ = val; reset(); }
    int trimRatio() const { return trimRatio_; }

    void setInclusionConstraint(bool val) { inclusionConstraint_ = val; }
    bool inclusionConstraint() const { return inclusionConstraint_; }

    void setBorderMode(int val) { borderMode_ = val; }
    int borderMode() const { return borderMode_; }

    void setInpainter(Ptr<IInpainter> val) { inpainter_ = val; reset(); }
    Ptr<IInpainter> inpainter() const { return inpainter_; }

    virtual void reset();
    virtual Mat nextFrame();

private:
    void estimateMotionsAndTrimRatio();
    void processFirstFrame(Mat &frame);
    bool processNextFrame();
    void stabilizeFrame(int idx);

    Ptr<IFrameSource> frameSource_;
    Ptr<IGlobalMotionEstimator> motionEstimator_;
    Ptr<IMotionFilter> motionFilter_;
    Ptr<IDeblurer> deblurer_;
    Ptr<IInpainter> inpainter_;
    bool mustEstimateTrimRatio_;
    float trimRatio_;
    bool inclusionConstraint_;
    int borderMode_;    
    Ptr<ILog> log_;

    Size frameSize_;
    Mat frameMask_;
    int radius_;
    int curPos_;
    int curStabilizedPos_;
    bool auxPassWasDone_;
    bool doDeblurring_;
    Mat preProcessedFrame_;
    bool doInpainting_;
    Mat inpaintingMask_;
    std::vector<Mat> frames_;
    std::vector<Mat> motions_; // motions_[i] is the motion from i to i+1 frame
    std::vector<float> blurrinessRates_;
    std::vector<Mat> stabilizedFrames_;
    std::vector<Mat> stabilizedMasks_;
    std::vector<Mat> stabilizationMotions_;
};

} // namespace videostab
} // namespace cv

#endif
