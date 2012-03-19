#ifndef __OPENCV_VIDEOSTAB_INPAINTINT_HPP__
#define __OPENCV_VIDEOSTAB_INPAINTINT_HPP__

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/videostab/optical_flow.hpp"
#include "opencv2/videostab/fast_marching.hpp"

namespace cv
{
namespace videostab
{

class IInpainter
{
public:
    IInpainter()
        : radius_(0), frames_(0), motions_(0),
          stabilizedFrames_(0), stabilizationMotions_(0) {}

    virtual ~IInpainter() {}

    virtual void setRadius(int val) { radius_ = val; }
    int radius() const { return radius_; }

    virtual void setFrames(const std::vector<Mat> &val) { frames_ = &val; }
    const std::vector<Mat>& frames() const { return *frames_; }

    virtual void setMotions(const std::vector<Mat> &val) { motions_ = &val; }
    const std::vector<Mat>& motions() const { return *motions_; }

    virtual void setStabilizedFrames(const std::vector<Mat> &val) { stabilizedFrames_ = &val; }
    const std::vector<Mat>& stabilizedFrames() const { return *stabilizedFrames_; }

    virtual void setStabilizationMotions(const std::vector<Mat> &val) { stabilizationMotions_ = &val; }
    const std::vector<Mat>& stabilizationMotions() const { return *stabilizationMotions_; }

    virtual void inpaint(int idx, Mat &frame, Mat &mask) = 0;

protected:
    int radius_;
    const std::vector<Mat> *frames_;
    const std::vector<Mat> *motions_;
    const std::vector<Mat> *stabilizedFrames_;
    const std::vector<Mat> *stabilizationMotions_;
};

class NullInpainter : public IInpainter
{
public:
    virtual void inpaint(int idx, Mat &frame, Mat &mask) {}
};

class InpaintingPipeline : public IInpainter
{
public:
    void pushBack(Ptr<IInpainter> inpainter) { inpainters_.push_back(inpainter); }
    bool empty() const { return inpainters_.empty(); }

    virtual void setRadius(int val);
    virtual void setFrames(const std::vector<Mat> &val);
    virtual void setMotions(const std::vector<Mat> &val);
    virtual void setStabilizedFrames(const std::vector<Mat> &val);
    virtual void setStabilizationMotions(const std::vector<Mat> &val);

    virtual void inpaint(int idx, Mat &frame, Mat &mask);

private:
    std::vector<Ptr<IInpainter> > inpainters_;
};

class ConsistentMosaicInpainter : public IInpainter
{
public:
    ConsistentMosaicInpainter();

    void setStdevThresh(float val) { stdevThresh_ = val; }
    float stdevThresh() const { return stdevThresh_; }

    virtual void inpaint(int idx, Mat &frame, Mat &mask);

private:
    float stdevThresh_;
};

class MotionInpainter : public IInpainter
{
public:
    MotionInpainter();

    void setOptFlowEstimator(Ptr<IDenseOptFlowEstimator> val) { optFlowEstimator_ = val; }
    Ptr<IDenseOptFlowEstimator> optFlowEstimator() const { return optFlowEstimator_; }

    void setFlowErrorThreshold(float val) { flowErrorThreshold_ = val; }
    float flowErrorThreshold() const { return flowErrorThreshold_; }

    void setBorderMode(int val) { borderMode_ = val; }
    int borderMode() const { return borderMode_; }

    virtual void inpaint(int idx, Mat &frame, Mat &mask);

private:
    FastMarchingMethod fmm_;
    Ptr<IDenseOptFlowEstimator> optFlowEstimator_;
    float flowErrorThreshold_;
    int borderMode_;

    Mat frame1_, transformedFrame1_;
    Mat_<uchar> grayFrame_, transformedGrayFrame1_;
    Mat_<uchar> mask1_, transformedMask1_;
    Mat_<float> flowX_, flowY_, flowErrors_;
    Mat_<uchar> flowMask_;
};

class ColorAverageInpainter : public IInpainter
{
public:
    virtual void inpaint(int idx, Mat &frame, Mat &mask);

private:
    FastMarchingMethod fmm_;
};

void calcFlowMask(
        const Mat &flowX, const Mat &flowY, const Mat &errors, float maxError,
        const Mat &mask0, const Mat &mask1, Mat &flowMask);

void completeFrameAccordingToFlow(
        const Mat &flowMask, const Mat &flowX, const Mat &flowY, const Mat &frame1, const Mat &mask1,
        Mat& frame0, Mat &mask0);

} // namespace videostab
} // namespace cv

#endif
