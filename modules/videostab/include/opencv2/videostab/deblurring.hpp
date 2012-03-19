#ifndef __OPENCV_VIDEOSTAB_DEBLURRING_HPP__
#define __OPENCV_VIDEOSTAB_DEBLURRING_HPP__

#include <vector>
#include "opencv2/core/core.hpp"

namespace cv
{
namespace videostab
{

float calcBlurriness(const Mat &frame);

class IDeblurer
{
public:
    IDeblurer() : radius_(0), frames_(0), motions_(0) {}

    virtual ~IDeblurer() {}

    virtual void setRadius(int val) { radius_ = val; }
    int radius() const { return radius_; }

    virtual void setFrames(const std::vector<Mat> &val) { frames_ = &val; }
    const std::vector<Mat>& frames() const { return *frames_; }

    virtual void setMotions(const std::vector<Mat> &val) { motions_ = &val; }
    const std::vector<Mat>& motions() const { return *motions_; }

    virtual void setBlurrinessRates(const std::vector<float> &val) { blurrinessRates_ = &val; }
    const std::vector<float>& blurrinessRates() const { return *blurrinessRates_; }

    virtual void deblur(int idx, Mat &frame) = 0;

protected:
    int radius_;
    const std::vector<Mat> *frames_;
    const std::vector<Mat> *motions_;
    const std::vector<float> *blurrinessRates_;
};

class NullDeblurer : public IDeblurer
{
public:
    virtual void deblur(int idx, Mat &frame) {}
};

class WeightingDeblurer : public IDeblurer
{
public:
    WeightingDeblurer();

    void setSensitivity(float val) { sensitivity_ = val; }
    float sensitivity() const { return sensitivity_; }

    virtual void deblur(int idx, Mat &frame);

private:
    float sensitivity_;
    Mat_<float> bSum_, gSum_, rSum_, wSum_;
};

} // namespace videostab
} // namespace cv

#endif
