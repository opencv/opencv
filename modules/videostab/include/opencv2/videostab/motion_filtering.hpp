#ifndef __OPENCV_VIDEOSTAB_MOTION_FILTERING_HPP__
#define __OPENCV_VIDEOSTAB_MOTION_FILTERING_HPP__

#include <vector>
#include "opencv2/core/core.hpp"

namespace cv
{
namespace videostab
{

class IMotionFilter
{
public:
    virtual ~IMotionFilter() {}
    virtual int radius() const = 0;
    virtual Mat apply(int index, std::vector<Mat> &Ms) const = 0;
};

class GaussianMotionFilter : public IMotionFilter
{
public:
    GaussianMotionFilter(int radius, float stdev);
    virtual int radius() const { return radius_; }
    virtual Mat apply(int idx, std::vector<Mat> &motions) const;

private:
    int radius_;
    std::vector<float> weight_;
};

} // namespace videostab
} // namespace

#endif
