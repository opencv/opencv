#include "precomp.hpp"
#include "opencv2/videostab/motion_filtering.hpp"
#include "opencv2/videostab/global_motion.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

GaussianMotionFilter::GaussianMotionFilter(int radius, float stdev) : radius_(radius)
{
    float sum = 0;
    weight_.resize(2*radius_ + 1);
    for (int i = -radius_; i <= radius_; ++i)
        sum += weight_[radius_ + i] = std::exp(-i*i/(stdev*stdev));
    for (int i = -radius_; i <= radius_; ++i)
        weight_[radius_ + i] /= sum;
}


Mat GaussianMotionFilter::apply(int idx, vector<Mat> &motions) const
{
    const Mat &cur = at(idx, motions);
    Mat res = Mat::zeros(cur.size(), cur.type());
    for (int i = -radius_; i <= radius_; ++i)
        res += weight_[radius_ + i] * getMotion(idx, idx + i, motions);
    return res;
}

} // namespace videostab
} // namespace cv
