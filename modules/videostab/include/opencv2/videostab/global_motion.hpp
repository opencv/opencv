#ifndef __OPENCV_VIDEOSTAB_GLOBAL_MOTION_HPP__
#define __OPENCV_VIDEOSTAB_GLOBAL_MOTION_HPP__

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/videostab/optical_flow.hpp"

namespace cv
{
namespace videostab
{

enum MotionModel
{
    TRANSLATION = 0,
    TRANSLATION_AND_SCALE = 1,
    AFFINE = 2
};

Mat estimateGlobalMotionLeastSquares(
        const std::vector<Point2f> &points0, const std::vector<Point2f> &points1,
        int model = AFFINE, float *rmse = 0);

struct RansacParams
{
    int size; // subset size
    float thresh; // max error to classify as inlier
    float eps; // max outliers ratio
    float prob; // probability of success

    RansacParams(int size, float thresh, float eps, float prob)
        : size(size), thresh(thresh), eps(eps), prob(prob) {}

    static RansacParams affine2dMotionStd() { return RansacParams(6, 0.5f, 0.5f, 0.99f); }
    static RansacParams translationAndScale2dMotionStd() { return RansacParams(3, 0.5f, 0.5f, 0.99f); }
    static RansacParams translationMotionStd() { return RansacParams(2, 0.5f, 0.5f, 0.99f); }
};

Mat estimateGlobalMotionRobust(
        const std::vector<Point2f> &points0, const std::vector<Point2f> &points1,
        int model = AFFINE, const RansacParams &params = RansacParams::affine2dMotionStd(),
        float *rmse = 0, int *ninliers = 0);

class IGlobalMotionEstimator
{
public:
    virtual ~IGlobalMotionEstimator() {}
    virtual Mat estimate(const Mat &frame0, const Mat &frame1) = 0;
};

class PyrLkRobustMotionEstimator : public IGlobalMotionEstimator
{
public:
    PyrLkRobustMotionEstimator();

    void setDetector(Ptr<FeatureDetector> val) { detector_ = val; }
    Ptr<FeatureDetector> detector() const { return detector_; }

    void setOptFlowEstimator(Ptr<ISparseOptFlowEstimator> val) { optFlowEstimator_ = val; }
    Ptr<ISparseOptFlowEstimator> optFlowEstimator() const { return optFlowEstimator_; }

    void setMotionModel(MotionModel val) { motionModel_ = val; }
    MotionModel motionModel() const { return motionModel_; }

    void setRansacParams(const RansacParams &val) { ransacParams_ = val; }
    RansacParams ransacParams() const { return ransacParams_; }

    void setMaxRmse(float val) { maxRmse_ = val; }
    float maxRmse() const { return maxRmse_; }

    void setMinInlierRatio(float val) { minInlierRatio_ = val; }
    float minInlierRatio() const { return minInlierRatio_; }

    virtual Mat estimate(const Mat &frame0, const Mat &frame1);

private:
    Ptr<FeatureDetector> detector_;
    Ptr<ISparseOptFlowEstimator> optFlowEstimator_;
    MotionModel motionModel_;
    RansacParams ransacParams_;
    std::vector<uchar> status_;
    std::vector<KeyPoint> keypointsPrev_;
    std::vector<Point2f> pointsPrev_, points_;
    std::vector<Point2f> pointsPrevGood_, pointsGood_;
    float maxRmse_;
    float minInlierRatio_;
};

Mat getMotion(int from, int to, const std::vector<Mat> &motions);

Mat ensureInclusionConstraint(const Mat &M, Size size, float trimRatio);

float estimateOptimalTrimRatio(const Mat &M, Size size);

// frame1 is non-transformed frame
float alignementError(const Mat &M, const Mat &frame0, const Mat &mask0, const Mat &frame1);

} // namespace videostab
} // namespace cv

#endif
