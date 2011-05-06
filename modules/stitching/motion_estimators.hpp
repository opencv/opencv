#ifndef __OPENCV_MOTION_ESTIMATORS_HPP__
#define __OPENCV_MOTION_ESTIMATORS_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include "matchers.hpp"
#include "util.hpp"

struct CameraParams
{
    CameraParams();
    CameraParams(const CameraParams& other);
    const CameraParams& operator =(const CameraParams& other);

    double focal;
    cv::Mat M, t;
};


class Estimator
{
public:
    void operator ()(const std::vector<cv::Mat> &images, const std::vector<ImageFeatures> &features,
                     const std::vector<MatchesInfo> &pairwise_matches, std::vector<CameraParams> &cameras)
    {
        estimate(images, features, pairwise_matches, cameras);
    }

protected:
    virtual void estimate(const std::vector<cv::Mat> &images, const std::vector<ImageFeatures> &features,
                          const std::vector<MatchesInfo> &pairwise_matches, std::vector<CameraParams> &cameras) = 0;
};


class HomographyBasedEstimator : public Estimator
{
public:
    HomographyBasedEstimator() : is_focals_estimated_(false) {}
    bool isFocalsEstimated() const { return is_focals_estimated_; }

private:   
    void estimate(const std::vector<cv::Mat> &images, const std::vector<ImageFeatures> &features,
                  const std::vector<MatchesInfo> &pairwise_matches, std::vector<CameraParams> &cameras);

    bool is_focals_estimated_;
};


class BundleAdjuster : public Estimator
{
public:
    enum { RAY_SPACE, FOCAL_RAY_SPACE };

    BundleAdjuster(int cost_space = FOCAL_RAY_SPACE, float dist_thresh = 1.f) 
        : cost_space_(cost_space), dist_thresh_(dist_thresh) {}

private:
    void estimate(const std::vector<cv::Mat> &images, const std::vector<ImageFeatures> &features,
                  const std::vector<MatchesInfo> &pairwise_matches, std::vector<CameraParams> &cameras);

    void calcError(cv::Mat &err);
    void calcJacobian();

    int num_images_;
    int total_num_matches_;
    const cv::Mat *images_;
    const ImageFeatures *features_;
    const MatchesInfo *pairwise_matches_;
    cv::Mat cameras_;
    std::vector<std::pair<int,int> > edges_;

    int cost_space_;
    float dist_thresh_;
    cv::Mat err_, err1_, err2_;
    cv::Mat J_;
};


void waveCorrect(std::vector<cv::Mat> &rmats);


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

void findMaxSpanningTree(int num_images, const std::vector<MatchesInfo> &pairwise_matches, 
                         Graph &span_tree, std::vector<int> &centers);

#endif // __OPENCV_MOTION_ESTIMATORS_HPP__
