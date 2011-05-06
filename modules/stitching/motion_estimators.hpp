#ifndef __OPENCV_MOTION_ESTIMATORS_HPP__
#define __OPENCV_MOTION_ESTIMATORS_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "util.hpp"

struct ImageFeatures
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};


class FeaturesFinder
{
public:
    virtual ~FeaturesFinder() {}
    void operator ()(const std::vector<cv::Mat> &images, std::vector<ImageFeatures> &features) { find(images, features); }

protected:
    virtual void find(const std::vector<cv::Mat> &images, std::vector<ImageFeatures> &features) = 0;
};


class SurfFeaturesFinder : public FeaturesFinder
{
public:
    explicit SurfFeaturesFinder(bool gpu_hint = true);

protected:
    void find(const std::vector<cv::Mat> &images, std::vector<ImageFeatures> &features);

    cv::Ptr<FeaturesFinder> impl_;
};


struct MatchesInfo
{
    MatchesInfo();
    MatchesInfo(const MatchesInfo &other);
    const MatchesInfo& operator =(const MatchesInfo &other);

    int src_img_idx, dst_img_idx; // Optional images indices
    std::vector<cv::DMatch> matches;
    int num_inliers; // Number of geometrically consistent matches
    cv::Mat H; // Homography
};


class FeaturesMatcher
{
public:
    virtual ~FeaturesMatcher() {}
    void operator ()(const cv::Mat &img1, const ImageFeatures &features1, const cv::Mat &img2, const ImageFeatures &features2,
                     MatchesInfo& matches_info) { match(img1, features1, img2, features2, matches_info); }
    void operator ()(const std::vector<cv::Mat> &images, const std::vector<ImageFeatures> &features,
                     std::vector<MatchesInfo> &pairwise_matches);

protected:
    virtual void match(const cv::Mat &img1, const ImageFeatures &features1, const cv::Mat &img2, const ImageFeatures &features2,
                       MatchesInfo& matches_info) = 0;
};


class BestOf2NearestMatcher : public FeaturesMatcher
{
public:
    explicit BestOf2NearestMatcher(bool gpu_hint = true, float match_conf = 0.55f, int num_matches_thresh1 = 5, int num_matches_thresh2 = 5);

protected:
    void match(const cv::Mat &img1, const ImageFeatures &features1, const cv::Mat &img2, const ImageFeatures &features2,
               MatchesInfo &matches_info);

    int num_matches_thresh1_;
    int num_matches_thresh2_;

    cv::Ptr<FeaturesMatcher> impl_;
};


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
