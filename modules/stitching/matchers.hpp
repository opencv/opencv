#ifndef __OPENCV_MATCHERS_HPP__
#define __OPENCV_MATCHERS_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

struct ImageFeatures
{    
    cv::Size img_size;
    cv::Mat hist;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};


class FeaturesFinder
{
public:
    virtual ~FeaturesFinder() {}
    void operator ()(const cv::Mat &image, ImageFeatures &features);

protected:
    virtual void find(const cv::Mat &image, ImageFeatures &features) = 0;
};


class SurfFeaturesFinder : public FeaturesFinder
{
public:
    SurfFeaturesFinder(bool try_use_gpu = true, double hess_thresh = 300.0, 
                       int num_octaves = 3, int num_layers = 4, 
                       int num_octaves_descr = 4, int num_layers_descr = 2);

protected:
    void find(const cv::Mat &image, ImageFeatures &features);

    cv::Ptr<FeaturesFinder> impl_;
};


struct MatchesInfo
{
    MatchesInfo();
    MatchesInfo(const MatchesInfo &other);
    const MatchesInfo& operator =(const MatchesInfo &other);

    int src_img_idx, dst_img_idx;       // Images indices (optional)
    std::vector<cv::DMatch> matches;
    std::vector<uchar> inliers_mask;    // Geometrically consistent matches mask
    int num_inliers;                    // Number of geometrically consistent matches
    cv::Mat H;                          // Estimated homography
    double confidence;                  // Confidence two images are from the same panorama
};


class FeaturesMatcher
{
public:
    virtual ~FeaturesMatcher() {}
    void operator ()(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info) 
            { match(features1, features2, matches_info); }
    void operator ()(const std::vector<ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches);

protected:
    virtual void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info) = 0;
};


class BestOf2NearestMatcher : public FeaturesMatcher
{
public:
    BestOf2NearestMatcher(bool try_use_gpu = true, float match_conf = 0.55f, int num_matches_thresh1 = 6, int num_matches_thresh2 = 6);

protected:
    void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo &matches_info);

    int num_matches_thresh1_;
    int num_matches_thresh2_;

    cv::Ptr<FeaturesMatcher> impl_;
};

#endif // __OPENCV_MATCHERS_HPP__