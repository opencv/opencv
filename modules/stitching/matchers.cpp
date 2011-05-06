#include <algorithm>
#include <functional>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "matchers.hpp"
#include "util.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

//////////////////////////////////////////////////////////////////////////////

namespace
{
    class CpuSurfFeaturesFinder : public FeaturesFinder
    {
    public:
        inline CpuSurfFeaturesFinder(double hess_thresh, int num_octaves, int num_layers, 
                                     int num_octaves_descr, int num_layers_descr) 
        {
            detector_ = new SurfFeatureDetector(hess_thresh, num_octaves, num_layers);
            extractor_ = new SurfDescriptorExtractor(num_octaves_descr, num_layers_descr);
        }

    protected:
        void find(const vector<Mat> &images, vector<ImageFeatures> &features);

    private:
        Ptr<FeatureDetector> detector_;
        Ptr<DescriptorExtractor> extractor_;
    };

    void CpuSurfFeaturesFinder::find(const vector<Mat> &images, vector<ImageFeatures> &features)
    {
        // Make images gray
        vector<Mat> gray_images(images.size());
        for (size_t i = 0; i < images.size(); ++i)
        {
            CV_Assert(images[i].depth() == CV_8U);
            cvtColor(images[i], gray_images[i], CV_BGR2GRAY);
        }

        features.resize(images.size());

        // Find keypoints in all images
        for (size_t i = 0; i < images.size(); ++i)
        {
            detector_->detect(gray_images[i], features[i].keypoints);
            extractor_->compute(gray_images[i], features[i].keypoints, features[i].descriptors);
        }
    }
    
    class GpuSurfFeaturesFinder : public FeaturesFinder
    {
    public:
        inline GpuSurfFeaturesFinder(double hess_thresh, int num_octaves, int num_layers, 
                                     int num_octaves_descr, int num_layers_descr) 
        {
            surf_.hessianThreshold = hess_thresh;
            surf_.extended = false;
            num_octaves_ = num_octaves;
            num_layers_ = num_layers;
            num_octaves_descr_ = num_octaves_descr;
            num_layers_descr_ = num_layers_descr;
        }

    protected:
        void find(const vector<Mat> &images, vector<ImageFeatures> &features);

    private:
        SURF_GPU surf_;
        int num_octaves_, num_layers_;
        int num_octaves_descr_, num_layers_descr_;
    };

    void GpuSurfFeaturesFinder::find(const vector<Mat> &images, vector<ImageFeatures> &features)
    {
        // Make images gray
        vector<GpuMat> gray_images(images.size());
        for (size_t i = 0; i < images.size(); ++i)
        {
            CV_Assert(images[i].depth() == CV_8U);
            cvtColor(GpuMat(images[i]), gray_images[i], CV_BGR2GRAY);
        }

        features.resize(images.size());

        // Find keypoints in all images
        GpuMat d_keypoints;
        GpuMat d_descriptors;
        for (size_t i = 0; i < images.size(); ++i)
        {
            surf_.nOctaves = num_octaves_;
            surf_.nOctaveLayers = num_layers_;
            surf_(gray_images[i], GpuMat(), d_keypoints);

            surf_.nOctaves = num_octaves_descr_;
            surf_.nOctaveLayers = num_layers_descr_;
            surf_(gray_images[i], GpuMat(), d_keypoints, d_descriptors, true);

            surf_.downloadKeypoints(d_keypoints, features[i].keypoints);
            d_descriptors.download(features[i].descriptors);
        }
    }
}

SurfFeaturesFinder::SurfFeaturesFinder(bool gpu_hint, double hess_thresh, int num_octaves, int num_layers, 
                                       int num_octaves_descr, int num_layers_descr)
{
    if (gpu_hint && getCudaEnabledDeviceCount() > 0)
        impl_ = new GpuSurfFeaturesFinder(hess_thresh, num_octaves, num_layers, num_octaves_descr, num_layers_descr);
    else
        impl_ = new CpuSurfFeaturesFinder(hess_thresh, num_octaves, num_layers, num_octaves_descr, num_layers_descr);
}


void SurfFeaturesFinder::find(const vector<Mat> &images, vector<ImageFeatures> &features)
{
    (*impl_)(images, features);
}


//////////////////////////////////////////////////////////////////////////////

MatchesInfo::MatchesInfo() : src_img_idx(-1), dst_img_idx(-1), num_inliers(0) {}


MatchesInfo::MatchesInfo(const MatchesInfo &other)
{
    *this = other;
}


const MatchesInfo& MatchesInfo::operator =(const MatchesInfo &other)
{
    src_img_idx = other.src_img_idx;
    dst_img_idx = other.dst_img_idx;
    matches = other.matches;
    num_inliers = other.num_inliers;
    H = other.H.clone();
    return *this;
}


//////////////////////////////////////////////////////////////////////////////

void FeaturesMatcher::operator ()(const vector<Mat> &images, const vector<ImageFeatures> &features,
                                  vector<MatchesInfo> &pairwise_matches)
{
    pairwise_matches.resize(images.size() * images.size());
    for (size_t i = 0; i < images.size(); ++i)
    {
        LOGLN("Processing image " << i << "... ");
        for (size_t j = 0; j < images.size(); ++j)
        {
            if (i == j)
                continue;
            size_t pair_idx = i * images.size() + j;
            (*this)(images[i], features[i], images[j], features[j], pairwise_matches[pair_idx]);
            pairwise_matches[pair_idx].src_img_idx = i;
            pairwise_matches[pair_idx].dst_img_idx = j;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////

namespace
{
    class CpuMatcher : public FeaturesMatcher
    {
    public:
        inline CpuMatcher(float match_conf) : match_conf_(match_conf) {}

        void match(const cv::Mat&, const ImageFeatures &features1, const cv::Mat&, const ImageFeatures &features2, MatchesInfo& matches_info);

    private:
        float match_conf_;
    };

    void CpuMatcher::match(const cv::Mat&, const ImageFeatures &features1, const cv::Mat&, const ImageFeatures &features2, MatchesInfo& matches_info)
    {
        matches_info.matches.clear();

        BruteForceMatcher< L2<float> > matcher;
        vector< vector<DMatch> > pair_matches;

        // Find 1->2 matches
        matcher.knnMatch(features1.descriptors, features2.descriptors, pair_matches, 2);
        for (size_t i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
            const DMatch& m0 = pair_matches[i][0];
            const DMatch& m1 = pair_matches[i][1];
            if (m0.distance < (1.f - match_conf_) * m1.distance)
                matches_info.matches.push_back(m0);
        }

        // Find 2->1 matches
        pair_matches.clear();
        matcher.knnMatch(features2.descriptors, features1.descriptors, pair_matches, 2);
        for (size_t i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
            const DMatch& m0 = pair_matches[i][0];
            const DMatch& m1 = pair_matches[i][1];
            if (m0.distance < (1.f - match_conf_) * m1.distance)
                matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
        }
    }
        
    class GpuMatcher : public FeaturesMatcher
    {
    public:
        inline GpuMatcher(float match_conf) : match_conf_(match_conf) {}

        void match(const cv::Mat&, const ImageFeatures &features1, const cv::Mat&, const ImageFeatures &features2, MatchesInfo& matches_info);

    private:
        float match_conf_;

        GpuMat descriptors1_;
        GpuMat descriptors2_;

        GpuMat trainIdx_, distance_, allDist_;
    };

    void GpuMatcher::match(const cv::Mat&, const ImageFeatures &features1, const cv::Mat&, const ImageFeatures &features2, MatchesInfo& matches_info)
    {
        matches_info.matches.clear();

        BruteForceMatcher_GPU< L2<float> > matcher;
        
        descriptors1_.upload(features1.descriptors);
        descriptors2_.upload(features2.descriptors);

        vector< vector<DMatch> > pair_matches;

        // Find 1->2 matches
        matcher.knnMatch(descriptors1_, descriptors2_, trainIdx_, distance_, allDist_, 2);
        matcher.knnMatchDownload(trainIdx_, distance_, pair_matches);
        for (size_t i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
            const DMatch& m0 = pair_matches[i][0];
            const DMatch& m1 = pair_matches[i][1];

            CV_Assert(m0.queryIdx < static_cast<int>(features1.keypoints.size()));
            CV_Assert(m0.trainIdx < static_cast<int>(features2.keypoints.size()));

            if (m0.distance < (1.f - match_conf_) * m1.distance)
                matches_info.matches.push_back(m0);
        }

        // Find 2->1 matches
        pair_matches.clear();
        matcher.knnMatch(descriptors2_, descriptors1_, trainIdx_, distance_, allDist_, 2);
        matcher.knnMatchDownload(trainIdx_, distance_, pair_matches);
        for (size_t i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
            const DMatch& m0 = pair_matches[i][0];
            const DMatch& m1 = pair_matches[i][1];

            CV_Assert(m0.trainIdx < static_cast<int>(features1.keypoints.size()));
            CV_Assert(m0.queryIdx < static_cast<int>(features2.keypoints.size()));

            if (m0.distance < (1.f - match_conf_) * m1.distance)
                matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
        }
    }
}

BestOf2NearestMatcher::BestOf2NearestMatcher(bool gpu_hint, float match_conf, int num_matches_thresh1, int num_matches_thresh2)
{
    if (gpu_hint && getCudaEnabledDeviceCount() > 0)
        impl_ = new GpuMatcher(match_conf);
    else
        impl_ = new CpuMatcher(match_conf);

    num_matches_thresh1_ = num_matches_thresh1;
    num_matches_thresh2_ = num_matches_thresh2;
}


void BestOf2NearestMatcher::match(const Mat &img1, const ImageFeatures &features1, const Mat &img2, const ImageFeatures &features2,
                                  MatchesInfo &matches_info)
{
    (*impl_)(img1, features1, img2, features2, matches_info);

    // Check if it makes sense to find homography
    if (matches_info.matches.size() < static_cast<size_t>(num_matches_thresh1_))
        return;

    // Construct point-point correspondences for homography estimation
    Mat src_points(1, matches_info.matches.size(), CV_32FC2);
    Mat dst_points(1, matches_info.matches.size(), CV_32FC2);
    for (size_t i = 0; i < matches_info.matches.size(); ++i)
    {
        const DMatch& m = matches_info.matches[i];

        Point2f p = features1.keypoints[m.queryIdx].pt;
        p.x -= img1.cols * 0.5f;
        p.y -= img1.rows * 0.5f;
        src_points.at<Point2f>(0, i) = p;

        p = features2.keypoints[m.trainIdx].pt;
        p.x -= img2.cols * 0.5f;
        p.y -= img2.rows * 0.5f;
        dst_points.at<Point2f>(0, i) = p;
    }

    // Find pair-wise motion
    matches_info.H = findHomography(src_points, dst_points, matches_info.inliers_mask, CV_RANSAC);

    // Find number of inliers
    matches_info.num_inliers = 0;
    for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
        if (matches_info.inliers_mask[i])
            matches_info.num_inliers++;

    // Check if we should try to refine motion
    if (matches_info.num_inliers < num_matches_thresh2_)
        return;

    // Construct point-point correspondences for inliers only
    src_points.create(1, matches_info.num_inliers, CV_32FC2);
    dst_points.create(1, matches_info.num_inliers, CV_32FC2);
    int inlier_idx = 0;
    for (size_t i = 0; i < matches_info.matches.size(); ++i)
    {
        if (!matches_info.inliers_mask[i])
            continue;

        const DMatch& m = matches_info.matches[i];

        Point2f p = features1.keypoints[m.queryIdx].pt;
        p.x -= img1.cols * 0.5f;
        p.y -= img2.rows * 0.5f;
        src_points.at<Point2f>(0, inlier_idx) = p;

        p = features2.keypoints[m.trainIdx].pt;
        p.x -= img2.cols * 0.5f;
        p.y -= img2.rows * 0.5f;
        dst_points.at<Point2f>(0, inlier_idx) = p;

        inlier_idx++;
    }

    // Rerun motion estimation on inliers only
    matches_info.H = findHomography(src_points, dst_points, CV_RANSAC);
}
