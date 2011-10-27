/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

#ifndef ANDROID
using namespace cv::gpu;
#endif

namespace {

struct DistIdxPair
{
    bool operator<(const DistIdxPair &other) const { return dist < other.dist; }
    double dist;
    int idx;
};


struct MatchPairsBody
{
    MatchPairsBody(const MatchPairsBody& other)
            : matcher(other.matcher), features(other.features),
              pairwise_matches(other.pairwise_matches), near_pairs(other.near_pairs) {}

    MatchPairsBody(FeaturesMatcher &matcher, const vector<ImageFeatures> &features,
                   vector<MatchesInfo> &pairwise_matches, vector<pair<int,int> > &near_pairs)
            : matcher(matcher), features(features),
              pairwise_matches(pairwise_matches), near_pairs(near_pairs) {}

    void operator ()(const BlockedRange &r) const
    {
        const int num_images = static_cast<int>(features.size());
        for (int i = r.begin(); i < r.end(); ++i)
        {
            int from = near_pairs[i].first;
            int to = near_pairs[i].second;
            int pair_idx = from*num_images + to;

            matcher(features[from], features[to], pairwise_matches[pair_idx]);
            pairwise_matches[pair_idx].src_img_idx = from;
            pairwise_matches[pair_idx].dst_img_idx = to;

            size_t dual_pair_idx = to*num_images + from;

            pairwise_matches[dual_pair_idx] = pairwise_matches[pair_idx];
            pairwise_matches[dual_pair_idx].src_img_idx = to;
            pairwise_matches[dual_pair_idx].dst_img_idx = from;

            if (!pairwise_matches[pair_idx].H.empty())
                pairwise_matches[dual_pair_idx].H = pairwise_matches[pair_idx].H.inv();

            for (size_t j = 0; j < pairwise_matches[dual_pair_idx].matches.size(); ++j)
                std::swap(pairwise_matches[dual_pair_idx].matches[j].queryIdx,
                          pairwise_matches[dual_pair_idx].matches[j].trainIdx);
            LOG(".");
        }
    }

    FeaturesMatcher &matcher;
    const vector<ImageFeatures> &features;
    vector<MatchesInfo> &pairwise_matches;
    vector<pair<int,int> > &near_pairs;

private:
    void operator =(const MatchPairsBody&);
};


//////////////////////////////////////////////////////////////////////////////

typedef set<pair<int,int> > MatchesSet;

// These two classes are aimed to find features matches only, not to
// estimate homography

class CpuMatcher : public FeaturesMatcher
{
public:
    CpuMatcher(float match_conf) : FeaturesMatcher(true), match_conf_(match_conf) {}
    void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info);

private:
    float match_conf_;
};

#ifndef ANDROID
class GpuMatcher : public FeaturesMatcher
{
public:
    GpuMatcher(float match_conf) : match_conf_(match_conf) {}
    void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info);

    void collectGarbage();

private:
    float match_conf_;
    GpuMat descriptors1_, descriptors2_;
    GpuMat train_idx_, distance_, all_dist_;
    vector< vector<DMatch> > pair_matches;
};
#endif


void CpuMatcher::match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info)
{
    CV_Assert(features1.descriptors.type() == features2.descriptors.type());
    CV_Assert(features2.descriptors.depth() == CV_8U || features2.descriptors.depth() == CV_32F);

#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::match2nearest(features1, features2, matches_info, match_conf_))
        return;
#endif

    matches_info.matches.clear();

    Ptr<flann::IndexParams> indexParams = new flann::KDTreeIndexParams();
    Ptr<flann::SearchParams> searchParams = new flann::SearchParams();

    if (features2.descriptors.depth() == CV_8U)
    {
        indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
        searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
    }

    FlannBasedMatcher matcher(indexParams, searchParams);
    vector< vector<DMatch> > pair_matches;
    MatchesSet matches;

    // Find 1->2 matches
    matcher.knnMatch(features1.descriptors, features2.descriptors, pair_matches, 2);
    for (size_t i = 0; i < pair_matches.size(); ++i)
    {
        if (pair_matches[i].size() < 2)
            continue;
        const DMatch& m0 = pair_matches[i][0];
        const DMatch& m1 = pair_matches[i][1];
        if (m0.distance < (1.f - match_conf_) * m1.distance)
        {
            matches_info.matches.push_back(m0);
            matches.insert(make_pair(m0.queryIdx, m0.trainIdx));
        }
    }
    LOG("\n1->2 matches: " << matches_info.matches.size() << endl);

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
            if (matches.find(make_pair(m0.trainIdx, m0.queryIdx)) == matches.end())
                matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
    }
    LOG("1->2 & 2->1 matches: " << matches_info.matches.size() << endl);
}

#ifndef ANDROID
void GpuMatcher::match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info)
{
    matches_info.matches.clear();

    ensureSizeIsEnough(features1.descriptors.size(), features1.descriptors.type(), descriptors1_);
    ensureSizeIsEnough(features2.descriptors.size(), features2.descriptors.type(), descriptors2_);

    descriptors1_.upload(features1.descriptors);
    descriptors2_.upload(features2.descriptors);

    BruteForceMatcher_GPU< L2<float> > matcher;
    MatchesSet matches;

    // Find 1->2 matches
    pair_matches.clear();
    matcher.knnMatchSingle(descriptors1_, descriptors2_, train_idx_, distance_, all_dist_, 2);
    matcher.knnMatchDownload(train_idx_, distance_, pair_matches);
    for (size_t i = 0; i < pair_matches.size(); ++i)
    {
        if (pair_matches[i].size() < 2)
            continue;
        const DMatch& m0 = pair_matches[i][0];
        const DMatch& m1 = pair_matches[i][1];
        if (m0.distance < (1.f - match_conf_) * m1.distance)
        {
            matches_info.matches.push_back(m0);
            matches.insert(make_pair(m0.queryIdx, m0.trainIdx));
        }
    }

    // Find 2->1 matches
    pair_matches.clear();
    matcher.knnMatchSingle(descriptors2_, descriptors1_, train_idx_, distance_, all_dist_, 2);
    matcher.knnMatchDownload(train_idx_, distance_, pair_matches);
    for (size_t i = 0; i < pair_matches.size(); ++i)
    {
        if (pair_matches[i].size() < 2)
            continue;
        const DMatch& m0 = pair_matches[i][0];
        const DMatch& m1 = pair_matches[i][1];
        if (m0.distance < (1.f - match_conf_) * m1.distance)
            if (matches.find(make_pair(m0.trainIdx, m0.queryIdx)) == matches.end())
                matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
    }
}

void GpuMatcher::collectGarbage()
{
    descriptors1_.release();
    descriptors2_.release();
    train_idx_.release();
    distance_.release();
    all_dist_.release();
    vector< vector<DMatch> >().swap(pair_matches);
}
#endif

} // namespace


namespace cv {
namespace detail {

void FeaturesFinder::operator ()(const Mat &image, ImageFeatures &features)
{ 
    find(image, features);
    features.img_size = image.size();
}


void FeaturesFinder::operator ()(const Mat &image, ImageFeatures &features, const vector<Rect> &rois)
{ 
    vector<ImageFeatures> roi_features(rois.size());
    size_t total_kps_count = 0;
    int total_descriptors_height = 0;

    for (size_t i = 0; i < rois.size(); ++i)
    {
        find(image(rois[i]), roi_features[i]);
        total_kps_count += roi_features[i].keypoints.size();
        total_descriptors_height += roi_features[i].descriptors.rows;
    }

    features.img_size = image.size();
    features.keypoints.resize(total_kps_count);
    features.descriptors.create(total_descriptors_height, 
                                roi_features[0].descriptors.cols, 
                                roi_features[0].descriptors.type());

    int kp_idx = 0;
    int descr_offset = 0;
    for (size_t i = 0; i < rois.size(); ++i)
    {
        for (size_t j = 0; j < roi_features[i].keypoints.size(); ++j, ++kp_idx)
        {
            features.keypoints[kp_idx] = roi_features[i].keypoints[j];
            features.keypoints[kp_idx].pt.x += (float)rois[i].x;
            features.keypoints[kp_idx].pt.y += (float)rois[i].y;
        }
        Mat subdescr = features.descriptors.rowRange(
                descr_offset, descr_offset + roi_features[i].descriptors.rows);
        roi_features[i].descriptors.copyTo(subdescr);
        descr_offset += roi_features[i].descriptors.rows;
    }
}


SurfFeaturesFinder::SurfFeaturesFinder(double hess_thresh, int num_octaves, int num_layers,
                                       int num_octaves_descr, int num_layers_descr)
{
    if (num_octaves_descr == num_octaves && num_layers_descr == num_layers)
    {
        surf = new SURF(hess_thresh, num_octaves, num_layers);
    }
    else
    {
        detector_ = new SurfFeatureDetector(hess_thresh, num_octaves, num_layers);
        extractor_ = new SurfDescriptorExtractor(num_octaves_descr, num_layers_descr);
    }
}

void SurfFeaturesFinder::find(const Mat &image, ImageFeatures &features)
{
    Mat gray_image;
    CV_Assert(image.type() == CV_8UC3);
    cvtColor(image, gray_image, CV_BGR2GRAY);
    if (surf == 0)
    {
        detector_->detect(gray_image, features.keypoints);
        extractor_->compute(gray_image, features.keypoints, features.descriptors);
    }
    else
    {
        vector<float> descriptors;
        (*surf)(gray_image, Mat(), features.keypoints, descriptors);
        features.descriptors = Mat(descriptors, true).reshape(1, (int)features.keypoints.size());
    }
}

OrbFeaturesFinder::OrbFeaturesFinder(Size _grid_size, size_t n_features, const ORB::CommonParams & detector_params)
{
    grid_size = _grid_size;
    orb = new ORB(n_features * (99 + grid_size.area())/100/grid_size.area(), detector_params);
}

void OrbFeaturesFinder::find(const Mat &image, ImageFeatures &features)
{
    Mat gray_image;
    CV_Assert(image.type() == CV_8UC3);
    cvtColor(image, gray_image, CV_BGR2GRAY);

    if (grid_size.area() == 1)
        (*orb)(gray_image, Mat(), features.keypoints, features.descriptors);
    else
    {
        features.keypoints.clear();
        features.descriptors.release();

        std::vector<KeyPoint> points;
        Mat descriptors;

        for (int r = 0; r < grid_size.height; ++r)
            for (int c = 0; c < grid_size.width; ++c)
            {
                int xl = c * gray_image.cols / grid_size.width;
                int yl = r * gray_image.rows / grid_size.height;
                int xr = (c+1) * gray_image.cols / grid_size.width;
                int yr = (r+1) * gray_image.rows / grid_size.height;

                (*orb)(gray_image(Range(yl, yr), Range(xl, xr)), Mat(), points, descriptors);

                features.keypoints.reserve(features.keypoints.size() + points.size());
                for (std::vector<KeyPoint>::iterator kp = points.begin(); kp != points.end(); ++kp)
                {
                    kp->pt.x += xl;
                    kp->pt.y += yl;
                    features.keypoints.push_back(*kp);
                }
                features.descriptors.push_back(descriptors);
            }
    }
}

#ifndef ANDROID
SurfFeaturesFinderGpu::SurfFeaturesFinderGpu(double hess_thresh, int num_octaves, int num_layers,
                                             int num_octaves_descr, int num_layers_descr)
{
    surf_.keypointsRatio = 0.1f;
    surf_.hessianThreshold = hess_thresh;
    surf_.extended = false;
    num_octaves_ = num_octaves;
    num_layers_ = num_layers;
    num_octaves_descr_ = num_octaves_descr;
    num_layers_descr_ = num_layers_descr;
}


void SurfFeaturesFinderGpu::find(const Mat &image, ImageFeatures &features)
{
    CV_Assert(image.depth() == CV_8U);

    ensureSizeIsEnough(image.size(), image.type(), image_);
    image_.upload(image);

    ensureSizeIsEnough(image.size(), CV_8UC1, gray_image_);
    cvtColor(image_, gray_image_, CV_BGR2GRAY);

    surf_.nOctaves = num_octaves_;
    surf_.nOctaveLayers = num_layers_;
    surf_.upright = false;
    surf_(gray_image_, GpuMat(), keypoints_);

    surf_.nOctaves = num_octaves_descr_;
    surf_.nOctaveLayers = num_layers_descr_;
    surf_.upright = true;
    surf_(gray_image_, GpuMat(), keypoints_, descriptors_, true);
    surf_.downloadKeypoints(keypoints_, features.keypoints);

    descriptors_.download(features.descriptors);
}

void SurfFeaturesFinderGpu::collectGarbage()
{
    surf_.releaseMemory();
    image_.release();
    gray_image_.release();
    keypoints_.release();
    descriptors_.release();
}
#endif


//////////////////////////////////////////////////////////////////////////////

MatchesInfo::MatchesInfo() : src_img_idx(-1), dst_img_idx(-1), num_inliers(0), confidence(0) {}

MatchesInfo::MatchesInfo(const MatchesInfo &other) { *this = other; }

const MatchesInfo& MatchesInfo::operator =(const MatchesInfo &other)
{
    src_img_idx = other.src_img_idx;
    dst_img_idx = other.dst_img_idx;
    matches = other.matches;
    inliers_mask = other.inliers_mask;
    num_inliers = other.num_inliers;
    H = other.H.clone();
    confidence = other.confidence;
    return *this;
}


//////////////////////////////////////////////////////////////////////////////

void FeaturesMatcher::operator ()(const vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches,
                                  const Mat &mask)
{
    const int num_images = static_cast<int>(features.size());

    CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.cols == num_images && mask.rows));
    Mat_<uchar> mask_(mask);
    if (mask_.empty())
        mask_ = Mat::ones(num_images, num_images, CV_8U);

    vector<pair<int,int> > near_pairs;
    for (int i = 0; i < num_images - 1; ++i)
        for (int j = i + 1; j < num_images; ++j)
            if (features[i].keypoints.size() > 0 && features[j].keypoints.size() > 0 && mask_(i, j))
                near_pairs.push_back(make_pair(i, j));

    pairwise_matches.resize(num_images * num_images);
    MatchPairsBody body(*this, features, pairwise_matches, near_pairs);

    if (is_thread_safe_)
        parallel_for(BlockedRange(0, static_cast<int>(near_pairs.size())), body);
    else
        body(BlockedRange(0, static_cast<int>(near_pairs.size())));
    LOGLN("");
}


//////////////////////////////////////////////////////////////////////////////

BestOf2NearestMatcher::BestOf2NearestMatcher(bool try_use_gpu, float match_conf, int num_matches_thresh1, int num_matches_thresh2)
{
#ifndef ANDROID
    if (try_use_gpu && getCudaEnabledDeviceCount() > 0)
        impl_ = new GpuMatcher(match_conf);
    else
#endif
        impl_ = new CpuMatcher(match_conf);

    is_thread_safe_ = impl_->isThreadSafe();
    num_matches_thresh1_ = num_matches_thresh1;
    num_matches_thresh2_ = num_matches_thresh2;
}


void BestOf2NearestMatcher::match(const ImageFeatures &features1, const ImageFeatures &features2,
                                  MatchesInfo &matches_info)
{
    (*impl_)(features1, features2, matches_info);

    // Check if it makes sense to find homography
    if (matches_info.matches.size() < static_cast<size_t>(num_matches_thresh1_))
        return;

    // Construct point-point correspondences for homography estimation
    Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
    Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
    for (size_t i = 0; i < matches_info.matches.size(); ++i)
    {
        const DMatch& m = matches_info.matches[i];

        Point2f p = features1.keypoints[m.queryIdx].pt;
        p.x -= features1.img_size.width * 0.5f;
        p.y -= features1.img_size.height * 0.5f;
        src_points.at<Point2f>(0, static_cast<int>(i)) = p;

        p = features2.keypoints[m.trainIdx].pt;
        p.x -= features2.img_size.width * 0.5f;
        p.y -= features2.img_size.height * 0.5f;
        dst_points.at<Point2f>(0, static_cast<int>(i)) = p;
    }

    // Find pair-wise motion
    matches_info.H = findHomography(src_points, dst_points, matches_info.inliers_mask, CV_RANSAC);
    if (std::abs(determinant(matches_info.H)) < numeric_limits<double>::epsilon())
        return;

    // Find number of inliers
    matches_info.num_inliers = 0;
    for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
        if (matches_info.inliers_mask[i])
            matches_info.num_inliers++;

    // These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching 
    // using Invariant Features"
    matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());

    // Set zero confidence to remove matches between too close images, as they don't provide 
    // additional information anyway. The threshold was set experimentally.
    matches_info.confidence = matches_info.confidence > 3. ? 0. : matches_info.confidence;

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
        p.x -= features1.img_size.width * 0.5f;
        p.y -= features1.img_size.height * 0.5f;
        src_points.at<Point2f>(0, inlier_idx) = p;

        p = features2.keypoints[m.trainIdx].pt;
        p.x -= features2.img_size.width * 0.5f;
        p.y -= features2.img_size.height * 0.5f;
        dst_points.at<Point2f>(0, inlier_idx) = p;

        inlier_idx++;
    }

    // Rerun motion estimation on inliers only
    matches_info.H = findHomography(src_points, dst_points, CV_RANSAC);
}

void BestOf2NearestMatcher::collectGarbage()
{
    impl_->collectGarbage();
}

} // namespace detail
} // namespace cv
