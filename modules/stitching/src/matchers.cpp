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
using namespace cv::gpu;

namespace {

class CpuSurfFeaturesFinder : public FeaturesFinder
{
public:
    CpuSurfFeaturesFinder(double hess_thresh, int num_octaves, int num_layers,
                          int num_octaves_descr, int num_layers_descr)
    {
        detector_ = new SurfFeatureDetector(hess_thresh, num_octaves, num_layers);
        extractor_ = new SurfDescriptorExtractor(num_octaves_descr, num_layers_descr);
    }

protected:
    void find(const Mat &image, ImageFeatures &features);

private:
    Ptr<FeatureDetector> detector_;
    Ptr<DescriptorExtractor> extractor_;
};


class GpuSurfFeaturesFinder : public FeaturesFinder
{
public:
    GpuSurfFeaturesFinder(double hess_thresh, int num_octaves, int num_layers,
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

    void releaseMemory();

protected:
    void find(const Mat &image, ImageFeatures &features);

private:
    GpuMat image_;
    GpuMat gray_image_;
    SURF_GPU surf_;
    GpuMat keypoints_;
    GpuMat descriptors_;
    int num_octaves_, num_layers_;
    int num_octaves_descr_, num_layers_descr_;
};


void CpuSurfFeaturesFinder::find(const Mat &image, ImageFeatures &features)
{
    Mat gray_image;
    CV_Assert(image.depth() == CV_8U);
    cvtColor(image, gray_image, CV_BGR2GRAY);
    detector_->detect(gray_image, features.keypoints);
    extractor_->compute(gray_image, features.keypoints, features.descriptors);
}


void GpuSurfFeaturesFinder::find(const Mat &image, ImageFeatures &features)
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

void GpuSurfFeaturesFinder::releaseMemory()
{
    surf_.releaseMemory();
    image_.release();
    gray_image_.release();
    keypoints_.release();
    descriptors_.release();
}


//////////////////////////////////////////////////////////////////////////////

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


class GpuMatcher : public FeaturesMatcher
{
public:
    GpuMatcher(float match_conf) : match_conf_(match_conf) {}
    void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info);

    void releaseMemory();

private:
    float match_conf_;
    GpuMat descriptors1_, descriptors2_;
    GpuMat train_idx_, distance_, all_dist_;
    vector< vector<DMatch> > pair_matches;
};


void CpuMatcher::match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info)
{
    matches_info.matches.clear();
    FlannBasedMatcher matcher;
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
}


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
    matcher.knnMatch(descriptors1_, descriptors2_, train_idx_, distance_, all_dist_, 2);
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
    matcher.knnMatch(descriptors2_, descriptors1_, train_idx_, distance_, all_dist_, 2);
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

void GpuMatcher::releaseMemory()
{
    descriptors1_.release();
    descriptors2_.release();
    train_idx_.release();
    distance_.release();
    all_dist_.release();
    vector< vector<DMatch> >().swap(pair_matches);
}

} // namespace


namespace cv {
namespace detail {

void FeaturesFinder::operator ()(const Mat &image, ImageFeatures &features)
{ 
    find(image, features);
    features.img_size = image.size();
    //features.img = image.clone();
}


SurfFeaturesFinder::SurfFeaturesFinder(bool try_use_gpu, double hess_thresh, int num_octaves, int num_layers,
                                       int num_octaves_descr, int num_layers_descr)
{
    if (try_use_gpu && getCudaEnabledDeviceCount() > 0)
        impl_ = new GpuSurfFeaturesFinder(hess_thresh, num_octaves, num_layers, num_octaves_descr, num_layers_descr);
    else
        impl_ = new CpuSurfFeaturesFinder(hess_thresh, num_octaves, num_layers, num_octaves_descr, num_layers_descr);
}


void SurfFeaturesFinder::find(const Mat &image, ImageFeatures &features)
{
    (*impl_)(image, features);
}


void SurfFeaturesFinder::releaseMemory()
{
    impl_->releaseMemory();
}


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

void FeaturesMatcher::operator ()(const vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches)
{
    const int num_images = static_cast<int>(features.size());

    vector<pair<int,int> > near_pairs;
    for (int i = 0; i < num_images - 1; ++i)
        for (int j = i + 1; j < num_images; ++j)
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
    if (try_use_gpu && getCudaEnabledDeviceCount() > 0)
        impl_ = new GpuMatcher(match_conf);
    else
        impl_ = new CpuMatcher(match_conf);

    is_thread_safe_ = impl_->isThreadSafe();
    num_matches_thresh1_ = num_matches_thresh1;
    num_matches_thresh2_ = num_matches_thresh2;
}


void BestOf2NearestMatcher::match(const ImageFeatures &features1, const ImageFeatures &features2,
                                  MatchesInfo &matches_info)
{
    (*impl_)(features1, features2, matches_info);

    //Mat out;
    //drawMatches(features1.img, features1.keypoints, features2.img, features2.keypoints, matches_info.matches, out);
    //stringstream ss;
    //ss << features1.img_idx << features2.img_idx << ".png";
    //imwrite(ss.str(), out);

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

    matches_info.confidence = matches_info.num_inliers / (8 + 0.3*matches_info.matches.size());

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

void BestOf2NearestMatcher::releaseMemory()
{
    impl_->releaseMemory();
}

} // namespace detail
} // namespace cv
