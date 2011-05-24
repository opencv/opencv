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
#ifndef __OPENCV_MATCHERS_HPP__
#define __OPENCV_MATCHERS_HPP__

#include "precomp.hpp"

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

    void operator ()(const ImageFeatures &features1, const ImageFeatures &features2, 
                     MatchesInfo& matches_info) { match(features1, features2, matches_info); }
    void operator ()(const std::vector<ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches);

    bool isThreadSafe() const { return is_thread_safe_; }

protected:
    FeaturesMatcher(bool is_thread_safe = false) : is_thread_safe_(is_thread_safe) {}

    virtual void match(const ImageFeatures &features1, const ImageFeatures &features2, 
                       MatchesInfo& matches_info) = 0;

    bool is_thread_safe_;
};


class BestOf2NearestMatcher : public FeaturesMatcher
{
public:
    BestOf2NearestMatcher(bool try_use_gpu = true, float match_conf = 0.55f, int num_matches_thresh1 = 6, 
                          int num_matches_thresh2 = 6);

protected:
    void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo &matches_info);

    int num_matches_thresh1_;
    int num_matches_thresh2_;
    cv::Ptr<FeaturesMatcher> impl_;
};

#endif // __OPENCV_MATCHERS_HPP__