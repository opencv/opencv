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

#ifndef __OPENCV_STITCHING_MATCHERS_HPP__
#define __OPENCV_STITCHING_MATCHERS_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/core/gpumat.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_NONFREE)
    #include "opencv2/nonfree/gpu.hpp"
#endif

namespace cv {
namespace detail {

struct CV_EXPORTS ImageFeatures
{
    int img_idx;
    Size img_size;
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
};


class CV_EXPORTS FeaturesFinder
{
public:
    virtual ~FeaturesFinder() {}
    void operator ()(const Mat &image, ImageFeatures &features);
    void operator ()(const Mat &image, ImageFeatures &features, const std::vector<cv::Rect> &rois);
    virtual void collectGarbage() {}

protected:
    virtual void find(const Mat &image, ImageFeatures &features) = 0;
};


class CV_EXPORTS SurfFeaturesFinder : public FeaturesFinder
{
public:
    SurfFeaturesFinder(double hess_thresh = 300., int num_octaves = 3, int num_layers = 4,
                       int num_octaves_descr = /*4*/3, int num_layers_descr = /*2*/4);

private:
    void find(const Mat &image, ImageFeatures &features);

    Ptr<FeatureDetector> detector_;
    Ptr<DescriptorExtractor> extractor_;
    Ptr<Feature2D> surf;
};

class CV_EXPORTS OrbFeaturesFinder : public FeaturesFinder
{
public:
    OrbFeaturesFinder(Size _grid_size = Size(3,1), int nfeatures=1500, float scaleFactor=1.3f, int nlevels=5);

private:
    void find(const Mat &image, ImageFeatures &features);

    Ptr<ORB> orb;
    Size grid_size;
};


#if defined(HAVE_OPENCV_NONFREE)
class CV_EXPORTS SurfFeaturesFinderGpu : public FeaturesFinder
{
public:
    SurfFeaturesFinderGpu(double hess_thresh = 300., int num_octaves = 3, int num_layers = 4,
                          int num_octaves_descr = 4, int num_layers_descr = 2);

    void collectGarbage();

private:
    void find(const Mat &image, ImageFeatures &features);

    gpu::GpuMat image_;
    gpu::GpuMat gray_image_;
    gpu::SURF_GPU surf_;
    gpu::GpuMat keypoints_;
    gpu::GpuMat descriptors_;
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    int num_octaves_, num_layers_;
    int num_octaves_descr_, num_layers_descr_;
#endif
};
#endif


struct CV_EXPORTS MatchesInfo
{
    MatchesInfo();
    MatchesInfo(const MatchesInfo &other);
    const MatchesInfo& operator =(const MatchesInfo &other);

    int src_img_idx, dst_img_idx;       // Images indices (optional)
    std::vector<DMatch> matches;
    std::vector<uchar> inliers_mask;    // Geometrically consistent matches mask
    int num_inliers;                    // Number of geometrically consistent matches
    Mat H;                              // Estimated homography
    double confidence;                  // Confidence two images are from the same panorama
};


class CV_EXPORTS FeaturesMatcher
{
public:
    virtual ~FeaturesMatcher() {}

    void operator ()(const ImageFeatures &features1, const ImageFeatures &features2,
                     MatchesInfo& matches_info) { match(features1, features2, matches_info); }

    void operator ()(const std::vector<ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches,
                     const cv::Mat &mask = cv::Mat());

    bool isThreadSafe() const { return is_thread_safe_; }

    virtual void collectGarbage() {}

protected:
    FeaturesMatcher(bool is_thread_safe = false) : is_thread_safe_(is_thread_safe) {}

    virtual void match(const ImageFeatures &features1, const ImageFeatures &features2,
                       MatchesInfo& matches_info) = 0;

    bool is_thread_safe_;
};


class CV_EXPORTS BestOf2NearestMatcher : public FeaturesMatcher
{
public:
    BestOf2NearestMatcher(bool try_use_gpu = false, float match_conf = 0.3f, int num_matches_thresh1 = 6,
                          int num_matches_thresh2 = 6);

    void collectGarbage();

protected:
    void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo &matches_info);

    int num_matches_thresh1_;
    int num_matches_thresh2_;
    Ptr<FeaturesMatcher> impl_;
};

} // namespace detail
} // namespace cv

#endif // __OPENCV_STITCHING_MATCHERS_HPP__
