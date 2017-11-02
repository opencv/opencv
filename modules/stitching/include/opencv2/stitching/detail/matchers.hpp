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

#ifndef OPENCV_STITCHING_MATCHERS_HPP
#define OPENCV_STITCHING_MATCHERS_HPP

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#  include "opencv2/xfeatures2d/cuda.hpp"
#endif

namespace cv {
namespace detail {

//! @addtogroup stitching_match
//! @{

/** @brief Structure containing image keypoints and descriptors. */
struct CV_EXPORTS ImageFeatures
{
    int img_idx;
    Size img_size;
    std::vector<KeyPoint> keypoints;
    UMat descriptors;
};

/** @brief Feature finders base class */
class CV_EXPORTS FeaturesFinder
{
public:
    virtual ~FeaturesFinder() {}
    /** @overload */
    void operator ()(InputArray image, ImageFeatures &features);
    /** @brief Finds features in the given image.

    @param image Source image
    @param features Found features
    @param rois Regions of interest

    @sa detail::ImageFeatures, Rect_
    */
    void operator ()(InputArray image, ImageFeatures &features, const std::vector<cv::Rect> &rois);
    /** @brief Finds features in the given images in parallel.

    @param images Source images
    @param features Found features for each image
    @param rois Regions of interest for each image

    @sa detail::ImageFeatures, Rect_
    */
    void operator ()(InputArrayOfArrays images, std::vector<ImageFeatures> &features,
                     const std::vector<std::vector<cv::Rect> > &rois);
    /** @overload */
    void operator ()(InputArrayOfArrays images, std::vector<ImageFeatures> &features);
    /** @brief Frees unused memory allocated before if there is any. */
    virtual void collectGarbage() {}

    /* TODO OpenCV ABI 4.x
    reimplement this as public method similar to FeaturesMatcher and remove private function hack
    @return True, if it's possible to use the same finder instance in parallel, false otherwise
    bool isThreadSafe() const { return is_thread_safe_; }
    */

protected:
    /** @brief This method must implement features finding logic in order to make the wrappers
    detail::FeaturesFinder::operator()_ work.

    @param image Source image
    @param features Found features

    @sa detail::ImageFeatures */
    virtual void find(InputArray image, ImageFeatures &features) = 0;
    /** @brief uses dynamic_cast to determine thread-safety
    @return True, if it's possible to use the same finder instance in parallel, false otherwise
    */
    bool isThreadSafe() const;
};

/** @brief SURF features finder.

@sa detail::FeaturesFinder, SURF
*/
class CV_EXPORTS SurfFeaturesFinder : public FeaturesFinder
{
public:
    SurfFeaturesFinder(double hess_thresh = 300., int num_octaves = 3, int num_layers = 4,
                       int num_octaves_descr = /*4*/3, int num_layers_descr = /*2*/4);

private:
    void find(InputArray image, ImageFeatures &features);

    Ptr<FeatureDetector> detector_;
    Ptr<DescriptorExtractor> extractor_;
    Ptr<Feature2D> surf;
};

/** @brief ORB features finder. :

@sa detail::FeaturesFinder, ORB
*/
class CV_EXPORTS OrbFeaturesFinder : public FeaturesFinder
{
public:
    OrbFeaturesFinder(Size _grid_size = Size(3,1), int nfeatures=1500, float scaleFactor=1.3f, int nlevels=5);

private:
    void find(InputArray image, ImageFeatures &features);

    Ptr<ORB> orb;
    Size grid_size;
};

/** @brief AKAZE features finder. :

@sa detail::FeaturesFinder, AKAZE
*/
class CV_EXPORTS AKAZEFeaturesFinder : public detail::FeaturesFinder
{
public:
    AKAZEFeaturesFinder(int descriptor_type = AKAZE::DESCRIPTOR_MLDB,
                        int descriptor_size = 0,
                        int descriptor_channels = 3,
                        float threshold = 0.001f,
                        int nOctaves = 4,
                        int nOctaveLayers = 4,
                        int diffusivity = KAZE::DIFF_PM_G2);

private:
    void find(InputArray image, detail::ImageFeatures &features);

    Ptr<AKAZE> akaze;
};

#ifdef HAVE_OPENCV_XFEATURES2D
class CV_EXPORTS SurfFeaturesFinderGpu : public FeaturesFinder
{
public:
    SurfFeaturesFinderGpu(double hess_thresh = 300., int num_octaves = 3, int num_layers = 4,
                          int num_octaves_descr = 4, int num_layers_descr = 2);

    void collectGarbage();

private:
    void find(InputArray image, ImageFeatures &features);

    cuda::GpuMat image_;
    cuda::GpuMat gray_image_;
    cuda::SURF_CUDA surf_;
    cuda::GpuMat keypoints_;
    cuda::GpuMat descriptors_;
    int num_octaves_, num_layers_;
    int num_octaves_descr_, num_layers_descr_;
};
#endif

/** @brief Structure containing information about matches between two images.

It's assumed that there is a transformation between those images. Transformation may be
homography or affine transformation based on selected matcher.

@sa detail::FeaturesMatcher
*/
struct CV_EXPORTS MatchesInfo
{
    MatchesInfo();
    MatchesInfo(const MatchesInfo &other);
    MatchesInfo& operator =(const MatchesInfo &other);

    int src_img_idx, dst_img_idx;       //!< Images indices (optional)
    std::vector<DMatch> matches;
    std::vector<uchar> inliers_mask;    //!< Geometrically consistent matches mask
    int num_inliers;                    //!< Number of geometrically consistent matches
    Mat H;                              //!< Estimated transformation
    double confidence;                  //!< Confidence two images are from the same panorama
};

/** @brief Feature matchers base class. */
class CV_EXPORTS FeaturesMatcher
{
public:
    virtual ~FeaturesMatcher() {}

    /** @overload
    @param features1 First image features
    @param features2 Second image features
    @param matches_info Found matches
    */
    void operator ()(const ImageFeatures &features1, const ImageFeatures &features2,
                     MatchesInfo& matches_info) { match(features1, features2, matches_info); }

    /** @brief Performs images matching.

    @param features Features of the source images
    @param pairwise_matches Found pairwise matches
    @param mask Mask indicating which image pairs must be matched

    The function is parallelized with the TBB library.

    @sa detail::MatchesInfo
    */
    void operator ()(const std::vector<ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches,
                     const cv::UMat &mask = cv::UMat());

    /** @return True, if it's possible to use the same matcher instance in parallel, false otherwise
    */
    bool isThreadSafe() const { return is_thread_safe_; }

    /** @brief Frees unused memory allocated before if there is any.
    */
    virtual void collectGarbage() {}

protected:
    FeaturesMatcher(bool is_thread_safe = false) : is_thread_safe_(is_thread_safe) {}

    /** @brief This method must implement matching logic in order to make the wrappers
    detail::FeaturesMatcher::operator()_ work.

    @param features1 first image features
    @param features2 second image features
    @param matches_info found matches
     */
    virtual void match(const ImageFeatures &features1, const ImageFeatures &features2,
                       MatchesInfo& matches_info) = 0;

    bool is_thread_safe_;
};

/** @brief Features matcher which finds two best matches for each feature and leaves the best one only if the
ratio between descriptor distances is greater than the threshold match_conf

@sa detail::FeaturesMatcher
 */
class CV_EXPORTS BestOf2NearestMatcher : public FeaturesMatcher
{
public:
    /** @brief Constructs a "best of 2 nearest" matcher.

    @param try_use_gpu Should try to use GPU or not
    @param match_conf Match distances ration threshold
    @param num_matches_thresh1 Minimum number of matches required for the 2D projective transform
    estimation used in the inliers classification step
    @param num_matches_thresh2 Minimum number of matches required for the 2D projective transform
    re-estimation on inliers
     */
    BestOf2NearestMatcher(bool try_use_gpu = false, float match_conf = 0.3f, int num_matches_thresh1 = 6,
                          int num_matches_thresh2 = 6);

    void collectGarbage();

protected:
    void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo &matches_info);

    int num_matches_thresh1_;
    int num_matches_thresh2_;
    Ptr<FeaturesMatcher> impl_;
};

class CV_EXPORTS BestOf2NearestRangeMatcher : public BestOf2NearestMatcher
{
public:
    BestOf2NearestRangeMatcher(int range_width = 5, bool try_use_gpu = false, float match_conf = 0.3f,
                            int num_matches_thresh1 = 6, int num_matches_thresh2 = 6);

    void operator ()(const std::vector<ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches,
                     const cv::UMat &mask = cv::UMat());


protected:
    int range_width_;
};

/** @brief Features matcher similar to cv::detail::BestOf2NearestMatcher which
finds two best matches for each feature and leaves the best one only if the
ratio between descriptor distances is greater than the threshold match_conf.

Unlike cv::detail::BestOf2NearestMatcher this matcher uses affine
transformation (affine trasformation estimate will be placed in matches_info).

@sa cv::detail::FeaturesMatcher cv::detail::BestOf2NearestMatcher
 */
class CV_EXPORTS AffineBestOf2NearestMatcher : public BestOf2NearestMatcher
{
public:
    /** @brief Constructs a "best of 2 nearest" matcher that expects affine trasformation
    between images

    @param full_affine whether to use full affine transformation with 6 degress of freedom or reduced
    transformation with 4 degrees of freedom using only rotation, translation and uniform scaling
    @param try_use_gpu Should try to use GPU or not
    @param match_conf Match distances ration threshold
    @param num_matches_thresh1 Minimum number of matches required for the 2D affine transform
    estimation used in the inliers classification step

    @sa cv::estimateAffine2D cv::estimateAffinePartial2D
     */
    AffineBestOf2NearestMatcher(bool full_affine = false, bool try_use_gpu = false,
                                float match_conf = 0.3f, int num_matches_thresh1 = 6) :
        BestOf2NearestMatcher(try_use_gpu, match_conf, num_matches_thresh1, num_matches_thresh1),
        full_affine_(full_affine) {}

protected:
    void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo &matches_info);

    bool full_affine_;
};

//! @} stitching_match

} // namespace detail
} // namespace cv

#endif // OPENCV_STITCHING_MATCHERS_HPP
