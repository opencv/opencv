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

#ifndef OPENCV_STITCHING_MOTION_ESTIMATORS_HPP
#define OPENCV_STITCHING_MOTION_ESTIMATORS_HPP

#include "opencv2/core.hpp"
#include "matchers.hpp"
#include "util.hpp"
#include "camera.hpp"

namespace cv {
namespace detail {

//! @addtogroup stitching_rotation
//! @{

/** @brief Rotation estimator base class.

It takes features of all images, pairwise matches between all images and estimates rotations of all
cameras.

@note The coordinate system origin is implementation-dependent, but you can always normalize the
rotations in respect to the first camera, for instance. :
 */
class CV_EXPORTS_W Estimator
{
public:
    virtual ~Estimator() {}

    /** @brief Estimates camera parameters.

    @param features Features of images
    @param pairwise_matches Pairwise matches of images
    @param cameras Estimated camera parameters
    @return True in case of success, false otherwise
     */
    CV_WRAP_AS(apply) bool operator ()(const std::vector<ImageFeatures> &features,
        const std::vector<MatchesInfo> &pairwise_matches,
        CV_OUT CV_IN_OUT std::vector<CameraParams> &cameras)
    {
        return estimate(features, pairwise_matches, cameras);
    }

protected:
    /** @brief This method must implement camera parameters estimation logic in order to make the wrapper
    detail::Estimator::operator()_ work.

    @param features Features of images
    @param pairwise_matches Pairwise matches of images
    @param cameras Estimated camera parameters
    @return True in case of success, false otherwise
     */
    virtual bool estimate(const std::vector<ImageFeatures> &features,
                          const std::vector<MatchesInfo> &pairwise_matches,
                          CV_OUT std::vector<CameraParams> &cameras) = 0;
};

/** @brief Homography based rotation estimator.
 */
class CV_EXPORTS_W HomographyBasedEstimator : public Estimator
{
public:
    CV_WRAP HomographyBasedEstimator(bool is_focals_estimated = false)
        : is_focals_estimated_(is_focals_estimated) {}

private:
    virtual bool estimate(const std::vector<ImageFeatures> &features,
                          const std::vector<MatchesInfo> &pairwise_matches,
                          std::vector<CameraParams> &cameras) CV_OVERRIDE;

    bool is_focals_estimated_;
};

/** @brief Affine transformation based estimator.

This estimator uses pairwise transformations estimated by matcher to estimate
final transformation for each camera.

@sa cv::detail::HomographyBasedEstimator
 */
class CV_EXPORTS_W AffineBasedEstimator : public Estimator
{
public:
    CV_WRAP AffineBasedEstimator(){}
private:
    virtual bool estimate(const std::vector<ImageFeatures> &features,
                          const std::vector<MatchesInfo> &pairwise_matches,
                          std::vector<CameraParams> &cameras) CV_OVERRIDE;
};

/** @brief Base class for all camera parameters refinement methods.
 */
class CV_EXPORTS_W BundleAdjusterBase : public Estimator
{
public:
    CV_WRAP Mat refinementMask() const { return refinement_mask_.clone(); }
    CV_WRAP void setRefinementMask(const Mat &mask)
    {
        CV_Assert(mask.type() == CV_8U && mask.size() == Size(3, 3));
        refinement_mask_ = mask.clone();
    }

    CV_WRAP double confThresh() const { return conf_thresh_; }
    CV_WRAP void setConfThresh(double conf_thresh) { conf_thresh_ = conf_thresh; }

    CV_WRAP TermCriteria termCriteria() { return term_criteria_; }
    CV_WRAP void setTermCriteria(const TermCriteria& term_criteria) { term_criteria_ = term_criteria; }

protected:
    /** @brief Construct a bundle adjuster base instance.

    @param num_params_per_cam Number of parameters per camera
    @param num_errs_per_measurement Number of error terms (components) per match
     */
    BundleAdjusterBase(int num_params_per_cam, int num_errs_per_measurement)
        : num_images_(0), total_num_matches_(0),
          num_params_per_cam_(num_params_per_cam),
          num_errs_per_measurement_(num_errs_per_measurement),
          features_(0), pairwise_matches_(0), conf_thresh_(0)
    {
        setRefinementMask(Mat::ones(3, 3, CV_8U));
        setConfThresh(1.);
        setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 1000, DBL_EPSILON));
    }

    // Runs bundle adjustment
    virtual bool estimate(const std::vector<ImageFeatures> &features,
                          const std::vector<MatchesInfo> &pairwise_matches,
                          std::vector<CameraParams> &cameras) CV_OVERRIDE;

    /** @brief Sets initial camera parameter to refine.

    @param cameras Camera parameters
     */
    virtual void setUpInitialCameraParams(const std::vector<CameraParams> &cameras) = 0;
    /** @brief Gets the refined camera parameters.

    @param cameras Refined camera parameters
     */
    virtual void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const = 0;
    /** @brief Calculates error vector.

    @param err Error column-vector of length total_num_matches \* num_errs_per_measurement
     */
    virtual void calcError(Mat &err) = 0;
    /** @brief Calculates the cost function jacobian.

    @param jac Jacobian matrix of dimensions
    (total_num_matches \* num_errs_per_measurement) x (num_images \* num_params_per_cam)
     */
    virtual void calcJacobian(Mat &jac) = 0;

    // 3x3 8U mask, where 0 means don't refine respective parameter, != 0 means refine
    Mat refinement_mask_;

    int num_images_;
    int total_num_matches_;

    int num_params_per_cam_;
    int num_errs_per_measurement_;

    const ImageFeatures *features_;
    const MatchesInfo *pairwise_matches_;

    // Threshold to filter out poorly matched image pairs
    double conf_thresh_;

    //Levenberg-Marquardt algorithm termination criteria
    TermCriteria term_criteria_;

    // Camera parameters matrix (CV_64F)
    Mat cam_params_;

    // Connected images pairs
    std::vector<std::pair<int,int> > edges_;
};


/** @brief Stub bundle adjuster that does nothing.
 */
class CV_EXPORTS_W NoBundleAdjuster : public BundleAdjusterBase
{
public:
    CV_WRAP NoBundleAdjuster() : BundleAdjusterBase(0, 0) {}

private:
    bool estimate(const std::vector<ImageFeatures> &, const std::vector<MatchesInfo> &,
                  std::vector<CameraParams> &) CV_OVERRIDE
    {
        return true;
    }
    void setUpInitialCameraParams(const std::vector<CameraParams> &) CV_OVERRIDE {}
    void obtainRefinedCameraParams(std::vector<CameraParams> &) const CV_OVERRIDE {}
    void calcError(Mat &) CV_OVERRIDE {}
    void calcJacobian(Mat &) CV_OVERRIDE {}
};


/** @brief Implementation of the camera parameters refinement algorithm which minimizes sum of the reprojection
error squares

It can estimate focal length, aspect ratio, principal point.
You can affect only on them via the refinement mask.
 */
class CV_EXPORTS_W BundleAdjusterReproj : public BundleAdjusterBase
{
public:
    CV_WRAP BundleAdjusterReproj() : BundleAdjusterBase(7, 2) {}

private:
    void setUpInitialCameraParams(const std::vector<CameraParams> &cameras) CV_OVERRIDE;
    void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const CV_OVERRIDE;
    void calcError(Mat &err) CV_OVERRIDE;
    void calcJacobian(Mat &jac) CV_OVERRIDE;

    Mat err1_, err2_;
};


/** @brief Implementation of the camera parameters refinement algorithm which minimizes sum of the distances
between the rays passing through the camera center and a feature. :

It can estimate focal length. It ignores the refinement mask for now.
 */
class CV_EXPORTS_W BundleAdjusterRay : public BundleAdjusterBase
{
public:
    CV_WRAP BundleAdjusterRay() : BundleAdjusterBase(4, 3) {}

private:
    void setUpInitialCameraParams(const std::vector<CameraParams> &cameras) CV_OVERRIDE;
    void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const CV_OVERRIDE;
    void calcError(Mat &err) CV_OVERRIDE;
    void calcJacobian(Mat &jac) CV_OVERRIDE;

    Mat err1_, err2_;
};


/** @brief Bundle adjuster that expects affine transformation
represented in homogeneous coordinates in R for each camera param. Implements
camera parameters refinement algorithm which minimizes sum of the reprojection
error squares

It estimates all transformation parameters. Refinement mask is ignored.

@sa AffineBasedEstimator AffineBestOf2NearestMatcher BundleAdjusterAffinePartial
 */
class CV_EXPORTS_W BundleAdjusterAffine : public BundleAdjusterBase
{
public:
    CV_WRAP BundleAdjusterAffine() : BundleAdjusterBase(6, 2) {}

private:
    void setUpInitialCameraParams(const std::vector<CameraParams> &cameras) CV_OVERRIDE;
    void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const CV_OVERRIDE;
    void calcError(Mat &err) CV_OVERRIDE;
    void calcJacobian(Mat &jac) CV_OVERRIDE;

    Mat err1_, err2_;
};


/** @brief Bundle adjuster that expects affine transformation with 4 DOF
represented in homogeneous coordinates in R for each camera param. Implements
camera parameters refinement algorithm which minimizes sum of the reprojection
error squares

It estimates all transformation parameters. Refinement mask is ignored.

@sa AffineBasedEstimator AffineBestOf2NearestMatcher BundleAdjusterAffine
 */
class CV_EXPORTS_W BundleAdjusterAffinePartial : public BundleAdjusterBase
{
public:
    CV_WRAP BundleAdjusterAffinePartial() : BundleAdjusterBase(4, 2) {}

private:
    void setUpInitialCameraParams(const std::vector<CameraParams> &cameras) CV_OVERRIDE;
    void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const CV_OVERRIDE;
    void calcError(Mat &err) CV_OVERRIDE;
    void calcJacobian(Mat &jac) CV_OVERRIDE;

    Mat err1_, err2_;
};


enum WaveCorrectKind
{
    WAVE_CORRECT_HORIZ,
    WAVE_CORRECT_VERT,
    WAVE_CORRECT_AUTO
};

/** @brief Tries to detect the wave correction kind depending
on whether a panorama spans horizontally or vertically

@param rmats Camera rotation matrices.
@return The correction kind to use for this panorama
 */
CV_EXPORTS
WaveCorrectKind autoDetectWaveCorrectKind(const std::vector<Mat> &rmats);

/** @brief Tries to make panorama more horizontal (or vertical).

@param rmats Camera rotation matrices.
@param kind Correction kind, see detail::WaveCorrectKind.
 */
void CV_EXPORTS_W waveCorrect(CV_IN_OUT std::vector<Mat> &rmats, WaveCorrectKind kind);


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

// Returns matches graph representation in DOT language
String CV_EXPORTS_W matchesGraphAsString(std::vector<String> &pathes, std::vector<MatchesInfo> &pairwise_matches,
                                            float conf_threshold);

CV_EXPORTS_W std::vector<int>  leaveBiggestComponent(
        std::vector<ImageFeatures> &features,
        std::vector<MatchesInfo> &pairwise_matches,
        float conf_threshold);

void CV_EXPORTS findMaxSpanningTree(
        int num_images, const std::vector<MatchesInfo> &pairwise_matches,
        Graph &span_tree, std::vector<int> &centers);

//! @} stitching_rotation

} // namespace detail
} // namespace cv

#endif // OPENCV_STITCHING_MOTION_ESTIMATORS_HPP
