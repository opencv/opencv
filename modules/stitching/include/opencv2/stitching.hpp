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

#ifndef __OPENCV_STITCHING_STITCHER_HPP__
#define __OPENCV_STITCHING_STITCHER_HPP__

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"

/**
@defgroup stitching Images stitching

This figure illustrates the stitching module pipeline implemented in the Stitcher class. Using that
class it's possible to configure/remove some steps, i.e. adjust the stitching pipeline according to
the particular needs. All building blocks from the pipeline are available in the detail namespace,
one can combine and use them separately.

The implemented stitching pipeline is very similar to the one proposed in @cite BL07 .

![image](StitchingPipeline.jpg)

@{
    @defgroup stitching_match Features Finding and Images Matching
    @defgroup stitching_rotation Rotation Estimation
    @defgroup stitching_autocalib Autocalibration
    @defgroup stitching_warp Images Warping
    @defgroup stitching_seam Seam Estimation
    @defgroup stitching_exposure Exposure Compensation
    @defgroup stitching_blend Image Blenders
@}
  */

namespace cv {

//! @addtogroup stitching
//! @{

/** @brief High level image stitcher.

It's possible to use this class without being aware of the entire stitching pipeline. However, to
be able to achieve higher stitching stability and quality of the final images at least being
familiar with the theory is recommended.

@note
   -   A basic example on image stitching can be found at
        opencv_source_code/samples/cpp/stitching.cpp
    -   A detailed example on image stitching can be found at
        opencv_source_code/samples/cpp/stitching_detailed.cpp
 */
class CV_EXPORTS_W Stitcher
{
public:
    enum { ORIG_RESOL = -1 };
    enum Status
    {
        OK = 0,
        ERR_NEED_MORE_IMGS = 1,
        ERR_HOMOGRAPHY_EST_FAIL = 2,
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
    };

   // Stitcher() {}
    /** @brief Creates a stitcher with the default parameters.

    @param try_use_gpu Flag indicating whether GPU should be used whenever it's possible.
    @return Stitcher class instance.
     */
    static Stitcher createDefault(bool try_use_gpu = false);

    CV_WRAP double registrationResol() const { return registr_resol_; }
    CV_WRAP void setRegistrationResol(double resol_mpx) { registr_resol_ = resol_mpx; }

    CV_WRAP double seamEstimationResol() const { return seam_est_resol_; }
    CV_WRAP void setSeamEstimationResol(double resol_mpx) { seam_est_resol_ = resol_mpx; }

    CV_WRAP double compositingResol() const { return compose_resol_; }
    CV_WRAP void setCompositingResol(double resol_mpx) { compose_resol_ = resol_mpx; }

    CV_WRAP double panoConfidenceThresh() const { return conf_thresh_; }
    CV_WRAP void setPanoConfidenceThresh(double conf_thresh) { conf_thresh_ = conf_thresh; }

    CV_WRAP bool waveCorrection() const { return do_wave_correct_; }
    CV_WRAP void setWaveCorrection(bool flag) { do_wave_correct_ = flag; }

    detail::WaveCorrectKind waveCorrectKind() const { return wave_correct_kind_; }
    void setWaveCorrectKind(detail::WaveCorrectKind kind) { wave_correct_kind_ = kind; }

    Ptr<detail::FeaturesFinder> featuresFinder() { return features_finder_; }
    const Ptr<detail::FeaturesFinder> featuresFinder() const { return features_finder_; }
    void setFeaturesFinder(Ptr<detail::FeaturesFinder> features_finder)
        { features_finder_ = features_finder; }

    Ptr<detail::FeaturesMatcher> featuresMatcher() { return features_matcher_; }
    const Ptr<detail::FeaturesMatcher> featuresMatcher() const { return features_matcher_; }
    void setFeaturesMatcher(Ptr<detail::FeaturesMatcher> features_matcher)
        { features_matcher_ = features_matcher; }

    const cv::UMat& matchingMask() const { return matching_mask_; }
    void setMatchingMask(const cv::UMat &mask)
    {
        CV_Assert(mask.type() == CV_8U && mask.cols == mask.rows);
        matching_mask_ = mask.clone();
    }

    Ptr<detail::BundleAdjusterBase> bundleAdjuster() { return bundle_adjuster_; }
    const Ptr<detail::BundleAdjusterBase> bundleAdjuster() const { return bundle_adjuster_; }
    void setBundleAdjuster(Ptr<detail::BundleAdjusterBase> bundle_adjuster)
        { bundle_adjuster_ = bundle_adjuster; }

    Ptr<WarperCreator> warper() { return warper_; }
    const Ptr<WarperCreator> warper() const { return warper_; }
    void setWarper(Ptr<WarperCreator> creator) { warper_ = creator; }

    Ptr<detail::ExposureCompensator> exposureCompensator() { return exposure_comp_; }
    const Ptr<detail::ExposureCompensator> exposureCompensator() const { return exposure_comp_; }
    void setExposureCompensator(Ptr<detail::ExposureCompensator> exposure_comp)
        { exposure_comp_ = exposure_comp; }

    Ptr<detail::SeamFinder> seamFinder() { return seam_finder_; }
    const Ptr<detail::SeamFinder> seamFinder() const { return seam_finder_; }
    void setSeamFinder(Ptr<detail::SeamFinder> seam_finder) { seam_finder_ = seam_finder; }

    Ptr<detail::Blender> blender() { return blender_; }
    const Ptr<detail::Blender> blender() const { return blender_; }
    void setBlender(Ptr<detail::Blender> b) { blender_ = b; }

    /** @overload */
    CV_WRAP Status estimateTransform(InputArrayOfArrays images);
    /** @brief These functions try to match the given images and to estimate rotations of each camera.

    @note Use the functions only if you're aware of the stitching pipeline, otherwise use
    Stitcher::stitch.

    @param images Input images.
    @param rois Region of interest rectangles.
    @return Status code.
     */
    Status estimateTransform(InputArrayOfArrays images, const std::vector<std::vector<Rect> > &rois);

    /** @overload */
    CV_WRAP Status composePanorama(OutputArray pano);
    /** @brief These functions try to compose the given images (or images stored internally from the other function
    calls) into the final pano under the assumption that the image transformations were estimated
    before.

    @note Use the functions only if you're aware of the stitching pipeline, otherwise use
    Stitcher::stitch.

    @param images Input images.
    @param pano Final pano.
    @return Status code.
     */
    Status composePanorama(InputArrayOfArrays images, OutputArray pano);

    /** @overload */
    CV_WRAP Status stitch(InputArrayOfArrays images, OutputArray pano);
    /** @brief These functions try to stitch the given images.

    @param images Input images.
    @param rois Region of interest rectangles.
    @param pano Final pano.
    @return Status code.
     */
    Status stitch(InputArrayOfArrays images, const std::vector<std::vector<Rect> > &rois, OutputArray pano);

    std::vector<int> component() const { return indices_; }
    std::vector<detail::CameraParams> cameras() const { return cameras_; }
    CV_WRAP double workScale() const { return work_scale_; }

private:
    //Stitcher() {}

    Status matchImages();
    Status estimateCameraParams();

    double registr_resol_;
    double seam_est_resol_;
    double compose_resol_;
    double conf_thresh_;
    Ptr<detail::FeaturesFinder> features_finder_;
    Ptr<detail::FeaturesMatcher> features_matcher_;
    cv::UMat matching_mask_;
    Ptr<detail::BundleAdjusterBase> bundle_adjuster_;
    bool do_wave_correct_;
    detail::WaveCorrectKind wave_correct_kind_;
    Ptr<WarperCreator> warper_;
    Ptr<detail::ExposureCompensator> exposure_comp_;
    Ptr<detail::SeamFinder> seam_finder_;
    Ptr<detail::Blender> blender_;

    std::vector<cv::UMat> imgs_;
    std::vector<std::vector<cv::Rect> > rois_;
    std::vector<cv::Size> full_img_sizes_;
    std::vector<detail::ImageFeatures> features_;
    std::vector<detail::MatchesInfo> pairwise_matches_;
    std::vector<cv::UMat> seam_est_imgs_;
    std::vector<int> indices_;
    std::vector<detail::CameraParams> cameras_;
    double work_scale_;
    double seam_scale_;
    double seam_work_aspect_;
    double warped_image_scale_;
};

CV_EXPORTS_W Ptr<Stitcher> createStitcher(bool try_use_gpu = false);

//! @} stitching

} // namespace cv

#endif // __OPENCV_STITCHING_STITCHER_HPP__
