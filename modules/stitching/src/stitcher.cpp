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

namespace cv {

Stitcher Stitcher::createDefault(bool try_use_gpu)
{
    Stitcher stitcher;
    stitcher.setRegistrationResol(0.6);
    stitcher.setSeamEstimationResol(0.1);
    stitcher.setCompositingResol(ORIG_RESOL);
    stitcher.setPanoConfidenceThresh(1);
    stitcher.setWaveCorrection(true);
    stitcher.setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);
    stitcher.setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(try_use_gpu));
    stitcher.setBundleAdjuster(makePtr<detail::BundleAdjusterRay>());

#ifdef HAVE_OPENCV_CUDALEGACY
    if (try_use_gpu && cuda::getCudaEnabledDeviceCount() > 0)
    {
#ifdef HAVE_OPENCV_XFEATURES2D
        stitcher.setFeaturesFinder(makePtr<detail::SurfFeaturesFinderGpu>());
#else
        stitcher.setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
#endif
        stitcher.setWarper(makePtr<SphericalWarperGpu>());
        stitcher.setSeamFinder(makePtr<detail::GraphCutSeamFinderGpu>());
    }
    else
#endif
    {
#ifdef HAVE_OPENCV_XFEATURES2D
        stitcher.setFeaturesFinder(makePtr<detail::SurfFeaturesFinder>());
#else
        stitcher.setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
#endif
        stitcher.setWarper(makePtr<SphericalWarper>());
        stitcher.setSeamFinder(makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR));
    }

    stitcher.setExposureCompensator(makePtr<detail::BlocksGainCompensator>());
    stitcher.setBlender(makePtr<detail::MultiBandBlender>(try_use_gpu));

    stitcher.work_scale_ = 1;
    stitcher.seam_scale_ = 1;
    stitcher.seam_work_aspect_ = 1;
    stitcher.warped_image_scale_ = 1;

    return stitcher;
}


Ptr<Stitcher> Stitcher::create(Mode mode, bool try_use_gpu)
{
    Stitcher stit = createDefault(try_use_gpu);
    Ptr<Stitcher> stitcher = makePtr<Stitcher>(stit);

    switch (mode)
    {
    case PANORAMA: // PANORAMA is the default
        // already setup
    break;

    case SCANS:
        stitcher->setWaveCorrection(false);
        stitcher->setFeaturesMatcher(makePtr<detail::AffineBestOf2NearestMatcher>(false, try_use_gpu));
        stitcher->setBundleAdjuster(makePtr<detail::BundleAdjusterAffinePartial>());
        stitcher->setWarper(makePtr<AffineWarper>());
        stitcher->setExposureCompensator(makePtr<detail::NoExposureCompensator>());
    break;

    default:
        CV_Error(Error::StsBadArg, "Invalid stitching mode. Must be one of Stitcher::Mode");
    break;
    }

    return stitcher;
}


Stitcher::Status Stitcher::estimateTransform(InputArrayOfArrays images)
{
    CV_INSTRUMENT_REGION()

    return estimateTransform(images, std::vector<std::vector<Rect> >());
}


Stitcher::Status Stitcher::estimateTransform(InputArrayOfArrays images, const std::vector<std::vector<Rect> > &rois)
{
    CV_INSTRUMENT_REGION()

    images.getUMatVector(imgs_);
    rois_ = rois;

    Status status;

    if ((status = matchImages()) != OK)
        return status;

    if ((status = estimateCameraParams()) != OK)
        return status;

    return OK;
}



Stitcher::Status Stitcher::composePanorama(OutputArray pano)
{
    CV_INSTRUMENT_REGION()

    return composePanorama(std::vector<UMat>(), pano);
}


Stitcher::Status Stitcher::composePanorama(InputArrayOfArrays images, OutputArray pano)
{
    CV_INSTRUMENT_REGION()

    LOGLN("Warping images (auxiliary)... ");

    std::vector<UMat> imgs;
    images.getUMatVector(imgs);
    if (!imgs.empty())
    {
        CV_Assert(imgs.size() == imgs_.size());

        UMat img;
        seam_est_imgs_.resize(imgs.size());

        for (size_t i = 0; i < imgs.size(); ++i)
        {
            imgs_[i] = imgs[i];
            resize(imgs[i], img, Size(), seam_scale_, seam_scale_, INTER_LINEAR_EXACT);
            seam_est_imgs_[i] = img.clone();
        }

        std::vector<UMat> seam_est_imgs_subset;
        std::vector<UMat> imgs_subset;

        for (size_t i = 0; i < indices_.size(); ++i)
        {
            imgs_subset.push_back(imgs_[indices_[i]]);
            seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
        }

        seam_est_imgs_ = seam_est_imgs_subset;
        imgs_ = imgs_subset;
    }

    UMat pano_;

#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    std::vector<Point> corners(imgs_.size());
    std::vector<UMat> masks_warped(imgs_.size());
    std::vector<UMat> images_warped(imgs_.size());
    std::vector<Size> sizes(imgs_.size());
    std::vector<UMat> masks(imgs_.size());

    // Prepare image masks
    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        masks[i].create(seam_est_imgs_[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks
    Ptr<detail::RotationWarper> w = warper_->create(float(warped_image_scale_ * seam_work_aspect_));
    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        Mat_<float> K;
        cameras_[i].K().convertTo(K, CV_32F);
        K(0,0) *= (float)seam_work_aspect_;
        K(0,2) *= (float)seam_work_aspect_;
        K(1,1) *= (float)seam_work_aspect_;
        K(1,2) *= (float)seam_work_aspect_;

        corners[i] = w->warp(seam_est_imgs_[i], K, cameras_[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        w->warp(masks[i], K, cameras_[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }


    LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Compensate exposure before finding seams
    exposure_comp_->feed(corners, images_warped, masks_warped);
    for (size_t i = 0; i < imgs_.size(); ++i)
        exposure_comp_->apply(int(i), corners[i], images_warped[i], masks_warped[i]);

    // Find seams
    std::vector<UMat> images_warped_f(imgs_.size());
    for (size_t i = 0; i < imgs_.size(); ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    seam_finder_->find(images_warped_f, corners, masks_warped);

    // Release unused memory
    seam_est_imgs_.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    LOGLN("Compositing...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    UMat img_warped, img_warped_s;
    UMat dilated_mask, seam_mask, mask, mask_warped;

    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;
    bool is_blender_prepared = false;

    double compose_scale = 1;
    bool is_compose_scale_set = false;

    std::vector<detail::CameraParams> cameras_scaled(cameras_);

    UMat full_img, img;
    for (size_t img_idx = 0; img_idx < imgs_.size(); ++img_idx)
    {
        LOGLN("Compositing image #" << indices_[img_idx] + 1);
#if ENABLE_LOG
        int64 compositing_t = getTickCount();
#endif

        // Read image and resize it if necessary
        full_img = imgs_[img_idx];
        if (!is_compose_scale_set)
        {
            if (compose_resol_ > 0)
                compose_scale = std::min(1.0, std::sqrt(compose_resol_ * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale_;
            compose_work_aspect = compose_scale / work_scale_;

            // Update warped image scale
            float warp_scale = static_cast<float>(warped_image_scale_ * compose_work_aspect);
            w = warper_->create(warp_scale);

            // Update corners and sizes
            for (size_t i = 0; i < imgs_.size(); ++i)
            {
                // Update intrinsics
                cameras_scaled[i].ppx *= compose_work_aspect;
                cameras_scaled[i].ppy *= compose_work_aspect;
                cameras_scaled[i].focal *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes_[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes_[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes_[i].height * compose_scale);
                }

                Mat K;
                cameras_scaled[i].K().convertTo(K, CV_32F);
                Rect roi = w->warpRoi(sz, K, cameras_scaled[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (std::abs(compose_scale - 1) > 1e-1)
        {
#if ENABLE_LOG
            int64 resize_t = getTickCount();
#endif
            resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
            LOGLN("  resize time: " << ((getTickCount() - resize_t) / getTickFrequency()) << " sec");
        }
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        LOGLN(" after resize time: " << ((getTickCount() - compositing_t) / getTickFrequency()) << " sec");

        Mat K;
        cameras_scaled[img_idx].K().convertTo(K, CV_32F);

#if ENABLE_LOG
        int64 pt = getTickCount();
#endif
        // Warp the current image
        w->warp(img, K, cameras_[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
        LOGLN(" warp the current image: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
        pt = getTickCount();
#endif

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        w->warp(mask, K, cameras_[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
        LOGLN(" warp the current image mask: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
        pt = getTickCount();
#endif

        // Compensate exposure
        exposure_comp_->apply((int)img_idx, corners[img_idx], img_warped, mask_warped);
        LOGLN(" compensate exposure: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
        pt = getTickCount();
#endif

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        // Make sure seam mask has proper size
        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);

        bitwise_and(seam_mask, mask_warped, mask_warped);

        LOGLN(" other: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
        pt = getTickCount();
#endif

        if (!is_blender_prepared)
        {
            blender_->prepare(corners, sizes);
            is_blender_prepared = true;
        }

        LOGLN(" other2: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");

        LOGLN(" feed...");
#if ENABLE_LOG
        int64 feed_t = getTickCount();
#endif
        // Blend the current image
        blender_->feed(img_warped_s, mask_warped, corners[img_idx]);
        LOGLN(" feed time: " << ((getTickCount() - feed_t) / getTickFrequency()) << " sec");
        LOGLN("Compositing ## time: " << ((getTickCount() - compositing_t) / getTickFrequency()) << " sec");
    }

#if ENABLE_LOG
        int64 blend_t = getTickCount();
#endif
    UMat result, result_mask;
    blender_->blend(result, result_mask);
    LOGLN("blend time: " << ((getTickCount() - blend_t) / getTickFrequency()) << " sec");

    LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Preliminary result is in CV_16SC3 format, but all values are in [0,255] range,
    // so convert it to avoid user confusing
    result.convertTo(pano, CV_8U);

    return OK;
}


Stitcher::Status Stitcher::stitch(InputArrayOfArrays images, OutputArray pano)
{
    CV_INSTRUMENT_REGION()

    Status status = estimateTransform(images);
    if (status != OK)
        return status;
    return composePanorama(pano);
}


Stitcher::Status Stitcher::stitch(InputArrayOfArrays images, const std::vector<std::vector<Rect> > &rois, OutputArray pano)
{
    CV_INSTRUMENT_REGION()

    Status status = estimateTransform(images, rois);
    if (status != OK)
        return status;
    return composePanorama(pano);
}


Stitcher::Status Stitcher::matchImages()
{
    if ((int)imgs_.size() < 2)
    {
        LOGLN("Need more images");
        return ERR_NEED_MORE_IMGS;
    }

    work_scale_ = 1;
    seam_work_aspect_ = 1;
    seam_scale_ = 1;
    bool is_work_scale_set = false;
    bool is_seam_scale_set = false;
    UMat full_img, img;
    features_.resize(imgs_.size());
    seam_est_imgs_.resize(imgs_.size());
    full_img_sizes_.resize(imgs_.size());

    LOGLN("Finding features...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    std::vector<UMat> feature_find_imgs(imgs_.size());
    std::vector<std::vector<Rect> > feature_find_rois(rois_.size());

    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        full_img = imgs_[i];
        full_img_sizes_[i] = full_img.size();

        if (registr_resol_ < 0)
        {
            img = full_img;
            work_scale_ = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale_ = std::min(1.0, std::sqrt(registr_resol_ * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale_, work_scale_, INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set)
        {
            seam_scale_ = std::min(1.0, std::sqrt(seam_est_resol_ * 1e6 / full_img.size().area()));
            seam_work_aspect_ = seam_scale_ / work_scale_;
            is_seam_scale_set = true;
        }

        if (rois_.empty())
            feature_find_imgs[i] = img;
        else
        {
            feature_find_rois[i].resize(rois_[i].size());
            for (size_t j = 0; j < rois_[i].size(); ++j)
            {
                Point tl(cvRound(rois_[i][j].x * work_scale_), cvRound(rois_[i][j].y * work_scale_));
                Point br(cvRound(rois_[i][j].br().x * work_scale_), cvRound(rois_[i][j].br().y * work_scale_));
                feature_find_rois[i][j] = Rect(tl, br);
            }
            feature_find_imgs[i] = img;
        }
        features_[i].img_idx = (int)i;
        LOGLN("Features in image #" << i+1 << ": " << features_[i].keypoints.size());

        resize(full_img, img, Size(), seam_scale_, seam_scale_, INTER_LINEAR_EXACT);
        seam_est_imgs_[i] = img.clone();
    }

    // find features possibly in parallel
    if (rois_.empty())
        (*features_finder_)(feature_find_imgs, features_);
    else
        (*features_finder_)(feature_find_imgs, features_, feature_find_rois);

    // Do it to save memory
    features_finder_->collectGarbage();
    full_img.release();
    img.release();
    feature_find_imgs.clear();
    feature_find_rois.clear();

    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOG("Pairwise matching");
#if ENABLE_LOG
    t = getTickCount();
#endif
    (*features_matcher_)(features_, pairwise_matches_, matching_mask_);
    features_matcher_->collectGarbage();
    LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Leave only images we are sure are from the same panorama
    indices_ = detail::leaveBiggestComponent(features_, pairwise_matches_, (float)conf_thresh_);
    std::vector<UMat> seam_est_imgs_subset;
    std::vector<UMat> imgs_subset;
    std::vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices_.size(); ++i)
    {
        imgs_subset.push_back(imgs_[indices_[i]]);
        seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
        full_img_sizes_subset.push_back(full_img_sizes_[indices_[i]]);
    }
    seam_est_imgs_ = seam_est_imgs_subset;
    imgs_ = imgs_subset;
    full_img_sizes_ = full_img_sizes_subset;

    if ((int)imgs_.size() < 2)
    {
        LOGLN("Need more images");
        return ERR_NEED_MORE_IMGS;
    }

    return OK;
}


Stitcher::Status Stitcher::estimateCameraParams()
{
    /* TODO OpenCV ABI 4.x
    get rid of this dynamic_cast hack and use estimator_
    */
    Ptr<detail::Estimator> estimator;
    if (dynamic_cast<detail::AffineBestOf2NearestMatcher*>(features_matcher_.get()))
        estimator = makePtr<detail::AffineBasedEstimator>();
    else
        estimator = makePtr<detail::HomographyBasedEstimator>();

    if (!(*estimator)(features_, pairwise_matches_, cameras_))
        return ERR_HOMOGRAPHY_EST_FAIL;

    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        Mat R;
        cameras_[i].R.convertTo(R, CV_32F);
        cameras_[i].R = R;
        //LOGLN("Initial intrinsic parameters #" << indices_[i] + 1 << ":\n " << cameras_[i].K());
    }

    bundle_adjuster_->setConfThresh(conf_thresh_);
    if (!(*bundle_adjuster_)(features_, pairwise_matches_, cameras_))
        return ERR_CAMERA_PARAMS_ADJUST_FAIL;

    // Find median focal length and use it as final image scale
    std::vector<double> focals;
    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        //LOGLN("Camera #" << indices_[i] + 1 << ":\n" << cameras_[i].K());
        focals.push_back(cameras_[i].focal);
    }

    std::sort(focals.begin(), focals.end());
    if (focals.size() % 2 == 1)
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    if (do_wave_correct_)
    {
        std::vector<Mat> rmats;
        for (size_t i = 0; i < cameras_.size(); ++i)
            rmats.push_back(cameras_[i].R.clone());
        detail::waveCorrect(rmats, wave_correct_kind_);
        for (size_t i = 0; i < cameras_.size(); ++i)
            cameras_[i].R = rmats[i];
    }

    return OK;
}


Ptr<Stitcher> createStitcher(bool try_use_gpu)
{
    CV_INSTRUMENT_REGION()

    Ptr<Stitcher> stitcher = makePtr<Stitcher>();
    stitcher->setRegistrationResol(0.6);
    stitcher->setSeamEstimationResol(0.1);
    stitcher->setCompositingResol(Stitcher::ORIG_RESOL);
    stitcher->setPanoConfidenceThresh(1);
    stitcher->setWaveCorrection(true);
    stitcher->setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);
    stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(try_use_gpu));
    stitcher->setBundleAdjuster(makePtr<detail::BundleAdjusterRay>());

    #ifdef HAVE_OPENCV_CUDALEGACY
    if (try_use_gpu && cuda::getCudaEnabledDeviceCount() > 0)
    {
        #ifdef HAVE_OPENCV_NONFREE
        stitcher->setFeaturesFinder(makePtr<detail::SurfFeaturesFinderGpu>());
        #else
        stitcher->setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
        #endif
        stitcher->setWarper(makePtr<SphericalWarperGpu>());
        stitcher->setSeamFinder(makePtr<detail::GraphCutSeamFinderGpu>());
    }
    else
    #endif
    {
        #ifdef HAVE_OPENCV_NONFREE
        stitcher->setFeaturesFinder(makePtr<detail::SurfFeaturesFinder>());
        #else
        stitcher->setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
        #endif
        stitcher->setWarper(makePtr<SphericalWarper>());
        stitcher->setSeamFinder(makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR));
    }

    stitcher->setExposureCompensator(makePtr<detail::BlocksGainCompensator>());
    stitcher->setBlender(makePtr<detail::MultiBandBlender>(try_use_gpu));

    return stitcher;
}
} // namespace cv
