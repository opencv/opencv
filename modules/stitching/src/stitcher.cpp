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

namespace cv {

Stitcher Stitcher::createDefault(bool try_use_gpu)
{
    Stitcher stitcher;
    stitcher.setRegistrationResol(0.6);
    stitcher.setSeamEstimationResol(0.1);
    stitcher.setCompositingResol(ORIG_RESOL);
    stitcher.setPanoConfidenceThresh(1);
    stitcher.setHorizontalStrightening(true);
    stitcher.setFeaturesMatcher(new detail::BestOf2NearestMatcher(try_use_gpu));

#ifndef ANDROID
    if (try_use_gpu && gpu::getCudaEnabledDeviceCount() > 0)
    {
        stitcher.setFeaturesFinder(new detail::SurfFeaturesFinderGpu());
        stitcher.setWarper(new SphericalWarperGpu());
    }
    else
#endif
    {
        stitcher.setFeaturesFinder(new detail::SurfFeaturesFinder());
        stitcher.setWarper(new SphericalWarper());
    }

    stitcher.setExposureCompenstor(new detail::BlocksGainCompensator());
    stitcher.setSeamFinder(new detail::GraphCutSeamFinder());
    stitcher.setBlender(new detail::MultiBandBlender(try_use_gpu));

    return stitcher;
}


Stitcher::Status Stitcher::stitch(InputArray imgs_, OutputArray pano_)
{
    // TODO split this func

    vector<Mat> imgs;
    imgs_.getMatVector(imgs);
    Mat &pano = pano_.getMatRef();

    int64 app_start_time = getTickCount();

    int num_imgs = static_cast<int>(imgs.size());
    if (num_imgs < 2)
    {
        LOGLN("Need more images");
        return ERR_NEED_MORE_IMGS;
    }

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    LOGLN("Finding features...");
    int64 t = getTickCount();

    vector<detail::ImageFeatures> features(num_imgs);
    Mat full_img, img;
    vector<Mat> seam_est_imgs(num_imgs);
    vector<Size> full_img_sizes(num_imgs);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_imgs; ++i)
    {
        full_img = imgs[i];
        full_img_sizes[i] = full_img.size();

        if (registr_resol_ < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(registr_resol_ * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_est_resol_ * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        (*features_finder_)(img, features[i]);
        features[i].img_idx = i;
        LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());

        resize(full_img, img, Size(), seam_scale, seam_scale);
        seam_est_imgs[i] = img.clone();
    }

    features_finder_->collectGarbage();
    full_img.release();
    img.release();

    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOG("Pairwise matching");
    t = getTickCount();
    vector<detail::MatchesInfo> pairwise_matches;
    (*features_matcher_)(features, pairwise_matches);
    features_matcher_->collectGarbage();
    LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Leave only images we are sure are from the same panorama
    vector<int> indices = detail::leaveBiggestComponent(features, pairwise_matches, conf_thresh_);
    vector<Mat> seam_est_imgs_subset;
    vector<Mat> imgs_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        imgs_subset.push_back(imgs[indices[i]]);
        seam_est_imgs_subset.push_back(seam_est_imgs[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    seam_est_imgs = seam_est_imgs_subset;
    imgs = imgs_subset;
    full_img_sizes = full_img_sizes_subset;

    num_imgs = static_cast<int>(imgs.size());
    if (num_imgs < 2)
    {
        LOGLN("Need more images");
        return ERR_NEED_MORE_IMGS;
    }

    vector<detail::CameraParams> cameras;
    detail::HomographyBasedEstimator estimator;
    estimator(features, pairwise_matches, cameras);

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        LOGLN("Initial intrinsic parameters #" << indices[i]+1 << ":\n " << cameras[i].K());
    }

    detail::BundleAdjuster adjuster(detail::BundleAdjuster::FOCAL_RAY_SPACE, conf_thresh_);
    adjuster(features, pairwise_matches, cameras);

    // Find median focal length
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        LOGLN("Camera #" << indices[i]+1 << ":\n" << cameras[i].K());
        focals.push_back(cameras[i].focal);
    }
    nth_element(focals.begin(), focals.begin() + focals.size()/2, focals.end());
    float warped_image_scale = static_cast<float>(focals[focals.size() / 2]);

    if (horiz_stright_)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R);
        detail::waveCorrect(rmats);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    LOGLN("Warping images (auxiliary)... ");
    t = getTickCount();

    vector<Point> corners(num_imgs);
    vector<Mat> masks_warped(num_imgs);
    vector<Mat> images_warped(num_imgs);
    vector<Size> sizes(num_imgs);
    vector<Mat> masks(num_imgs);

    // Preapre images masks
    for (int i = 0; i < num_imgs; ++i)
    {
        masks[i].create(seam_est_imgs[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks
    Ptr<detail::Warper> warper = warper_->create(warped_image_scale * seam_work_aspect);
    for (int i = 0; i < num_imgs; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        K(0,0) *= seam_work_aspect; K(0,2) *= seam_work_aspect;
        K(1,1) *= seam_work_aspect; K(1,2) *= seam_work_aspect;

        corners[i] = warper->warp(seam_est_imgs[i], K, cameras[i].R, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, masks_warped[i], INTER_NEAREST, BORDER_CONSTANT);
    }

    vector<Mat> images_warped_f(num_imgs);
    for (int i = 0; i < num_imgs; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    exposure_comp_->feed(corners, images_warped, masks_warped);
    seam_finder_->find(images_warped_f, corners, masks_warped);

    // Release unused memory
    seam_est_imgs.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    LOGLN("Compositing...");
    t = getTickCount();

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    double compose_seam_aspect = 1;
    double compose_work_aspect = 1;
    bool is_blender_prepared = false;

    for (int img_idx = 0; img_idx < num_imgs; ++img_idx)
    {
        LOGLN("Compositing image #" << indices[img_idx]+1);

        // Read image and resize it if necessary
        full_img = imgs[img_idx];
        if (!is_compose_scale_set)
        {
            if (compose_resol_ > 0)
                compose_scale = min(1.0, sqrt(compose_resol_ * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < num_imgs; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (std::abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, mask_warped, INTER_NEAREST, BORDER_CONSTANT);

        // Compensate exposure
        exposure_comp_->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;

        if (!is_blender_prepared)
        {
            blender_->prepare(corners, sizes);
            is_blender_prepared = true;
        }

        // Blend the current image
        blender_->feed(img_warped_s, mask_warped, corners[img_idx]);
    }

    Mat result, result_mask;
    blender_->blend(result, result_mask);

    LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Preliminary result is in CV_16SC3 format, but all values are in [0,255] range,
    // so convert it to avoid user confusing
    result.convertTo(pano, CV_8U);

    LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");

    return OK;
}

} // namespace cv
