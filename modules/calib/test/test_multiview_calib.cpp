/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

#include "test_precomp.hpp"
#include <opencv2/ts/cuda_test.hpp> // EXPECT_MAT_NEAR

namespace opencv_test { namespace {

TEST(multiview_calibration, accuracy) {
    // convert euler angles to rotation matrix
    const auto euler2rot = [] (double x, double y, double z) {
        cv::Matx33d R_x(1, 0, 0, 0, cos(x), -sin(x), 0, sin(x), cos(x));
        cv::Matx33d R_y(cos(y), 0, sin(y), 0, 1, 0, -sin(y), 0, cos(y));
        cv::Matx33d R_z(cos(z), -sin(z), 0, sin(z), cos(z), 0, 0, 0, 1);
        return cv::Mat(R_z * R_y * R_x);
    };
    const cv::Size board_size (5,4);
    cv::RNG &rng = cv::theRNG();
    const double board_len = 0.08, noise_std = 0.04;
    const int num_cameras = 4, num_pts = board_size.area();
    std::vector<cv::Vec3f> board_pattern (num_pts);
    // fill pattern points
    for (int j = 0; j < board_size.height; j++) {
        for (int i = 0; i < board_size.width; i++) {
            board_pattern[j*board_size.width+i] = cv::Vec3f ((float)i, (float)j, 0)*board_len;
        }
    }
    std::vector<bool> is_fisheye(num_cameras, false);
    std::vector<cv::Size> image_sizes(num_cameras);
    std::vector<cv::Mat> Ks_gt, distortions_gt, Rs_gt, Ts_gt;
    for (int c = 0; c < num_cameras; c++) {
        // generate intrinsics and extrinsics
        image_sizes[c] = cv::Size(rng.uniform(1300, 1500), rng.uniform(900, 1300));
        const double focal = rng.uniform(900.0, 1300.0);
        cv::Matx33d K(focal, 0, (double)image_sizes[c].width/2.,
                    0, focal, (double)image_sizes[c].height/2.,
                    0, 0, 1);
        cv::Matx<double, 1, 5> dist (rng.uniform(3e-1, 8e-1), 0, 0, 0, 0);
        Ks_gt.emplace_back(cv::Mat(K));
        distortions_gt.emplace_back(cv::Mat(dist));
        if (c == 0) {
            // I | 0
            Rs_gt.emplace_back(cv::Mat(cv::Matx33d::eye()));
            Ts_gt.emplace_back(cv::Mat(cv::Vec3d::zeros()));
        } else {
            const double ty_min = -.3, ty_max = .3, tx_min = -.3, tx_max = .3, tz_min = -.1, tz_max = .1;
            const double yaw_min = -20, yaw_max = 20, pitch_min = -20, pitch_max = 20, roll_min = -20, roll_max = 20;
            Rs_gt.emplace_back(euler2rot(rng.uniform(yaw_min, yaw_max)*M_PI/180,
                                         rng.uniform(pitch_min, pitch_max)*M_PI/180,
                                         rng.uniform(roll_min, roll_max)*M_PI/180));
            Ts_gt.emplace_back(cv::Mat(cv::Vec3d(rng.uniform(tx_min, tx_max),
                                                 rng.uniform(ty_min, ty_max),
                                                 rng.uniform(tz_min, tz_max))));
        }
    }

    const int MAX_SAMPLES = 2000, MAX_FRAMES = 50;
    cv::Mat pattern (board_pattern, true/*copy*/);
    pattern = pattern.reshape(1).t();
    pattern.row(2) = 2.0; // set approximate depth of object points
    const double ty_min = -2, ty_max = 2, tx_min = -2, tx_max = 2, tz_min = -1, tz_max = 1;
    const double yaw_min = -45, yaw_max = 45, pitch_min = -45, pitch_max = 45, roll_min = -45, roll_max = 45;
    std::vector<std::vector<cv::Vec3f>> objPoints;
    std::vector<std::vector<cv::Mat>> image_points_all(num_cameras);
    cv::Mat ones = cv::Mat_<float>::ones(1, num_pts);
    std::vector<std::vector<bool>> visibility;
    cv::Mat centroid = cv::Mat(cv::Matx31f(
            (float)cv::mean(pattern.row(0)).val[0],
            (float)cv::mean(pattern.row(1)).val[0],
            (float)cv::mean(pattern.row(2)).val[0]));
    for (int f = 0; f < MAX_SAMPLES; f++) {
        cv::Mat R = euler2rot(rng.uniform(yaw_min, yaw_max)*M_PI/180,
                              rng.uniform(pitch_min, pitch_max)*M_PI/180,
                              rng.uniform(roll_min, roll_max)*M_PI/180);
        cv::Mat t = cv::Mat(cv::Matx31f(
                (float)rng.uniform(tx_min, tx_max),
                (float)rng.uniform(ty_min, ty_max),
                (float)rng.uniform(tz_min, tz_max)));

        R.convertTo(R, CV_32F);
        cv::Mat pattern_new = (R * (pattern - centroid * ones) + centroid * ones  + t * ones).t();

        std::vector<cv::Mat> img_pts_cams(num_cameras);
        std::vector<bool> visible(num_cameras, false);
        int num_visible_patterns = 0;
        for (int c = 0; c < num_cameras; c++) {
            cv::Mat img_pts;
            if (is_fisheye[c]) {
                cv::fisheye::projectPoints(pattern_new, img_pts, Rs_gt[c], Ts_gt[c], Ks_gt[c], distortions_gt[c]);
            } else {
                cv::projectPoints(pattern_new, Rs_gt[c], Ts_gt[c], Ks_gt[c], distortions_gt[c], img_pts);
            }

            // add normal / Gaussian noise to image points
            cv::Mat noise (img_pts.rows, img_pts.cols, img_pts.type());
            rng.fill(noise, cv::RNG::NORMAL, 0, noise_std);
            img_pts += noise;

            bool are_all_pts_in_image = true;
            const auto * const pts = (float *) img_pts.data;
            for (int i = 0; i < num_pts; i++) {
                if (pts[i*2  ] < 0 || pts[i*2  ] > (float)image_sizes[c].width ||
                    pts[i*2+1] < 0 || pts[i*2+1] > (float)image_sizes[c].height) {
                    are_all_pts_in_image = false;
                    break;
                }
            }
            if (are_all_pts_in_image) {
                visible[c] = true;
                num_visible_patterns += 1;
                img_pts.copyTo(img_pts_cams[c]);
            }
        }

        if (num_visible_patterns >= 2) {
            objPoints.emplace_back(board_pattern);
            visibility.emplace_back(visible);
            for (int c = 0; c < num_cameras; c++) {
                image_points_all[c].emplace_back(img_pts_cams[c].clone());
            }
            if (objPoints.size() >= MAX_FRAMES)
                break;
        }
    }
    cv::Mat visibility_mat = cv::Mat_<bool>(num_cameras, (int)objPoints.size());
    for (int c = 0; c < num_cameras; c++) {
        for (int f = 0; f < (int)objPoints.size(); f++) {
            visibility_mat.at<bool>(c, f) = visibility[f][c];
        }
    }

    std::vector<cv::Mat> Ks, distortions, Rs, Ts;
    cv::Mat errors_mat, output_pairs, rvecs0, tvecs0;
    EXPECT_TRUE(calibrateMultiview (objPoints, image_points_all, image_sizes, visibility_mat,
       Rs, Ts, Ks, distortions, rvecs0, tvecs0, is_fisheye, errors_mat, output_pairs));

    const double K_err_tol = 1e1, dist_tol = 2e-1, R_tol = 1e-2, T_tol = 1e-2;
    for (int c = 0; c < num_cameras; c++) {
        cv::Mat R;
        cv::Rodrigues(Rs[c], R);
        EXPECT_MAT_NEAR(Ks_gt[c], Ks[c], K_err_tol);
        EXPECT_MAT_NEAR(distortions_gt[c], distortions[c], dist_tol);
        EXPECT_MAT_NEAR(Rs_gt[c], R, R_tol);
        EXPECT_MAT_NEAR(Ts_gt[c], Ts[c], T_tol);
    }
}
}}