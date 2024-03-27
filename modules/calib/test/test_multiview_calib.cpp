// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/core/utils/logger.hpp>
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
    const cv::Size board_size (9, 7);
    cv::RNG rng(0);
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
    std::vector<cv::Mat> Ks_gt, distortions_gt, Rs_gt, Ts_gt, rvecs_gt;
    for (int c = 0; c < num_cameras; c++) {
        // Each camera has 0.5 probability to be a fisheye camera
        double p = rng.uniform(0., 1.);
        if (p > 0.5) {
            is_fisheye[c] = true;
        }

        // generate intrinsics and extrinsics
        image_sizes[c] = cv::Size(rng.uniform(1300, 1500), rng.uniform(900, 1300));
        const double focal = rng.uniform(900.0, 1300.0);
        cv::Matx33d K(focal, 0, (double)image_sizes[c].width/2.,
                    0, focal, (double)image_sizes[c].height/2.,
                    0, 0, 1);
        Ks_gt.emplace_back(cv::Mat(K));
        // set the distortion to be of length 5 if it is a pnihole camera, otherwise, set to be 4
        if (!is_fisheye[c]) {
            cv::Matx<double, 1, 5> dist (rng.uniform(1e-1, 3e-1), rng.uniform(1e-2, 5e-2), rng.uniform(1e-2, 5e-2), rng.uniform(1e-2, 5e-2), rng.uniform(1e-2, 5e-2));
            distortions_gt.emplace_back(cv::Mat(dist));
        } else {
            cv::Matx<double, 1, 4> dist (rng.uniform(0., 1e-1), rng.uniform(0., 5e-3), rng.uniform(0., 5e-4), rng.uniform(0., 5e-5));
            distortions_gt.emplace_back(cv::Mat(dist));
        }

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
        cv::Mat rvec;
        cv::Rodrigues(Rs_gt[c], rvec);
        rvecs_gt.emplace_back(rvec);
    }

    const int MAX_SAMPLES = 2000, MAX_FRAMES = 100;
    cv::Mat pattern (board_pattern, true/*copy*/);
    pattern = pattern.reshape(1, num_pts).t();
    pattern.row(2) = 2.0; // set approximate depth of object points
    const double ty_min = -2, ty_max = 2, tx_min = -2, tx_max = 2, tz_min = -1, tz_max = 1;
    const double yaw_min = -45, yaw_max = 45, pitch_min = -45, pitch_max = 45, roll_min = -45, roll_max = 45;
    std::vector<std::vector<cv::Vec3f>> objPoints;
    std::vector<std::vector<cv::Mat>> image_points_all(num_cameras);
    cv::Mat ones = cv::Mat_<float>::ones(1, num_pts);
    std::vector<std::vector<uchar>> visibility;
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
        pattern_new = pattern_new.reshape(3);

        std::vector<cv::Mat> img_pts_cams(num_cameras);
        std::vector<uchar> visible(num_cameras, (uchar)0);
        int num_visible_patterns = 0;
        for (int c = 0; c < num_cameras; c++) {
            cv::Mat img_pts;
            if (is_fisheye[c]) {
                cv::fisheye::projectPoints(pattern_new, img_pts, rvecs_gt[c], Ts_gt[c], Ks_gt[c], distortions_gt[c]);
            } else {
                cv::projectPoints(pattern_new, rvecs_gt[c], Ts_gt[c], Ks_gt[c], distortions_gt[c], img_pts);
            }

            // add normal / Gaussian noise to image points
            cv::Mat noise (img_pts.rows, img_pts.cols, img_pts.type());
            rng.fill(noise, cv::RNG::NORMAL, 0, noise_std);
            img_pts += noise;

            auto * const pts = (float *) img_pts.data;
            int num_valid_pts = num_pts;
            for (int i = 0; i < num_pts; i++) {
                if (pts[i*2  ] < 0 || pts[i*2  ] > (float)image_sizes[c].width ||
                    pts[i*2+1] < 0 || pts[i*2+1] > (float)image_sizes[c].height ||
                    pattern_new.at<Point3f>(i).z < 1e-2) {
                    num_valid_pts--;
                    pts[i*2  ] = pts[i*2+1] = -1;
                }
            }
            if (num_valid_pts > 3) { // requires at least 4 pts per image
                visible[c] = 1;
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
    cv::Mat visibility_mat = cv::Mat_<uchar>(num_cameras, (int)objPoints.size());
    for (int c = 0; c < num_cameras; c++) {
        for (int f = 0; f < (int)objPoints.size(); f++) {
            visibility_mat.at<uchar>(c, f) = visibility[f][c];
        }
    }

    std::vector<cv::Mat> Ks, distortions, Rs, Ts;
    cv::Mat errors_mat, output_pairs, rvecs0, tvecs0;
    calibrateMultiview (objPoints, image_points_all, image_sizes, visibility_mat,
       Rs, Ts, Ks, distortions, rvecs0, tvecs0, is_fisheye, errors_mat, output_pairs, false);

    const double angle_tol = 1.*M_PI/180., pos_tol = 5e-2, dist_tol = 5.;
    const int rows = 10; // Number of rows in the grid
    const int cols = 10; // Number of columns in the grid
    const int total_num = rows * cols;
    cv::Mat xGrid, yGrid;

    std::vector<double> t_x, t_y;
    for (int i = 0; i < cols; i++) t_x.push_back(double(i));
    for (int i = 0; i < rows; i++) t_y.push_back(double(i));

    cv::repeat(cv::Mat(t_x).reshape(1,1), rows, 1, xGrid);
    cv::repeat(cv::Mat(t_y).reshape(1,1).t(), 1, cols, yGrid);

    cv::Mat xGridSingleCol = xGrid.reshape(1, total_num); // Reshape to 1 row and multiple columns
    cv::Mat yGridSingleCol = yGrid.reshape(1, total_num); // Reshape to 1 row and multiple columns


    cv::Mat image_pts;
    image_pts.create(total_num, 2, CV_64F);

    cv::Mat pt_norm;
    pt_norm.create(total_num, 3, CV_64F);
    cv::Mat col = pt_norm.col(2);
    col.setTo(cv::Scalar(1.));

    std::vector<double> err_dist(total_num);
    for (int c = 0; c < num_cameras; c++) {
        CV_LOG_INFO(NULL, "true  distortions: " << distortions_gt[c]);
        CV_LOG_INFO(NULL, "found distortions: " << distortions[c]);

        // compare the calculated R, T
        double cos_r = (cv::trace(Rs_gt[c].t() * Rs[c])[0] - 1) / 2.;
        double angle = std::acos(std::max(std::min(cos_r, 1.), -1.));
        cv::Mat dist_mat;
        subtract(Rs_gt[c].t() * Ts_gt[c], Rs[c].t() * Ts[c], dist_mat);
        double dist = cv::norm(dist_mat);
        CV_LOG_INFO(NULL, "rotation error: " << angle);
        CV_LOG_INFO(NULL, "position error: " << dist);
        ASSERT_NEAR(angle, 0., angle_tol);
        ASSERT_NEAR(dist, 0., pos_tol);

        // create the respective grids
        cv::Mat img_pts_col0 = image_pts.col(0);
        cv::Mat img_pts_col1 = image_pts.col(1);
        img_pts_col0 = xGridSingleCol * double(image_sizes[c].width) / double(cols);
        img_pts_col1 = yGridSingleCol * double(image_sizes[c].height) / double(rows);

        image_pts = image_pts.reshape(2);

        // undistort with the estimated intrinsics
        cv::Mat undist;
        if (is_fisheye[c]) {
            cv::fisheye::undistortPoints(image_pts, undist, Ks[c], distortions[c]);
        } else {
            cv::undistortPoints(image_pts, undist, Ks[c], distortions[c]);
        }
        undist = undist.reshape(1, total_num);
        cv::Mat pt_norm_col = pt_norm.colRange(0, 2);
        undist.copyTo(pt_norm_col);

        pt_norm = pt_norm.reshape(3);

        // ndistort with the ground truth intrinsics
        cv::Mat pt_distorted;
        if (is_fisheye[c]) {
            cv::fisheye::projectPoints(pt_norm, pt_distorted, cv::Mat(cv::Vec3d::zeros()), cv::Mat(cv::Vec3d::zeros()), Ks_gt[c], distortions_gt[c]);
        } else {
            cv::projectPoints(pt_norm, cv::Mat(cv::Vec3d::zeros()), cv::Mat(cv::Vec3d::zeros()), Ks_gt[c], distortions_gt[c], pt_distorted);
        }

        subtract(pt_distorted, image_pts, pt_distorted);

        cv::Mat diff;
        pt_distorted.convertTo(diff, CV_64FC2);

        // compute the difference as error
        for (int i = 0; i < total_num; i++)
            err_dist[i] = cv::norm(diff.at<Point2d>(i));
        size_t n = total_num / 2;
        std::nth_element(err_dist.begin(), err_dist.begin() + n, err_dist.end());

        ASSERT_NEAR(err_dist[n], 0., dist_tol);
        CV_LOG_INFO(NULL, "median distortion error: " << err_dist[n]);

        // reshape it back
        image_pts = image_pts.reshape(1);
        pt_norm = pt_norm.reshape(1);
    }
}
}}
