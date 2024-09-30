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
    const cv::Size board_size (5,4);
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
    std::vector<cv::Mat> Ks_gt, distortions_gt, Rs_gt, Ts_gt;
    for (int c = 0; c < num_cameras; c++) {
        // generate intrinsics and extrinsics
        image_sizes[c] = cv::Size(rng.uniform(1300, 1500), rng.uniform(900, 1300));
        const double focal = rng.uniform(900.0, 1300.0);
        cv::Matx33d K(focal, 0, (double)image_sizes[c].width/2.,
                    0, focal, (double)image_sizes[c].height/2.,
                    0, 0, 1);
        cv::Matx<double, 1, 5> dist (rng.uniform(1e-1, 3e-1), rng.uniform(1e-2, 5e-2), rng.uniform(1e-2, 5e-2), rng.uniform(1e-2, 5e-2), rng.uniform(1e-2, 5e-2));
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

        std::vector<cv::Mat> img_pts_cams(num_cameras);
        std::vector<uchar> visible(num_cameras, (uchar)0);
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

    const double K_err_tol = 1e1, dist_tol = 5e-2, R_tol = 1e-2, T_tol = 1e-2;
    for (int c = 0; c < num_cameras; c++) {
        cv::Mat R;
        cv::Rodrigues(Rs[c], R);
        EXPECT_MAT_NEAR(Ks_gt[c], Ks[c], K_err_tol);
        CV_LOG_INFO(NULL, "true  distortions: " << distortions_gt[c]);
        CV_LOG_INFO(NULL, "found distortions: " << distortions[c]);
        EXPECT_MAT_NEAR(distortions_gt[c], distortions[c], dist_tol);
        EXPECT_MAT_NEAR(Rs_gt[c], R, R_tol);
        EXPECT_MAT_NEAR(Ts_gt[c], Ts[c], T_tol);
    }
}

struct MultiViewTest : public ::testing::Test
{
    std::vector<cv::Vec3f> genAsymmetricObjectPoints(cv::Size board_size = cv::Size(8, 11), float square_size = 0.04)
    {
        std::vector<cv::Vec3f> objectPoints;
        objectPoints.reserve(board_size.height*board_size.width);
        for( int i = 0; i < board_size.height; i++ )
        {
            for( int j = 0; j < board_size.width; j++ )
            {
                objectPoints.push_back(cv::Point3f((2*j + i % 2)*square_size, i*square_size, 0));
            }
        }

        return objectPoints;
    }

    void loadImagePoints(const std::string& base_dir, const std::vector<std::string> cameras, int frameCount,
                         std::vector<std::vector<cv::Mat>>& image_points_all, cv::Mat& visibility)
    {
        image_points_all.clear();
        visibility.create(static_cast<int>(cameras.size()), frameCount, CV_8UC1);
        for (int c = 0; c < static_cast<int>(cameras.size()); c++)
        {
            std::vector<cv::Mat> camera_image_points;
            std::string fname = base_dir + cameras[c] + ".json";
            FileStorage fs(fname, cv::FileStorage::READ);
            ASSERT_TRUE(fs.isOpened()) << "Cannot open points file " << fname;
            for (int i = 0; i < frameCount; i++)
            {
                std::string nodeName = cv::format("frame_%d", i);
                FileNode node = fs[nodeName];
                if (!node.empty())
                {
                    camera_image_points.push_back(node.mat().reshape(2, 1));
                    visibility.at<uchar>(c, i) = 255;
                }
                else
                {
                    camera_image_points.push_back(cv::Mat());
                    visibility.at<uchar>(c, i) = 0;
                }
            }
            fs.release();
            image_points_all.push_back(camera_image_points);
        }
    }

    void validateCameraPose(const Mat& R, Mat T, const Mat& R_gt, const Mat& T_gt,
                            double angle_tol = 1.*M_PI/180., double pos_tol = 0.01)
    {
        double cos_r = (cv::trace(R_gt.t() * R)[0] - 1) / 2.;
        double angle = std::acos(std::max(std::min(cos_r, 1.), -1.));
        cv::Mat dist_mat;
        subtract(R_gt.t() * T_gt, R.t() * T, dist_mat);
        double dist = cv::norm(dist_mat);
        CV_LOG_INFO(NULL, "rotation error: " << angle);
        CV_LOG_INFO(NULL, "position error: " << dist);
        EXPECT_NEAR(angle, 0., angle_tol);
        EXPECT_NEAR(dist, 0., pos_tol);
    }

    void validateAllPoses(const std::vector<cv::Mat>& Rs_gt,
                          const std::vector<cv::Mat>& Ts_gt,
                          const std::vector<cv::Mat>& Rs,
                          const std::vector<cv::Mat>& Ts,
                          double angle_tol = 1.*M_PI/180.,
                          double pos_tol = 0.01)
    {
        ASSERT_EQ(Rs_gt.size(), Ts_gt.size());
        ASSERT_EQ(Rs.size(), Ts.size());
        ASSERT_EQ(Rs_gt.size(), Rs.size());

        const size_t num_cameras = Rs_gt.size();
        for (size_t c = 1; c < num_cameras; c++)
        {
            validateCameraPose(Rs[c], Ts[c], Rs_gt[c], Ts_gt[c], angle_tol, pos_tol);
        }
    }
};

TEST_F(MultiViewTest, OneLine)
{
    const string root = cvtest::TS::ptr()->get_data_path() + "cv/cameracalibration/multiview/3cams-one-line/";
    const std::vector<std::string> cam_names = {"cam_0", "cam_1", "cam_3"};
    const std::vector<cv::Size> image_sizes = {{1920, 1080}, {1920, 1080}, {1920, 1080} };
    std::vector<bool> is_fisheye(3, false);

    double rs_1_gt_data[9] = {
        0.9996914489704484, -0.01160060078752197, -0.02196435559568884,
        0.012283315339906, 0.9994374509454836, 0.03120739995344806,
        0.02158997497973892, -0.03146756598408248, 0.9992715673286274
    };
    double rs_2_gt_data[9] = {
        0.9988848194142131, -0.0255827884561986, -0.03968171466355882,
        0.0261796234191418, 0.999550713317242, 0.0145944792515729,
        0.03929051872229011, -0.0156170561181697, 0.9991057815350362
    };

    double ts_1_gt_data[3] = {0.5078811293323259, 0.002753469433719865, 0.02413521839310227};
    double ts_2_gt_data[3] = {1.007213763725429, 0.01645068247976361, 0.05394643957910365};

    std::vector<cv::Mat> Rs_gt = {
        cv::Mat::eye(3, 3, CV_64FC1),
        cv::Mat(3, 3, CV_64FC1, rs_1_gt_data),
        cv::Mat(3, 3, CV_64FC1, rs_2_gt_data)
    };

    std::vector<cv::Mat> Ts_gt = {
        cv::Mat::zeros(3, 1, CV_64FC1),
        cv::Mat(3, 1, CV_64FC1, ts_1_gt_data),
        cv::Mat(3, 1, CV_64FC1, ts_2_gt_data)
    };

    const int num_frames = 96;
    std::vector<std::vector<cv::Mat>> image_points_all;
    cv::Mat visibility;
    loadImagePoints(root, cam_names, num_frames, image_points_all, visibility);
    EXPECT_EQ(cam_names.size(), image_points_all.size());
    for(size_t i = 0; i < cam_names.size(); i++)
    {
        EXPECT_TRUE(!image_points_all[i].empty());
    }

    std::vector<cv::Vec3f> board_pattern = genAsymmetricObjectPoints();
    std::vector<std::vector<cv::Vec3f>> objPoints(num_frames, board_pattern);

    std::vector<int> flagsForIntrinsics(3, CALIB_RATIONAL_MODEL);

    std::vector<cv::Mat> Ks, distortions, Rs, Rs_rvec, Ts;
    cv::Mat errors_mat, output_pairs, rvecs0, tvecs0;
    double rms = calibrateMultiview(objPoints, image_points_all, image_sizes, visibility,
                    Rs_rvec, Ts, Ks, distortions, rvecs0, tvecs0, is_fisheye, errors_mat, output_pairs,
                    false, flagsForIntrinsics);
    CV_LOG_INFO(NULL, "RMS: "  << rms);

    EXPECT_LE(rms, .3);

    Rs.resize(Rs_rvec.size());
    for(int c = 0; c < 3; c++)
    {
        cv::Rodrigues(Rs_rvec[c], Rs[c]);
        CV_LOG_INFO(NULL, "R" << c << ":" << Rs[c]);
        CV_LOG_INFO(NULL, "T" << c << ":" << Ts[c]);
    }

    validateAllPoses(Rs_gt, Ts_gt, Rs, Ts);
}

TEST_F(MultiViewTest, CamsToFloor)
{
    const string root = cvtest::TS::ptr()->get_data_path() + "cv/cameracalibration/multiview/3cams-to-floor/";
    const std::vector<std::string> cam_names = {"cam_0", "cam_1", "cam_2"};
    std::vector<cv::Size> image_sizes = {{1920, 1080}, {1920, 1080}, {1280, 720}};
    std::vector<bool> is_fisheye(3, false);

    double rs_1_gt_data[9] = {
        -0.05217184989559624, 0.6470741242690249, -0.7606399777686852,
        -0.526982982144755, 0.6291523784496631, 0.5713634755748329,
        0.8482729717539585, 0.4306534133065782, 0.3081730082260634
    };
    double rs_2_gt_data[9] = {
        0.001580678474783847, -0.62542080411436, 0.7802860496231537,
        0.4843796328138114, 0.683118871472744, 0.5465573883435866,
        -0.8748564869569847, 0.3770907387072139, 0.304020890746888
    };

    double ts_1_gt_data[3] = {1.064278166833888, -0.7727142268275895, 1.140555926119704};
    double ts_2_gt_data[3] = {-0.9391478506021244, -1.048084838193036, 1.3973875466639};

    std::vector<cv::Mat> Rs_gt = {
        cv::Mat::eye(3, 3, CV_64FC1),
        cv::Mat(3, 3, CV_64FC1, rs_1_gt_data),
        cv::Mat(3, 3, CV_64FC1, rs_2_gt_data)
    };

    std::vector<cv::Mat> Ts_gt = {
        cv::Mat::zeros(3, 1, CV_64FC1),
        cv::Mat(3, 1, CV_64FC1, ts_1_gt_data),
        cv::Mat(3, 1, CV_64FC1, ts_2_gt_data)
    };

    const int num_frames = 125;
    std::vector<std::vector<cv::Mat>> image_points_all;
    cv::Mat visibility;
    loadImagePoints(root, cam_names, num_frames, image_points_all, visibility);
    EXPECT_EQ(cam_names.size(), image_points_all.size());
    for(size_t i = 0; i < cam_names.size(); i++)
    {
        EXPECT_TRUE(!image_points_all[i].empty());
    }

    std::vector<cv::Vec3f> board_pattern = genAsymmetricObjectPoints();
    std::vector<std::vector<cv::Vec3f>> objPoints(num_frames, board_pattern);

    std::vector<int> flagsForIntrinsics(3, cv::CALIB_RATIONAL_MODEL);

    std::vector<cv::Mat> Ks, distortions, Rs, Rs_rvec, Ts;
    cv::Mat errors_mat, output_pairs, rvecs0, tvecs0;
    double rms = calibrateMultiview(objPoints, image_points_all, image_sizes, visibility,
                    Rs_rvec, Ts, Ks, distortions, rvecs0, tvecs0, is_fisheye, errors_mat, output_pairs,
                    false, flagsForIntrinsics);
    CV_LOG_INFO(NULL, "RMS: "  << rms);

    EXPECT_LE(rms, 1.);

    Rs.resize(Rs_rvec.size());
    for(int c = 0; c < 3; c++)
    {
        cv::Rodrigues(Rs_rvec[c], Rs[c]);
        CV_LOG_INFO(NULL, "R" << c << ":" << Rs[c]);
        CV_LOG_INFO(NULL, "T" << c << ":" << Ts[c]);
    }

    validateAllPoses(Rs_gt, Ts_gt, Rs, Ts);
}

struct RegisterCamerasTest: public MultiViewTest
{
    double calibrateMono(const std::vector<cv::Vec3f>& board_pattern,
                         const std::vector<cv::Mat>& image_points,
                         const cv::Size& image_size,
                         cv::CameraModel model,
                         int flags,
                         Mat& K,
                         Mat& dist)
    {
        std::vector<cv::Mat> filtered_image_points;
        for(size_t i = 0; i < image_points.size(); i++)
        {
            if(!image_points[i].empty())
                filtered_image_points.push_back(image_points[i]);
        }
        std::vector<std::vector<cv::Vec3f>> objPoints(filtered_image_points.size(), board_pattern);

        std::vector<cv::Mat> rvec, tvec;
        cv::Mat K1, dist1;
        if(model == cv::CALIB_MODEL_PINHOLE)
        {
            return cv::calibrateCamera(objPoints, filtered_image_points, image_size, K, dist, rvec, tvec, flags);
        }
        else if(model == cv::CALIB_MODEL_FISHEYE)
        {
            return cv::fisheye::calibrate(objPoints, filtered_image_points, image_size, K, dist, rvec, tvec, flags);
        }
        else
        {
            CV_Error(Error::StsBadArg, "Unsupported camera model!");
        }

        return FLT_MAX;
    }

    void filterPoints(const std::vector<std::vector<cv::Mat>>& image_points_all,
                      std::vector<cv::Mat>& visible_image_points1,
                      std::vector<cv::Mat>& visible_image_points2)
    {
        for (size_t i = 0; i < std::min(image_points_all[0].size(), image_points_all[1].size()); i++)
        {
            if(!image_points_all[0][i].empty() && !image_points_all[1][i].empty())
            {
                visible_image_points1.push_back(image_points_all[0][i]);
                visible_image_points2.push_back(image_points_all[1][i]);
            }
        }
    }
};

TEST_F(RegisterCamerasTest, hetero1)
{
    const string root = cvtest::TS::ptr()->get_data_path() + "cv/cameracalibration/multiview/3cams-hetero/";
    const std::vector<std::string> cam_names = {"cam_7", "cam_4"};
    std::vector<cv::Size> image_sizes = {{1920, 1080}, {2048, 2048}};
    std::vector<cv::CameraModel> models = {cv::CALIB_MODEL_PINHOLE, cv::CALIB_MODEL_FISHEYE};
    std::vector<int> flagsForIntrinsics = {cv::CALIB_RATIONAL_MODEL, cv::CALIB_RECOMPUTE_EXTRINSIC+cv::CALIB_FIX_SKEW};
    const int num_frames = 127;
    std::vector<cv::Vec3f> board_pattern = genAsymmetricObjectPoints();

    double rs_1_gt_data[9] = {
        0.9923998627583629, 0.1102270543935739, 0.05470382872247866,
        -0.05295473891691575, -0.01873572048960163, 0.9984211377990636,
        0.1110779367085268, -0.9937298270945939, -0.01275628155556733
    };
    cv::Mat R_gt(3, 3, CV_64FC1, rs_1_gt_data);

    double ts_1_gt_data[3] = {0.5132123397314717, -0.345554256449513, 0.7851208074917889};
    cv::Mat T_gt(3, 1, CV_64FC1, ts_1_gt_data);

    std::vector<std::vector<cv::Mat>> image_points_all;
    cv::Mat visibility;
    loadImagePoints(root, cam_names, num_frames, image_points_all, visibility);
    EXPECT_EQ(cam_names.size(), image_points_all.size());
    for(size_t i = 0; i < cam_names.size(); i++)
    {
        EXPECT_TRUE(!image_points_all[i].empty());
    }

    cv::Mat K1, dist1;
    double rms = calibrateMono(board_pattern, image_points_all[0], image_sizes[0], models[0], flagsForIntrinsics[0], K1, dist1);
    CV_LOG_INFO(NULL, "Mono #1 RMS: "  << rms);
    EXPECT_LE(rms, 1.);

    cv::Mat K2, dist2;
    rms = calibrateMono(board_pattern, image_points_all[1], image_sizes[1], models[1], flagsForIntrinsics[1], K2, dist2);
    CV_LOG_INFO(NULL, "Mono #2 RMS: "  << rms);
    EXPECT_LE(rms, 1.);

    std::vector<cv::Mat> visible_image_points1, visible_image_points2;
    filterPoints(image_points_all, visible_image_points1, visible_image_points2);
    std::vector<std::vector<cv::Vec3f>> object_points(visible_image_points1.size(), board_pattern);

    cv::Mat R, T, E, F;
    cv::Mat rvec_reg, tvec_reg, per_view_err;
    rms = registerCameras(object_points, object_points, visible_image_points1, visible_image_points2,
                                 K1, dist1, cv::CALIB_MODEL_PINHOLE, K2, dist2, cv::CALIB_MODEL_FISHEYE,
                                 R, T, E, F, rvec_reg, tvec_reg, per_view_err);
    CV_LOG_INFO(NULL, "Register RMS: "  << rms);
    EXPECT_LE(rms, 1.);

    CV_LOG_INFO(NULL, "R:" << R);
    CV_LOG_INFO(NULL, "T:" << T);

    validateCameraPose(R, T, R_gt, T_gt);
}

TEST_F(RegisterCamerasTest, hetero2)
{
    const string root = cvtest::TS::ptr()->get_data_path() + "cv/cameracalibration/multiview/3cams-hetero/";
    const std::vector<std::string> cam_names = {"cam_4", "cam_8"};
    std::vector<cv::Size> image_sizes = {{2048, 2048}, {1920, 1080}};
    std::vector<cv::CameraModel> models = {cv::CALIB_MODEL_FISHEYE, cv::CALIB_MODEL_PINHOLE};
    std::vector<int> flagsForIntrinsics = { cv::CALIB_RECOMPUTE_EXTRINSIC+cv::CALIB_FIX_SKEW, cv::CALIB_RATIONAL_MODEL};
    const int num_frames = 127;
    std::vector<cv::Vec3f> board_pattern = genAsymmetricObjectPoints();

    double rs_1_gt_data[9] = {
        0.9987381520324473, -0.03742623778583679, 0.0334870183804049,
        0.03272769253311544, -0.02072052049800844, -0.9992494974588425,
        0.03809201775004091, 0.999084549352801, -0.01946949994840527
    };
    cv::Mat R_gt(3, 3, CV_64FC1, rs_1_gt_data);

    double ts_1_gt_data[3] = {0.4660746974363485, 0.7703195273112146, 0.3243138654899712};
    cv::Mat T_gt(3, 1, CV_64FC1, ts_1_gt_data);

    std::vector<std::vector<cv::Mat>> image_points_all;
    cv::Mat visibility;
    loadImagePoints(root, cam_names, num_frames, image_points_all, visibility);
    EXPECT_EQ(cam_names.size(), image_points_all.size());
    for(size_t i = 0; i < cam_names.size(); i++)
    {
        EXPECT_TRUE(!image_points_all[i].empty());
    }

    cv::Mat K1, dist1;
    double rms = calibrateMono(board_pattern, image_points_all[0], image_sizes[0], models[0], flagsForIntrinsics[0], K1, dist1);
    CV_LOG_INFO(NULL, "Mono #1 RMS: "  << rms);
    EXPECT_LE(rms, 1.);

    cv::Mat K2, dist2;
    rms = calibrateMono(board_pattern, image_points_all[1], image_sizes[1], models[1], flagsForIntrinsics[1], K2, dist2);
    CV_LOG_INFO(NULL, "Mono #2 RMS: "  << rms);
    EXPECT_LE(rms, 1.);

    std::vector<cv::Mat> visible_image_points1, visible_image_points2;
    filterPoints(image_points_all, visible_image_points1, visible_image_points2);
    std::vector<std::vector<cv::Vec3f>> object_points(visible_image_points1.size(), board_pattern);

    cv::Mat R, T, E, F;
    cv::Mat rvec_reg, tvec_reg, per_view_err;
    rms = registerCameras(object_points, object_points, visible_image_points1, visible_image_points2,
                                 K1, dist1, cv::CALIB_MODEL_FISHEYE, K2, dist2, cv::CALIB_MODEL_PINHOLE,
                                 R, T, E, F, rvec_reg, tvec_reg, per_view_err);
    CV_LOG_INFO(NULL, "Register RMS: "  << rms);
    EXPECT_LE(rms, 1.);

    CV_LOG_INFO(NULL, "R:" << R);
    CV_LOG_INFO(NULL, "T:" << T);

    validateCameraPose(R, T, R_gt, T_gt);
}

}}
