// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/calib3d.hpp"

namespace opencv_test { namespace {

static std::string getMethodName(HandEyeCalibrationMethod method)
{
    std::string method_name = "";
    switch (method)
    {
    case CALIB_HAND_EYE_TSAI:
        method_name = "Tsai";
        break;

    case CALIB_HAND_EYE_PARK:
        method_name = "Park";
        break;

    case CALIB_HAND_EYE_HORAUD:
        method_name = "Horaud";
        break;

    case CALIB_HAND_EYE_ANDREFF:
        method_name = "Andreff";
        break;

    case CALIB_HAND_EYE_DANIILIDIS:
        method_name = "Daniilidis";
        break;

    default:
        break;
    }

    return method_name;
}

class CV_CalibrateHandEyeTest : public cvtest::BaseTest
{
public:
    CV_CalibrateHandEyeTest() {
        eps_rvec[CALIB_HAND_EYE_TSAI] = 1.0e-8;
        eps_rvec[CALIB_HAND_EYE_PARK] = 1.0e-8;
        eps_rvec[CALIB_HAND_EYE_HORAUD] = 1.0e-8;
        eps_rvec[CALIB_HAND_EYE_ANDREFF] = 1.0e-8;
        eps_rvec[CALIB_HAND_EYE_DANIILIDIS] = 1.0e-8;

        eps_tvec[CALIB_HAND_EYE_TSAI] = 1.0e-8;
        eps_tvec[CALIB_HAND_EYE_PARK] = 1.0e-8;
        eps_tvec[CALIB_HAND_EYE_HORAUD] = 1.0e-8;
        eps_tvec[CALIB_HAND_EYE_ANDREFF] = 1.0e-8;
        eps_tvec[CALIB_HAND_EYE_DANIILIDIS] = 1.0e-8;

        eps_rvec_noise[CALIB_HAND_EYE_TSAI] = 2.0e-2;
        eps_rvec_noise[CALIB_HAND_EYE_PARK] = 2.0e-2;
        eps_rvec_noise[CALIB_HAND_EYE_HORAUD] = 2.0e-2;
        eps_rvec_noise[CALIB_HAND_EYE_ANDREFF] = 1.0e-2;
        eps_rvec_noise[CALIB_HAND_EYE_DANIILIDIS] = 1.0e-2;

        eps_tvec_noise[CALIB_HAND_EYE_TSAI] = 5.0e-2;
        eps_tvec_noise[CALIB_HAND_EYE_PARK] = 5.0e-2;
        eps_tvec_noise[CALIB_HAND_EYE_HORAUD] = 5.0e-2;
        eps_tvec_noise[CALIB_HAND_EYE_ANDREFF] = 5.0e-2;
        eps_tvec_noise[CALIB_HAND_EYE_DANIILIDIS] = 5.0e-2;
    }
protected:
    virtual void run(int);
    void generatePose(RNG& rng, double min_theta, double max_theta,
                      double min_tx, double max_tx,
                      double min_ty, double max_ty,
                      double min_tz, double max_tz,
                      Mat& R, Mat& tvec,
                      bool randSign=false);
    void simulateData(RNG& rng, int nPoses,
                      std::vector<Mat> &R_gripper2base, std::vector<Mat> &t_gripper2base,
                      std::vector<Mat> &R_target2cam, std::vector<Mat> &t_target2cam,
                      bool noise, Mat& R_cam2gripper, Mat& t_cam2gripper);
    Mat homogeneousInverse(const Mat& T);
    double sign_double(double val);

    double eps_rvec[5];
    double eps_tvec[5];
    double eps_rvec_noise[5];
    double eps_tvec_noise[5];
};

void CV_CalibrateHandEyeTest::run(int)
{
    ts->set_failed_test_info(cvtest::TS::OK);

    RNG& rng = ts->get_rng();

    std::vector<std::vector<double> > vec_rvec_diff(5);
    std::vector<std::vector<double> > vec_tvec_diff(5);
    std::vector<std::vector<double> > vec_rvec_diff_noise(5);
    std::vector<std::vector<double> > vec_tvec_diff_noise(5);

    std::vector<HandEyeCalibrationMethod> methods;
    methods.push_back(CALIB_HAND_EYE_TSAI);
    methods.push_back(CALIB_HAND_EYE_PARK);
    methods.push_back(CALIB_HAND_EYE_HORAUD);
    methods.push_back(CALIB_HAND_EYE_ANDREFF);
    methods.push_back(CALIB_HAND_EYE_DANIILIDIS);

    const int nTests = 100;
    for (int i = 0; i < nTests; i++)
    {
        const int nPoses = 10;
        {
            //No noise
            std::vector<Mat> R_gripper2base, t_gripper2base;
            std::vector<Mat> R_target2cam, t_target2cam;
            Mat R_cam2gripper_true, t_cam2gripper_true;

            const bool noise = false;
            simulateData(rng, nPoses, R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, noise, R_cam2gripper_true, t_cam2gripper_true);

            for (size_t idx = 0; idx < methods.size(); idx++)
            {
                Mat rvec_cam2gripper_true;
                cv::Rodrigues(R_cam2gripper_true, rvec_cam2gripper_true);

                Mat R_cam2gripper_est, t_cam2gripper_est;
                calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, R_cam2gripper_est, t_cam2gripper_est, methods[idx]);

                Mat rvec_cam2gripper_est;
                cv::Rodrigues(R_cam2gripper_est, rvec_cam2gripper_est);

                double rvecDiff = cvtest::norm(rvec_cam2gripper_true, rvec_cam2gripper_est, NORM_L2);
                double tvecDiff = cvtest::norm(t_cam2gripper_true, t_cam2gripper_est, NORM_L2);

                vec_rvec_diff[idx].push_back(rvecDiff);
                vec_tvec_diff[idx].push_back(tvecDiff);

                const double epsilon_rvec = eps_rvec[idx];
                const double epsilon_tvec = eps_tvec[idx];

                //Maybe a better accuracy test would be to compare the mean and std errors with some thresholds?
                if (rvecDiff > epsilon_rvec || tvecDiff > epsilon_tvec)
                {
                    ts->printf(cvtest::TS::LOG, "Invalid accuracy (no noise) for method: %s, rvecDiff: %f, epsilon_rvec: %f, tvecDiff: %f, epsilon_tvec: %f\n",
                               getMethodName(methods[idx]).c_str(), rvecDiff, epsilon_rvec, tvecDiff, epsilon_tvec);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                }
            }
        }

        {
            //Gaussian noise on transformations between calibration target frame and camera frame and between gripper and robot base frames
            std::vector<Mat> R_gripper2base, t_gripper2base;
            std::vector<Mat> R_target2cam, t_target2cam;
            Mat R_cam2gripper_true, t_cam2gripper_true;

            const bool noise = true;
            simulateData(rng, nPoses, R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, noise, R_cam2gripper_true, t_cam2gripper_true);

            for (size_t idx = 0; idx < methods.size(); idx++)
            {
                Mat rvec_cam2gripper_true;
                cv::Rodrigues(R_cam2gripper_true, rvec_cam2gripper_true);

                Mat R_cam2gripper_est, t_cam2gripper_est;
                calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, R_cam2gripper_est, t_cam2gripper_est, methods[idx]);

                Mat rvec_cam2gripper_est;
                cv::Rodrigues(R_cam2gripper_est, rvec_cam2gripper_est);

                double rvecDiff = cvtest::norm(rvec_cam2gripper_true, rvec_cam2gripper_est, NORM_L2);
                double tvecDiff = cvtest::norm(t_cam2gripper_true, t_cam2gripper_est, NORM_L2);

                vec_rvec_diff_noise[idx].push_back(rvecDiff);
                vec_tvec_diff_noise[idx].push_back(tvecDiff);

                const double epsilon_rvec = eps_rvec_noise[idx];
                const double epsilon_tvec = eps_tvec_noise[idx];

                //Maybe a better accuracy test would be to compare the mean and std errors with some thresholds?
                if (rvecDiff > epsilon_rvec || tvecDiff > epsilon_tvec)
                {
                    ts->printf(cvtest::TS::LOG, "Invalid accuracy (noise) for method: %s, rvecDiff: %f, epsilon_rvec: %f, tvecDiff: %f, epsilon_tvec: %f\n",
                               getMethodName(methods[idx]).c_str(), rvecDiff, epsilon_rvec, tvecDiff, epsilon_tvec);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                }
            }
        }
    }

    for (size_t idx = 0; idx < methods.size(); idx++)
    {
        {
            double max_rvec_diff = *std::max_element(vec_rvec_diff[idx].begin(), vec_rvec_diff[idx].end());
            double mean_rvec_diff = std::accumulate(vec_rvec_diff[idx].begin(),
                                                    vec_rvec_diff[idx].end(), 0.0) / vec_rvec_diff[idx].size();
            double sq_sum_rvec_diff = std::inner_product(vec_rvec_diff[idx].begin(), vec_rvec_diff[idx].end(),
                                                         vec_rvec_diff[idx].begin(), 0.0);
            double std_rvec_diff = std::sqrt(sq_sum_rvec_diff / vec_rvec_diff[idx].size() - mean_rvec_diff * mean_rvec_diff);

            double max_tvec_diff = *std::max_element(vec_tvec_diff[idx].begin(), vec_tvec_diff[idx].end());
            double mean_tvec_diff = std::accumulate(vec_tvec_diff[idx].begin(),
                                                    vec_tvec_diff[idx].end(), 0.0) / vec_tvec_diff[idx].size();
            double sq_sum_tvec_diff = std::inner_product(vec_tvec_diff[idx].begin(), vec_tvec_diff[idx].end(),
                                                         vec_tvec_diff[idx].begin(), 0.0);
            double std_tvec_diff = std::sqrt(sq_sum_tvec_diff / vec_tvec_diff[idx].size() - mean_tvec_diff * mean_tvec_diff);

            std::cout << "\nMethod " << getMethodName(methods[idx]) << ":\n"
                      << "Max rvec error: " << max_rvec_diff << ", Mean rvec error: " << mean_rvec_diff
                      << ", Std rvec error: " << std_rvec_diff << "\n"
                      << "Max tvec error: " << max_tvec_diff << ", Mean tvec error: " << mean_tvec_diff
                      << ", Std tvec error: " << std_tvec_diff << std::endl;
        }
        {
            double max_rvec_diff = *std::max_element(vec_rvec_diff_noise[idx].begin(), vec_rvec_diff_noise[idx].end());
            double mean_rvec_diff = std::accumulate(vec_rvec_diff_noise[idx].begin(),
                                                    vec_rvec_diff_noise[idx].end(), 0.0) / vec_rvec_diff_noise[idx].size();
            double sq_sum_rvec_diff = std::inner_product(vec_rvec_diff_noise[idx].begin(), vec_rvec_diff_noise[idx].end(),
                                                         vec_rvec_diff_noise[idx].begin(), 0.0);
            double std_rvec_diff = std::sqrt(sq_sum_rvec_diff / vec_rvec_diff_noise[idx].size() - mean_rvec_diff * mean_rvec_diff);

            double max_tvec_diff = *std::max_element(vec_tvec_diff_noise[idx].begin(), vec_tvec_diff_noise[idx].end());
            double mean_tvec_diff = std::accumulate(vec_tvec_diff_noise[idx].begin(),
                                                    vec_tvec_diff_noise[idx].end(), 0.0) / vec_tvec_diff_noise[idx].size();
            double sq_sum_tvec_diff = std::inner_product(vec_tvec_diff_noise[idx].begin(), vec_tvec_diff_noise[idx].end(),
                                                         vec_tvec_diff_noise[idx].begin(), 0.0);
            double std_tvec_diff = std::sqrt(sq_sum_tvec_diff / vec_tvec_diff_noise[idx].size() - mean_tvec_diff * mean_tvec_diff);

            std::cout << "Method (noise) " << getMethodName(methods[idx]) << ":\n"
                      << "Max rvec error: " << max_rvec_diff << ", Mean rvec error: " << mean_rvec_diff
                      << ", Std rvec error: " << std_rvec_diff << "\n"
                      << "Max tvec error: " << max_tvec_diff << ", Mean tvec error: " << mean_tvec_diff
                      << ", Std tvec error: " << std_tvec_diff << std::endl;
        }
    }
}

void CV_CalibrateHandEyeTest::generatePose(RNG& rng, double min_theta, double max_theta,
                                               double min_tx, double max_tx,
                                               double min_ty, double max_ty,
                                               double min_tz, double max_tz,
                                               Mat& R, Mat& tvec,
                                               bool random_sign)
{
    Mat axis(3, 1, CV_64FC1);
    for (int i = 0; i < 3; i++)
    {
        axis.at<double>(i,0) = rng.uniform(-1.0, 1.0);
    }
    double theta = rng.uniform(min_theta, max_theta);
    if (random_sign)
    {
        theta *= sign_double(rng.uniform(-1.0, 1.0));
    }

    Mat rvec(3, 1, CV_64FC1);
    rvec.at<double>(0,0) = theta*axis.at<double>(0,0);
    rvec.at<double>(1,0) = theta*axis.at<double>(1,0);
    rvec.at<double>(2,0) = theta*axis.at<double>(2,0);

    tvec.create(3, 1, CV_64FC1);
    tvec.at<double>(0,0) = rng.uniform(min_tx, max_tx);
    tvec.at<double>(1,0) = rng.uniform(min_ty, max_ty);
    tvec.at<double>(2,0) = rng.uniform(min_tz, max_tz);

    if (random_sign)
    {
        tvec.at<double>(0,0) *= sign_double(rng.uniform(-1.0, 1.0));
        tvec.at<double>(1,0) *= sign_double(rng.uniform(-1.0, 1.0));
        tvec.at<double>(2,0) *= sign_double(rng.uniform(-1.0, 1.0));
    }

    cv::Rodrigues(rvec, R);
}

void CV_CalibrateHandEyeTest::simulateData(RNG& rng, int nPoses,
                                               std::vector<Mat> &R_gripper2base, std::vector<Mat> &t_gripper2base,
                                               std::vector<Mat> &R_target2cam, std::vector<Mat> &t_target2cam,
                                               bool noise, Mat& R_cam2gripper, Mat& t_cam2gripper)
{
    //to avoid generating values close to zero,
    //we use positive range values and randomize the sign
    const bool random_sign = true;
    generatePose(rng, 10.0*CV_PI/180.0, 50.0*CV_PI/180.0,
                 0.05, 0.5, 0.05, 0.5, 0.05, 0.5,
                 R_cam2gripper, t_cam2gripper, random_sign);

    Mat R_target2base, t_target2base;
    generatePose(rng, 5.0*CV_PI/180.0, 85.0*CV_PI/180.0,
                 0.5, 3.5, 0.5, 3.5, 0.5, 3.5,
                 R_target2base, t_target2base, random_sign);

    for (int i = 0; i < nPoses; i++)
    {
        Mat R_gripper2base_, t_gripper2base_;
        generatePose(rng, 5.0*CV_PI/180.0, 45.0*CV_PI/180.0,
                     0.5, 1.5, 0.5, 1.5, 0.5, 1.5,
                     R_gripper2base_, t_gripper2base_, random_sign);

        R_gripper2base.push_back(R_gripper2base_);
        t_gripper2base.push_back(t_gripper2base_);

        Mat T_cam2gripper = Mat::eye(4, 4, CV_64FC1);
        R_cam2gripper.copyTo(T_cam2gripper(Rect(0, 0, 3, 3)));
        t_cam2gripper.copyTo(T_cam2gripper(Rect(3, 0, 1, 3)));

        Mat T_gripper2base = Mat::eye(4, 4, CV_64FC1);
        R_gripper2base_.copyTo(T_gripper2base(Rect(0, 0, 3, 3)));
        t_gripper2base_.copyTo(T_gripper2base(Rect(3, 0, 1, 3)));

        Mat T_base2cam = homogeneousInverse(T_cam2gripper) * homogeneousInverse(T_gripper2base);
        Mat T_target2base = Mat::eye(4, 4, CV_64FC1);
        R_target2base.copyTo(T_target2base(Rect(0, 0, 3, 3)));
        t_target2base.copyTo(T_target2base(Rect(3, 0, 1, 3)));
        Mat T_target2cam = T_base2cam * T_target2base;

        if (noise)
        {
            //Add some noise for the transformation between the target and the camera
            Mat R_target2cam_noise = T_target2cam(Rect(0, 0, 3, 3));
            Mat rvec_target2cam_noise;
            cv::Rodrigues(R_target2cam_noise, rvec_target2cam_noise);
            rvec_target2cam_noise.at<double>(0,0) += rng.gaussian(0.002);
            rvec_target2cam_noise.at<double>(1,0) += rng.gaussian(0.002);
            rvec_target2cam_noise.at<double>(2,0) += rng.gaussian(0.002);

            cv::Rodrigues(rvec_target2cam_noise, R_target2cam_noise);

            Mat t_target2cam_noise = T_target2cam(Rect(3, 0, 1, 3));
            t_target2cam_noise.at<double>(0,0) += rng.gaussian(0.005);
            t_target2cam_noise.at<double>(1,0) += rng.gaussian(0.005);
            t_target2cam_noise.at<double>(2,0) += rng.gaussian(0.005);

            //Add some noise for the transformation between the gripper and the robot base
            Mat R_gripper2base_noise = T_gripper2base(Rect(0, 0, 3, 3));
            Mat rvec_gripper2base_noise;
            cv::Rodrigues(R_gripper2base_noise, rvec_gripper2base_noise);
            rvec_gripper2base_noise.at<double>(0,0) += rng.gaussian(0.001);
            rvec_gripper2base_noise.at<double>(1,0) += rng.gaussian(0.001);
            rvec_gripper2base_noise.at<double>(2,0) += rng.gaussian(0.001);

            cv::Rodrigues(rvec_gripper2base_noise, R_gripper2base_noise);

            Mat t_gripper2base_noise = T_gripper2base(Rect(3, 0, 1, 3));
            t_gripper2base_noise.at<double>(0,0) += rng.gaussian(0.001);
            t_gripper2base_noise.at<double>(1,0) += rng.gaussian(0.001);
            t_gripper2base_noise.at<double>(2,0) += rng.gaussian(0.001);
        }

        // test rvec represenation
        Mat rvec_target2cam;
        cv::Rodrigues(T_target2cam(Rect(0, 0, 3, 3)), rvec_target2cam);
        R_target2cam.push_back(rvec_target2cam);
        t_target2cam.push_back(T_target2cam(Rect(3, 0, 1, 3)));
    }
}

Mat CV_CalibrateHandEyeTest::homogeneousInverse(const Mat& T)
{
    CV_Assert( T.rows == 4 && T.cols == 4 );

    Mat R = T(Rect(0, 0, 3, 3));
    Mat t = T(Rect(3, 0, 1, 3));
    Mat Rt = R.t();
    Mat tinv = -Rt * t;
    Mat Tinv = Mat::eye(4, 4, T.type());
    Rt.copyTo(Tinv(Rect(0, 0, 3, 3)));
    tinv.copyTo(Tinv(Rect(3, 0, 1, 3)));

    return Tinv;
}

double CV_CalibrateHandEyeTest::sign_double(double val)
{
    return (0 < val) - (val < 0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Calib3d_CalibrateHandEye, regression) { CV_CalibrateHandEyeTest test; test.safe_run(); }

TEST(Calib3d_CalibrateHandEye, regression_17986)
{
    const std::string camera_poses_filename = findDataFile("cv/hand_eye_calibration/cali.txt");
    const std::string end_effector_poses = findDataFile("cv/hand_eye_calibration/robot_cali.txt");

    std::vector<Mat> R_target2cam;
    std::vector<Mat> t_target2cam;
    // Parse camera poses
    {
        std::ifstream file(camera_poses_filename.c_str());
        ASSERT_TRUE(file.is_open());

        int ndata = 0;
        file >> ndata;
        R_target2cam.reserve(ndata);
        t_target2cam.reserve(ndata);

        std::string image_name;
        Matx33d cameraMatrix;
        Matx33d R;
        Matx31d t;
        Matx16d distCoeffs;
        Matx13d distCoeffs2;
        while (file >> image_name >>
               cameraMatrix(0,0) >> cameraMatrix(0,1) >> cameraMatrix(0,2) >>
               cameraMatrix(1,0) >> cameraMatrix(1,1) >> cameraMatrix(1,2) >>
               cameraMatrix(2,0) >> cameraMatrix(2,1) >> cameraMatrix(2,2) >>
               R(0,0) >> R(0,1) >> R(0,2) >>
               R(1,0) >> R(1,1) >> R(1,2) >>
               R(2,0) >> R(2,1) >> R(2,2) >>
               t(0) >> t(1) >> t(2) >>
               distCoeffs(0) >> distCoeffs(1) >> distCoeffs(2) >> distCoeffs(3) >> distCoeffs(4) >>
               distCoeffs2(0) >> distCoeffs2(1) >> distCoeffs2(2)) {
            R_target2cam.push_back(Mat(R));
            t_target2cam.push_back(Mat(t));
        }
    }

    std::vector<Mat> R_gripper2base;
    std::vector<Mat> t_gripper2base;
    // Parse end-effector poses
    {
        std::ifstream file(end_effector_poses.c_str());
        ASSERT_TRUE(file.is_open());

        int ndata = 0;
        file >> ndata;
        R_gripper2base.reserve(ndata);
        t_gripper2base.reserve(ndata);

        Matx33d R;
        Matx31d t;
        Matx14d last_row;
        while (file >>
               R(0,0) >> R(0,1) >> R(0,2) >> t(0) >>
               R(1,0) >> R(1,1) >> R(1,2) >> t(1) >>
               R(2,0) >> R(2,1) >> R(2,2) >> t(2) >>
               last_row(0) >> last_row(1) >> last_row(2) >> last_row(3)) {
            R_gripper2base.push_back(Mat(R));
            t_gripper2base.push_back(Mat(t));
        }
    }

    std::vector<HandEyeCalibrationMethod> methods;
    methods.push_back(CALIB_HAND_EYE_TSAI);
    methods.push_back(CALIB_HAND_EYE_PARK);
    methods.push_back(CALIB_HAND_EYE_HORAUD);
    methods.push_back(CALIB_HAND_EYE_ANDREFF);
    methods.push_back(CALIB_HAND_EYE_DANIILIDIS);

    for (size_t idx = 0; idx < methods.size(); idx++) {
        SCOPED_TRACE(cv::format("method=%s", getMethodName(methods[idx]).c_str()));

        Matx33d R_cam2gripper_est;
        Matx31d t_cam2gripper_est;
        calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, R_cam2gripper_est, t_cam2gripper_est, methods[idx]);

        EXPECT_TRUE(checkRange(R_cam2gripper_est));
        EXPECT_TRUE(checkRange(t_cam2gripper_est));
    }
}

TEST(Calib3d_CalibrateHandEye, regression_24871)
{
    RNG& rng = cv::theRNG();
    std::vector<Mat> R_target2cam, t_target2cam;
    std::vector<Mat> R_gripper2base, t_gripper2base;
    Mat T_true_cam2gripper;

    T_true_cam2gripper = (cv::Mat_<double>(4, 4) <<  0,  0, -1, 0.1,
                                                     1,  0,  0, 0.2,
                                                     0, -1,  0, 0.3,
                                                     0,  0,  0, 1);

    R_target2cam.push_back((cv::Mat_<double>(3, 3) <<
            0.04964505493834381, 0.5136826827431226, 0.8565427426404346,
            -0.3923117691818854, 0.7987004864191318, -0.4562554205214679,
            -0.9184916136152514, -0.3133809733274676, 0.2411752915926112));
    t_target2cam.push_back((cv::Mat_<double>(3, 1) <<
            -1.588728904724121,
            0.07843752950429916,
            -1.002813339233398));

    R_gripper2base.push_back((cv::Mat_<double>(3, 3) <<
            -0.4143743581399177, -0.6105088815982459, -0.6749613298595637,
            -0.1598851232573451, -0.6812625208693498, 0.71436554019614,
            -0.895952364066927, 0.4039310376145889, 0.1846864320259794));
    t_gripper2base.push_back((cv::Mat_<double>(3, 1) <<
            -1.249274406461827,
            -1.916570771580279,
            2.005069553422765));

    R_target2cam.push_back((cv::Mat_<double>(3, 3) <<
            -0.3048000068139332, 0.6971848192711539, 0.6488684640388026,
            -0.9377589344241749, -0.3387497187353627, -0.07652979135179161,
            0.1664486009369332, -0.6318084803439735, 0.7570422097951847));
    t_target2cam.push_back((cv::Mat_<double>(3, 1) <<
            -1.906493663787842,
            -0.07281044125556946,
            0.6088893413543701));

    R_gripper2base.push_back((cv::Mat_<double>(3, 3) <<
            0.7262439860936567, -0.201662933718935, -0.6571923111439066,
            -0.4640017362244384, -0.8491808316335328, -0.2521791108852766,
            -0.5072199339965884, 0.4880819361030014, -0.7102844234575628));
    t_gripper2base.push_back((cv::Mat_<double>(3, 1) <<
            -0.7375172846804027,
            -2.579760910816792,
            1.336561572270101));

    R_target2cam.push_back((cv::Mat_<double>(3, 3) <<
            -0.590234879685801, -0.7051138289845309, -0.3929850823848928,
            0.6017371069678565, -0.7088332765096816, 0.3680595606834615,
            -0.5380847896941907, -0.01923211603859842, 0.8426712792141644));
    t_target2cam.push_back((cv::Mat_<double>(3, 1) <<
            -0.9809040427207947,
            -0.2707894444465637,
            -0.2577074766159058));

    R_gripper2base.push_back((cv::Mat_<double>(3, 3) <<
            0.2541996332132083, 0.6186461729765909, 0.7434106934499181,
            0.2194912986375709, 0.711701808961156, -0.6673111005698995,
            -0.9419161938817396, 0.3328024155303503, 0.04512688689130734));
    t_gripper2base.push_back((cv::Mat_<double>(3, 1) <<
            -1.040123533893404,
            -0.1303773962721222,
            1.068029475621886));

    R_target2cam.push_back((cv::Mat_<double>(3, 3) <<
            0.7643667483125168, -0.08523002870239212, 0.63912386614923,
            -0.2583463792779588, 0.8676987164647345, 0.424683512464778,
            -0.5907627462764713, -0.489729292214425, 0.6412211770980741));
    t_target2cam.push_back((cv::Mat_<double>(3, 1) <<
            -1.58987033367157,
            -1.924914002418518,
            -0.3109001517295837));

    R_gripper2base.push_back((cv::Mat_<double>(3, 3) <<
            0.116348305340805, -0.9917998080681939, 0.0528792261688552,
            -0.2760629007224059, 0.01884966191381591, 0.9609547154213178,
            -0.9540714578526358, -0.1264034452126562, -0.2716060057313114));
    t_gripper2base.push_back((cv::Mat_<double>(3, 1) <<
            -2.551899142554571,
            -2.986937398237611,
            1.317613923218308));

    Mat R_true_cam2gripper;
    Mat t_true_cam2gripper;
    R_true_cam2gripper = T_true_cam2gripper(Rect(0, 0, 3, 3));
    t_true_cam2gripper = T_true_cam2gripper(Rect(3, 0, 1, 3));

    std::vector<HandEyeCalibrationMethod> methods = {CALIB_HAND_EYE_TSAI,
                                                     CALIB_HAND_EYE_PARK,
                                                     CALIB_HAND_EYE_HORAUD,
                                                     CALIB_HAND_EYE_ANDREFF,
                                                     CALIB_HAND_EYE_DANIILIDIS};

    for (auto method : methods) {
        SCOPED_TRACE(cv::format("method=%s", getMethodName(method).c_str()));

        Matx33d R_cam2gripper_est;
        Matx31d t_cam2gripper_est;
        calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, R_cam2gripper_est, t_cam2gripper_est, method);

        EXPECT_TRUE(cv::norm(R_cam2gripper_est - R_true_cam2gripper) < 1e-12);
        EXPECT_TRUE(cv::norm(t_cam2gripper_est - t_true_cam2gripper) < 1e-12);
    }
}

}} // namespace
