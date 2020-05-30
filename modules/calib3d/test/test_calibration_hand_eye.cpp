// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/calib3d.hpp"

namespace opencv_test { namespace {

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
    std::string getMethodName(HandEyeCalibrationMethod method);
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

        R_target2cam.push_back(T_target2cam(Rect(0, 0, 3, 3)));
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

std::string CV_CalibrateHandEyeTest::getMethodName(HandEyeCalibrationMethod method)
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

double CV_CalibrateHandEyeTest::sign_double(double val)
{
    return (0 < val) - (val < 0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Calib3d_CalibrateHandEye, regression) { CV_CalibrateHandEyeTest test; test.safe_run(); }

}} // namespace
