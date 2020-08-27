// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/calib3d.hpp"

namespace opencv_test { namespace {

static void generatePose(RNG& rng, double min_theta, double max_theta,
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
        theta *= std::copysign(1.0, rng.uniform(-1.0, 1.0));
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
        tvec.at<double>(0,0) *= std::copysign(1.0, rng.uniform(-1.0, 1.0));
        tvec.at<double>(1,0) *= std::copysign(1.0, rng.uniform(-1.0, 1.0));
        tvec.at<double>(2,0) *= std::copysign(1.0, rng.uniform(-1.0, 1.0));
    }

    cv::Rodrigues(rvec, R);
}

static Mat homogeneousInverse(const Mat& T)
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

static void simulateDataEyeInHand(RNG& rng, int nPoses,
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

        //Test rvec representation
        Mat rvec_target2cam;
        cv::Rodrigues(T_target2cam(Rect(0, 0, 3, 3)), rvec_target2cam);
        R_target2cam.push_back(rvec_target2cam);
        t_target2cam.push_back(T_target2cam(Rect(3, 0, 1, 3)));
    }
}

static void simulateDataEyeToHand(RNG& rng, int nPoses,
                                  std::vector<Mat> &R_base2gripper, std::vector<Mat> &t_base2gripper,
                                  std::vector<Mat> &R_target2cam, std::vector<Mat> &t_target2cam,
                                  bool noise, Mat& R_cam2base, Mat& t_cam2base)
{
    //to avoid generating values close to zero,
    //we use positive range values and randomize the sign
    const bool random_sign = true;
    generatePose(rng, 10.0*CV_PI/180.0, 50.0*CV_PI/180.0,
                 0.5, 3.5, 0.5, 3.5, 0.5, 3.5,
                 R_cam2base, t_cam2base, random_sign);

    Mat R_target2gripper, t_target2gripper;
    generatePose(rng, 5.0*CV_PI/180.0, 85.0*CV_PI/180.0,
                 0.05, 0.5, 0.05, 0.5, 0.05, 0.5,
                 R_target2gripper, t_target2gripper, random_sign);

    Mat T_target2gripper = Mat::eye(4, 4, CV_64FC1);
    R_target2gripper.copyTo(T_target2gripper(Rect(0, 0, 3, 3)));
    t_target2gripper.copyTo(T_target2gripper(Rect(3, 0, 1, 3)));

    for (int i = 0; i < nPoses; i++)
    {
        Mat R_gripper2base_, t_gripper2base_;
        generatePose(rng, 5.0*CV_PI/180.0, 45.0*CV_PI/180.0,
                     0.5, 1.5, 0.5, 1.5, 0.5, 1.5,
                     R_gripper2base_, t_gripper2base_, random_sign);

        Mat R_base2gripper_ = R_gripper2base_.t();
        Mat t_base2gripper_ = -R_base2gripper_ * t_gripper2base_;

        Mat T_gripper2base = Mat::eye(4, 4, CV_64FC1);
        R_gripper2base_.copyTo(T_gripper2base(Rect(0, 0, 3, 3)));
        t_gripper2base_.copyTo(T_gripper2base(Rect(3, 0, 1, 3)));

        Mat T_cam2base = Mat::eye(4, 4, CV_64FC1);
        R_cam2base.copyTo(T_cam2base(Rect(0, 0, 3, 3)));
        t_cam2base.copyTo(T_cam2base(Rect(3, 0, 1, 3)));

        Mat T_target2cam = homogeneousInverse(T_cam2base) * T_gripper2base * T_target2gripper;

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

            //Add some noise for the transformation between the robot base and the gripper
            Mat rvec_base2gripper_noise;
            cv::Rodrigues(R_base2gripper_, rvec_base2gripper_noise);
            rvec_base2gripper_noise.at<double>(0,0) += rng.gaussian(0.001);
            rvec_base2gripper_noise.at<double>(1,0) += rng.gaussian(0.001);
            rvec_base2gripper_noise.at<double>(2,0) += rng.gaussian(0.001);

            cv::Rodrigues(rvec_base2gripper_noise, R_base2gripper_);

            t_base2gripper_.at<double>(0,0) += rng.gaussian(0.001);
            t_base2gripper_.at<double>(1,0) += rng.gaussian(0.001);
            t_base2gripper_.at<double>(2,0) += rng.gaussian(0.001);
        }

        R_base2gripper.push_back(R_base2gripper_);
        t_base2gripper.push_back(t_base2gripper_);

        //Test rvec representation
        Mat rvec_target2cam;
        cv::Rodrigues(T_target2cam(Rect(0, 0, 3, 3)), rvec_target2cam);
        R_target2cam.push_back(rvec_target2cam);
        t_target2cam.push_back(T_target2cam(Rect(3, 0, 1, 3)));
    }
}

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

static std::string getMethodName(RobotWorldHandEyeCalibrationMethod method)
{
    std::string method_name = "";
    switch (method)
    {
    case CALIB_ROBOT_WORLD_HAND_EYE_SHAH:
        method_name = "Shah";
        break;

    case CALIB_ROBOT_WORLD_HAND_EYE_LI:
        method_name = "Li";
        break;

    default:
        break;
    }

    return method_name;
}

static void printStats(const std::string& methodName, const std::vector<double>& rvec_diff, const std::vector<double>& tvec_diff)
{
    double max_rvec_diff = *std::max_element(rvec_diff.begin(), rvec_diff.end());
    double mean_rvec_diff = std::accumulate(rvec_diff.begin(),
                                            rvec_diff.end(), 0.0) / rvec_diff.size();
    double sq_sum_rvec_diff = std::inner_product(rvec_diff.begin(), rvec_diff.end(),
                                                 rvec_diff.begin(), 0.0);
    double std_rvec_diff = std::sqrt(sq_sum_rvec_diff / rvec_diff.size() - mean_rvec_diff * mean_rvec_diff);

    double max_tvec_diff = *std::max_element(tvec_diff.begin(), tvec_diff.end());
    double mean_tvec_diff = std::accumulate(tvec_diff.begin(),
                                            tvec_diff.end(), 0.0) / tvec_diff.size();
    double sq_sum_tvec_diff = std::inner_product(tvec_diff.begin(), tvec_diff.end(),
                                                 tvec_diff.begin(), 0.0);
    double std_tvec_diff = std::sqrt(sq_sum_tvec_diff / tvec_diff.size() - mean_tvec_diff * mean_tvec_diff);

    std::cout << "Method " << methodName << ":\n"
              << "Max rvec error: " << max_rvec_diff << ", Mean rvec error: " << mean_rvec_diff
              << ", Std rvec error: " << std_rvec_diff << "\n"
              << "Max tvec error: " << max_tvec_diff << ", Mean tvec error: " << mean_tvec_diff
              << ", Std tvec error: " << std_tvec_diff << std::endl;
}

static void loadDataset(std::vector<Mat>& R_target2cam, std::vector<Mat>& t_target2cam,
                        std::vector<Mat>& R_base2gripper, std::vector<Mat>& t_base2gripper)
{
    const std::string camera_poses_filename = findDataFile("cv/robot_world_hand_eye_calibration/cali.txt");
    const std::string end_effector_poses = findDataFile("cv/robot_world_hand_eye_calibration/robot_cali.txt");

    // Parse camera poses, the pose of the chessboard in the camera frame
    {
        std::ifstream file(camera_poses_filename);
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

    // Parse robot poses, the pose of the robot base in the robot hand frame
    {
        std::ifstream file(end_effector_poses);
        ASSERT_TRUE(file.is_open());

        int ndata = 0;
        file >> ndata;
        R_base2gripper.reserve(ndata);
        t_base2gripper.reserve(ndata);

        Matx33d R;
        Matx31d t;
        Matx14d last_row;
        while (file >>
               R(0,0) >> R(0,1) >> R(0,2) >> t(0) >>
               R(1,0) >> R(1,1) >> R(1,2) >> t(1) >>
               R(2,0) >> R(2,1) >> R(2,2) >> t(2) >>
               last_row(0) >> last_row(1) >> last_row(2) >> last_row(3)) {
            R_base2gripper.push_back(Mat(R));
            t_base2gripper.push_back(Mat(t));
        }
    }
}

static void loadResults(Matx33d& wRb, Matx31d& wtb, Matx33d& cRg, Matx31d& ctg)
{
    const std::string transformations_filename = findDataFile("cv/robot_world_hand_eye_calibration/rwhe_AA_RPI/transformations.txt");
    std::ifstream file(transformations_filename);
    ASSERT_TRUE(file.is_open());

    std::string str;
    //Parse X
    file >> str;
    Matx44d wTb;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            file >> wTb(i,j);
        }
    }

    //Parse Z
    file >> str;
    int cam_num = 0;
    //Parse camera number
    file >> cam_num;
    Matx44d cTg;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            file >> cTg(i,j);
        }
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            wRb(i,j) = wTb(i,j);
            cRg(i,j) = cTg(i,j);
        }
        wtb(i) = wTb(i,3);
        ctg(i) = cTg(i,3);
    }
}

class CV_CalibrateHandEyeTest : public cvtest::BaseTest
{
public:
    CV_CalibrateHandEyeTest(bool eyeToHand) : eyeToHandConfig(eyeToHand) {
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
        if (eyeToHandConfig)
        {
            eps_tvec_noise[CALIB_HAND_EYE_ANDREFF] = 7.0e-2;
        }
        else
        {
            eps_tvec_noise[CALIB_HAND_EYE_ANDREFF] = 5.0e-2;
        }
        eps_tvec_noise[CALIB_HAND_EYE_DANIILIDIS] = 5.0e-2;
    }
protected:
    virtual void run(int);

    bool eyeToHandConfig;
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
        if (eyeToHandConfig)
        {
            {
                //No noise
                std::vector<Mat> R_base2gripper, t_base2gripper;
                std::vector<Mat> R_target2cam, t_target2cam;
                Mat R_cam2base_true, t_cam2base_true;

                const bool noise = false;
                simulateDataEyeToHand(rng, nPoses, R_base2gripper, t_base2gripper, R_target2cam, t_target2cam, noise,
                             R_cam2base_true, t_cam2base_true);

                for (size_t idx = 0; idx < methods.size(); idx++)
                {
                    Mat rvec_cam2base_true;
                    cv::Rodrigues(R_cam2base_true, rvec_cam2base_true);

                    Mat R_cam2base_est, t_cam2base_est;
                    calibrateHandEye(R_base2gripper, t_base2gripper, R_target2cam, t_target2cam, R_cam2base_est, t_cam2base_est, methods[idx]);

                    Mat rvec_cam2base_est;
                    cv::Rodrigues(R_cam2base_est, rvec_cam2base_est);

                    double rvecDiff = cvtest::norm(rvec_cam2base_true, rvec_cam2base_est, NORM_L2);
                    double tvecDiff = cvtest::norm(t_cam2base_true, t_cam2base_est, NORM_L2);

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
                //Gaussian noise on transformations between calibration target frame and camera frame and between robot base and gripper frames
                std::vector<Mat> R_base2gripper, t_base2gripper;
                std::vector<Mat> R_target2cam, t_target2cam;
                Mat R_cam2base_true, t_cam2base_true;

                const bool noise = true;
                simulateDataEyeToHand(rng, nPoses, R_base2gripper, t_base2gripper, R_target2cam, t_target2cam, noise,
                                      R_cam2base_true, t_cam2base_true);

                for (size_t idx = 0; idx < methods.size(); idx++)
                {
                    Mat rvec_cam2base_true;
                    cv::Rodrigues(R_cam2base_true, rvec_cam2base_true);

                    Mat R_cam2base_est, t_cam2base_est;
                    calibrateHandEye(R_base2gripper, t_base2gripper, R_target2cam, t_target2cam, R_cam2base_est, t_cam2base_est, methods[idx]);

                    Mat rvec_cam2base_est;
                    cv::Rodrigues(R_cam2base_est, rvec_cam2base_est);

                    double rvecDiff = cvtest::norm(rvec_cam2base_true, rvec_cam2base_est, NORM_L2);
                    double tvecDiff = cvtest::norm(t_cam2base_true, t_cam2base_est, NORM_L2);

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
        else
        {
            {
                //No noise
                std::vector<Mat> R_gripper2base, t_gripper2base;
                std::vector<Mat> R_target2cam, t_target2cam;
                Mat R_cam2gripper_true, t_cam2gripper_true;

                const bool noise = false;
                simulateDataEyeInHand(rng, nPoses, R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, noise,
                                      R_cam2gripper_true, t_cam2gripper_true);

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
                simulateDataEyeInHand(rng, nPoses, R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, noise,
                                      R_cam2gripper_true, t_cam2gripper_true);

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
    }

    for (size_t idx = 0; idx < methods.size(); idx++)
    {
        std::cout << std::endl;
        printStats(getMethodName(methods[idx]), vec_rvec_diff[idx], vec_tvec_diff[idx]);
        printStats("(noise) " + getMethodName(methods[idx]), vec_rvec_diff_noise[idx], vec_tvec_diff_noise[idx]);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Calib3d_CalibrateHandEye, regression_eye_in_hand)
{
    //Eye-in-Hand configuration (camera mounted on the robot end-effector observing a static calibration pattern)
    const bool eyeToHand = false;
    CV_CalibrateHandEyeTest test(eyeToHand);
    test.safe_run();
}

TEST(Calib3d_CalibrateHandEye, regression_eye_to_hand)
{
    //Eye-to-Hand configuration (static camera observing a calibration pattern mounted on the robot end-effector)
    const bool eyeToHand = true;
    CV_CalibrateHandEyeTest test(eyeToHand);
    test.safe_run();
}

TEST(Calib3d_CalibrateHandEye, regression_17986)
{
    std::vector<Mat> R_target2cam, t_target2cam;
    // Dataset contains transformation from base to gripper frame since it contains data for AX = ZB calibration problem
    std::vector<Mat> R_base2gripper, t_base2gripper;
    loadDataset(R_target2cam, t_target2cam, R_base2gripper, t_base2gripper);

    std::vector<HandEyeCalibrationMethod> methods = {CALIB_HAND_EYE_TSAI,
                                                     CALIB_HAND_EYE_PARK,
                                                     CALIB_HAND_EYE_HORAUD,
                                                     CALIB_HAND_EYE_ANDREFF,
                                                     CALIB_HAND_EYE_DANIILIDIS};

    for (auto method : methods) {
        SCOPED_TRACE(cv::format("method=%s", getMethodName(method).c_str()));

        Matx33d R_cam2base_est;
        Matx31d t_cam2base_est;
        calibrateHandEye(R_base2gripper, t_base2gripper, R_target2cam, t_target2cam, R_cam2base_est, t_cam2base_est, method);

        EXPECT_TRUE(checkRange(R_cam2base_est));
        EXPECT_TRUE(checkRange(t_cam2base_est));
    }
}

TEST(Calib3d_CalibrateRobotWorldHandEye, regression)
{
    std::vector<Mat> R_world2cam, t_worldt2cam;
    std::vector<Mat> R_base2gripper, t_base2gripper;
    loadDataset(R_world2cam, t_worldt2cam, R_base2gripper, t_base2gripper);

    std::vector<Mat> rvec_R_world2cam;
    rvec_R_world2cam.reserve(R_world2cam.size());
    for (size_t i = 0; i < R_world2cam.size(); i++)
    {
        Mat rvec;
        cv::Rodrigues(R_world2cam[i], rvec);
        rvec_R_world2cam.push_back(rvec);
    }

    std::vector<RobotWorldHandEyeCalibrationMethod> methods = {CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
                                                               CALIB_ROBOT_WORLD_HAND_EYE_LI};

    Matx33d wRb, cRg;
    Matx31d wtb, ctg;
    loadResults(wRb, wtb, cRg, ctg);

    for (auto method : methods) {
        SCOPED_TRACE(cv::format("method=%s", getMethodName(method).c_str()));

        Matx33d wRb_est, cRg_est;
        Matx31d wtb_est, ctg_est;
        calibrateRobotWorldHandEye(rvec_R_world2cam, t_worldt2cam, R_base2gripper, t_base2gripper,
                                   wRb_est, wtb_est, cRg_est, ctg_est, method);

        EXPECT_TRUE(checkRange(wRb_est));
        EXPECT_TRUE(checkRange(wtb_est));
        EXPECT_TRUE(checkRange(cRg_est));
        EXPECT_TRUE(checkRange(ctg_est));

        //Arbitrary thresholds
        const double rotation_threshold = 1.0; //1deg
        const double translation_threshold = 50.0; //5cm

        //X
        //rotation error
        Matx33d wRw_est = wRb * wRb_est.t();
        Matx31d rvec_wRw_est;
        cv::Rodrigues(wRw_est, rvec_wRw_est);
        double X_rotation_error = cv::norm(rvec_wRw_est)*180/CV_PI;
        //translation error
        double X_t_error = cv::norm(wtb_est - wtb);
        SCOPED_TRACE(cv::format("X rotation error=%f", X_rotation_error));
        SCOPED_TRACE(cv::format("X translation error=%f", X_t_error));
        EXPECT_TRUE(X_rotation_error < rotation_threshold);
        EXPECT_TRUE(X_t_error < translation_threshold);

        //Z
        //rotation error
        Matx33d cRc_est = cRg * cRg_est.t();
        Matx31d rvec_cMc_est;
        cv::Rodrigues(cRc_est, rvec_cMc_est);
        double Z_rotation_error = cv::norm(rvec_cMc_est)*180/CV_PI;
        //translation error
        double Z_t_error = cv::norm(ctg_est - ctg);
        SCOPED_TRACE(cv::format("Z rotation error=%f", Z_rotation_error));
        SCOPED_TRACE(cv::format("Z translation error=%f", Z_t_error));
        EXPECT_TRUE(Z_rotation_error < rotation_threshold);
        EXPECT_TRUE(Z_t_error < translation_threshold);
    }
}

}} // namespace
