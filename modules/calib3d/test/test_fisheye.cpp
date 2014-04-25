#include "test_precomp.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <climits>
#include <algorithm>

#include <opencv2/ts/gpu_test.hpp>
#include <opencv2/ts/ts_perf.hpp>
#include <opencv2/ts/ts.hpp>

#define DEF_PARAM_TEST(name, ...)  typedef ::perf::TestBaseWithParam< std::tr1::tuple< __VA_ARGS__ > > name
#define PARAM_TEST_CASE(name, ...) struct name : testing::TestWithParam< std::tr1::tuple< __VA_ARGS__ > >


namespace FishEye
{
    const static cv::Size imageSize(1280, 800);

    const static cv::Matx33d K(558.478087865323,               0, 620.458515360843,
                                  0, 560.506767351568, 381.939424848348,
                                  0,               0,                1);

    const static cv::Vec4d D(-0.0014613319981768, -0.00329861110580401, 0.00605760088590183, -0.00374209380722371);

    const static cv::Matx33d R ( 9.9756700084424932e-01, 6.9698277640183867e-02, 1.4929569991321144e-03,
                                -6.9711825162322980e-02, 9.9748249845531767e-01, 1.2997180766418455e-02,
                                -5.8331736398316541e-04,-1.3069635393884985e-02, 9.9991441852366736e-01);

    const static cv::Vec3d T(-9.9217369356044638e-02, 3.1741831972356663e-03, 1.8551007952921010e-04);
}

namespace{
std::string combine(const std::string& _item1, const std::string& _item2)
{
    std::string item1 = _item1, item2 = _item2;
    std::replace(item1.begin(), item1.end(), '\\', '/');
    std::replace(item2.begin(), item2.end(), '\\', '/');

    if (item1.empty())
        return item2;

    if (item2.empty())
        return item1;

    char last = item1[item1.size()-1];
    return item1 + (last != '/' ? "/" : "") + item2;
}

std::string combine_format(const std::string& item1, const std::string& item2, ...)
{
    std::string fmt = combine(item1, item2);
    char buffer[1 << 16];
    va_list args;
    va_start( args, item2 );
    vsprintf( buffer, fmt.c_str(), args );
    va_end( args );
    return std::string(buffer);
}

void readPoins(std::vector<std::vector<cv::Point3d> >& objectPoints,
               std::vector<std::vector<cv::Point2d> >& imagePoints,
               const std::string& path, const int n_images, const int n_points)
{
    objectPoints.resize(n_images);
    imagePoints.resize(n_images);

    std::vector<cv::Point2d> image(n_points);
    std::vector<cv::Point3d> object(n_points);

    std::ifstream ipStream;
    std::ifstream opStream;

    for (int image_idx = 0; image_idx < n_images; image_idx++)
    {
        std::stringstream ss;
        ss << image_idx;
        std::string idxStr = ss.str();

        ipStream.open(combine(path, std::string(std::string("x_") + idxStr + std::string(".csv"))).c_str(), std::ifstream::in);
        opStream.open(combine(path, std::string(std::string("X_") + idxStr + std::string(".csv"))).c_str(), std::ifstream::in);
        CV_Assert(ipStream.is_open() && opStream.is_open());

        for (int point_idx = 0; point_idx < n_points; point_idx++)
        {
            double x, y, z;
            char delim;
            ipStream >> x >> delim >> y;
            image[point_idx] = cv::Point2d(x, y);
            opStream >> x >> delim >> y >> delim >> z;
            object[point_idx] = cv::Point3d(x, y, z);
        }
        ipStream.close();
        opStream.close();

        imagePoints[image_idx] = image;
        objectPoints[image_idx] = object;
    }
}

void readExtrinsics(const std::string& file, cv::OutputArray _R, cv::OutputArray _T, cv::OutputArray _R1, cv::OutputArray _R2,
                    cv::OutputArray _P1, cv::OutputArray _P2, cv::OutputArray _Q)
{
    cv::FileStorage fs(file, cv::FileStorage::READ);
    CV_Assert(fs.isOpened());

    cv::Mat R, T, R1, R2, P1, P2, Q;
    fs["R"] >> R;   fs["T"] >> T;   fs["R1"] >> R1;   fs["R2"] >> R2;   fs["P1"] >> P1;   fs["P2"] >> P2;   fs["Q"] >> Q;
    if (_R.needed()) R.copyTo(_R); if(_T.needed()) T.copyTo(_T); if (_R1.needed()) R1.copyTo(_R1); if (_R2.needed()) R2.copyTo(_R2);
    if(_P1.needed()) P1.copyTo(_P1); if(_P2.needed()) P2.copyTo(_P2); if(_Q.needed()) Q.copyTo(_Q);
}

cv::Mat mergeRectification(const cv::Mat& l, const cv::Mat& r, double scale)
{
    CV_Assert(l.type() == r.type() && l.size() == r.size());
    cv::Mat merged(l.rows, l.cols * 2, l.type());
    cv::Mat lpart = merged.colRange(0, l.cols);
    cv::Mat rpart = merged.colRange(l.cols, merged.cols);
    l.copyTo(lpart);
    r.copyTo(rpart);

    for(int i = 0; i < l.rows; i+=20)
        cv::line(merged, cv::Point(0, i), cv::Point(merged.cols, i), CV_RGB(0, 255, 0));

    return merged;
}

}



/// Change this parameter via CMake: cmake -DDATASETS_REPOSITORY_FOLDER=<path>
//const static std::string datasets_repository_path = "DATASETS_REPOSITORY_FOLDER";
const static std::string datasets_repository_path = "/home/krylov/data";

TEST(FisheyeTest, projectPoints)
{
    double cols = FishEye::imageSize.width,
           rows = FishEye::imageSize.height;

    const int N = 20;
    cv::Mat distorted0(1, N*N, CV_64FC2), undist1, undist2, distorted1, distorted2;
    undist2.create(distorted0.size(), CV_MAKETYPE(distorted0.depth(), 3));
    cv::Vec2d* pts = distorted0.ptr<cv::Vec2d>();

    cv::Vec2d c(FishEye::K(0, 2), FishEye::K(1, 2));
    for(int y = 0, k = 0; y < N; ++y)
        for(int x = 0; x < N; ++x)
        {
            cv::Vec2d point(x*cols/(N-1.f), y*rows/(N-1.f));
            pts[k++] = (point - c) * 0.85 + c;
        }

    cv::Fisheye::undistortPoints(distorted0, undist1, FishEye::K, FishEye::D);

    cv::Vec2d* u1 = undist1.ptr<cv::Vec2d>();
    cv::Vec3d* u2 = undist2.ptr<cv::Vec3d>();
    for(int i = 0; i  < (int)distorted0.total(); ++i)
        u2[i] = cv::Vec3d(u1[i][0], u1[i][1], 1.0);

    cv::Fisheye::distortPoints(undist1, distorted1, FishEye::K, FishEye::D);
    cv::Fisheye::projectPoints(undist2, distorted2, cv::Vec3d::all(0), cv::Vec3d::all(0), FishEye::K, FishEye::D);

    EXPECT_MAT_NEAR(distorted0, distorted1, 1e-5);
    EXPECT_MAT_NEAR(distorted0, distorted2, 1e-5);
}

TEST(FisheyeTest, undistortImage)
{
    cv::Matx33d K = FishEye::K;
    cv::Mat D = cv::Mat(FishEye::D);
    std::string file = combine(datasets_repository_path, "image000001.png");

    cv::Matx33d newK = K;
    cv::Mat distorted = cv::imread(file), undistorted;

    {
        newK(0, 0) = 100;
        newK(1, 1) = 100;
        cv::Fisheye::undistortImage(distorted, undistorted, K, D, newK);
        cv::Mat correct = cv::imread(combine(datasets_repository_path, "test_undistortImage/new_f_100.png"));
        if (correct.empty())
            CV_Assert(cv::imwrite(combine(datasets_repository_path, "test_undistortImage/new_f_100.png"), undistorted));
        else
            EXPECT_MAT_NEAR(correct, undistorted, 1e-15);
    }
    {
        double balance = 1.0;
        cv::Fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, distorted.size(), cv::noArray(), newK, balance);
        cv::Fisheye::undistortImage(distorted, undistorted, K, D, newK);
        cv::Mat correct = cv::imread(combine(datasets_repository_path, "test_undistortImage/balance_1.0.png"));
        if (correct.empty())
            CV_Assert(cv::imwrite(combine(datasets_repository_path, "test_undistortImage/balance_1.0.png"), undistorted));
        else
            EXPECT_MAT_NEAR(correct, undistorted, 1e-15);
    }

    {
        double balance = 0.0;
        cv::Fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, distorted.size(), cv::noArray(), newK, balance);
        cv::Fisheye::undistortImage(distorted, undistorted, K, D, newK);
        cv::Mat correct = cv::imread(combine(datasets_repository_path, "test_undistortImage/balance_0.0.png"));
        if (correct.empty())
            CV_Assert(cv::imwrite(combine(datasets_repository_path, "test_undistortImage/balance_0.0.png"), undistorted));
        else
            EXPECT_MAT_NEAR(correct, undistorted, 1e-15);
    }

    cv::waitKey();
}

TEST(FisheyeTest, jacobians)
{
    int n = 10;
    cv::Mat X(1, n, CV_32FC4);
    cv::Mat om(3, 1, CV_64F), T(3, 1, CV_64F);
    cv::Mat f(2, 1, CV_64F), c(2, 1, CV_64F);
    cv::Mat k(4, 1, CV_64F);
    double alpha;

    cv::RNG& r = cv::theRNG();

    r.fill(X, cv::RNG::NORMAL, 0, 1);
    X = cv::abs(X) * 10;

    r.fill(om, cv::RNG::NORMAL, 0, 1);
    om = cv::abs(om);

    r.fill(T, cv::RNG::NORMAL, 0, 1);
    T = cv::abs(T); T.at<double>(2) = 4; T *= 10;

    r.fill(f, cv::RNG::NORMAL, 0, 1);
    f = cv::abs(f) * 1000;

    r.fill(c, cv::RNG::NORMAL, 0, 1);
    c = cv::abs(c) * 1000;

    r.fill(k, cv::RNG::NORMAL, 0, 1);
    k*= 0.5;

    alpha = 0.01*r.gaussian(1);


    CV_Assert(!"/////////");
}

TEST(FisheyeTest, Calibration)
{
    const int n_images = 34;
    const int n_points = 48;

    cv::Size imageSize = cv::Size(1280, 800);
    std::vector<std::vector<cv::Point2d> > imagePoints;
    std::vector<std::vector<cv::Point3d> > objectPoints;

    readPoins(objectPoints, imagePoints, combine(datasets_repository_path, "calib-3_stereo_from_JY/left"), n_images, n_points);

    int flag = 0;
    flag |= cv::Fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::Fisheye::CALIB_CHECK_COND;
    flag |= cv::Fisheye::CALIB_FIX_SKEW;

    cv::Matx33d K;
    cv::Vec4d D;

    cv::Fisheye::calibrate(objectPoints, imagePoints, imageSize, K, D,
                           cv::noArray(), cv::noArray(), flag, cv::TermCriteria(3, 20, 1e-6));

    EXPECT_MAT_NEAR(K, FishEye::K, 1e-11);
    EXPECT_MAT_NEAR(D, FishEye::D, 1e-12);
}

TEST(FisheyeTest, Homography)
{
    const int n_images = 1;
    const int n_points = 48;

    cv::Size imageSize = cv::Size(1280, 800);
    std::vector<std::vector<cv::Point2d> > imagePoints;
    std::vector<std::vector<cv::Point3d> > objectPoints;

    readPoins(objectPoints, imagePoints, combine(datasets_repository_path, "calib-3_stereo_from_JY/left"), n_images, n_points);
    cv::internal::IntrinsicParams param;
    param.Init(cv::Vec2d(cv::max(imageSize.width, imageSize.height) / CV_PI, cv::max(imageSize.width, imageSize.height) / CV_PI),
               cv::Vec2d(imageSize.width  / 2.0 - 0.5, imageSize.height / 2.0 - 0.5));

    cv::Mat _imagePoints (imagePoints[0]);
    cv::Mat _objectPoints(objectPoints[0]);

    cv::Mat imagePointsNormalized = NormalizePixels(_imagePoints, param).reshape(1).t();
    _objectPoints = _objectPoints.reshape(1).t();
    cv::Mat objectPointsMean, covObjectPoints;

    int Np = imagePointsNormalized.cols;
    cv::calcCovarMatrix(_objectPoints, covObjectPoints, objectPointsMean, CV_COVAR_NORMAL | CV_COVAR_COLS);
    cv::SVD svd(covObjectPoints);
    cv::Mat R(svd.vt);

    if (cv::norm(R(cv::Rect(2, 0, 1, 2))) < 1e-6)
        R = cv::Mat::eye(3,3, CV_64FC1);
    if (cv::determinant(R) < 0)
        R = -R;

    cv::Mat T = -R * objectPointsMean;
    cv::Mat X_new = R * _objectPoints + T * cv::Mat::ones(1, Np, CV_64FC1);
    cv::Mat H = cv::internal::ComputeHomography(imagePointsNormalized, X_new.rowRange(0, 2));

    cv::Mat M = cv::Mat::ones(3, X_new.cols, CV_64FC1);
    X_new.rowRange(0, 2).copyTo(M.rowRange(0, 2));
    cv::Mat mrep = H * M;

    cv::divide(mrep, cv::Mat::ones(3,1, CV_64FC1) * mrep.row(2).clone(), mrep);

    cv::Mat merr = (mrep.rowRange(0, 2) - imagePointsNormalized).t();

    cv::Vec2d std_err;
    cv::meanStdDev(merr.reshape(2), cv::noArray(), std_err);
    std_err *= sqrt((double)merr.reshape(2).total() / (merr.reshape(2).total() - 1));

    cv::Vec2d correct_std_err(0.00516740156010384, 0.00644205331553901);
    EXPECT_MAT_NEAR(std_err, correct_std_err, 1e-16);
}

TEST(TestFisheye, EtimateUncertainties)
{
    const int n_images = 34;
    const int n_points = 48;

    cv::Size imageSize = cv::Size(1280, 800);
    std::vector<std::vector<cv::Point2d> > imagePoints;
    std::vector<std::vector<cv::Point3d> > objectPoints;

    readPoins(objectPoints, imagePoints, combine(datasets_repository_path, "calib-3_stereo_from_JY/left"), n_images, n_points);

    int flag = 0;
    flag |= cv::Fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::Fisheye::CALIB_CHECK_COND;
    flag |= cv::Fisheye::CALIB_FIX_SKEW;

    cv::Matx33d K;
    cv::Vec4d D;
    std::vector<cv::Vec3d> rvec;
    std::vector<cv::Vec3d> tvec;

    cv::Fisheye::calibrate(objectPoints, imagePoints, imageSize, K, D,
                           cv::noArray(), cv::noArray(), flag, cv::TermCriteria(3, 20, 1e-6));

    cv::internal::IntrinsicParams param, errors;
    cv::Vec2d err_std;
    double thresh_cond = 1e6;
    int check_cond = 1;
    param.Init(cv::Vec2d(K(0,0), K(1,1)), cv::Vec2d(K(0,2), K(1, 2)), D);
    param.isEstimate = std::vector<int>(9, 1);
    param.isEstimate[4] = 0;

    errors.isEstimate = param.isEstimate;

    double rms;

    cv::internal::EstimateUncertainties(objectPoints, imagePoints, param,  rvec, tvec,
                                        errors, err_std, thresh_cond, check_cond, rms);

    EXPECT_MAT_NEAR(errors.f, cv::Vec2d(1.29837104202046,  1.31565641071524), 1e-14);
    EXPECT_MAT_NEAR(errors.c, cv::Vec2d(0.890439368129246, 0.816096854937896), 1e-15);
    EXPECT_MAT_NEAR(errors.k, cv::Vec4d(0.00516248605191506, 0.0168181467500934, 0.0213118690274604, 0.00916010877545648), 1e-15);
    EXPECT_MAT_NEAR(err_std, cv::Vec2d(0.187475975266883, 0.185678953263995), 1e-15);
    CV_Assert(abs(rms - 0.263782587133546) < 1e-15);
    CV_Assert(errors.alpha == 0);
  }

TEST(FisheyeTest, rectify)
{
    const std::string folder =combine(datasets_repository_path, "calib-3_stereo_from_JY");

    cv::Size calibration_size = FishEye::imageSize, requested_size = calibration_size;
    cv::Matx33d K1 = FishEye::K, K2 = K1;
    cv::Mat D1 = cv::Mat(FishEye::D), D2 = D1;

    cv::Vec3d T = FishEye::T;
    cv::Matx33d R = FishEye::R;

    double balance = 0.0, fov_scale = 1.1;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Fisheye::stereoRectify(K1, D1, K2, D2, calibration_size, R, T, R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, requested_size, balance, fov_scale);

    cv::Mat lmapx, lmapy, rmapx, rmapy;
    //rewrite for fisheye
    cv::Fisheye::initUndistortRectifyMap(K1, D1, R1, P1, requested_size, CV_32F, lmapx, lmapy);
    cv::Fisheye::initUndistortRectifyMap(K2, D2, R2, P2, requested_size, CV_32F, rmapx, rmapy);

    cv::Mat l, r, lundist, rundist;
    cv::VideoCapture lcap(combine(folder, "left/stereo_pair_%03d.jpg")),
                     rcap(combine(folder, "right/stereo_pair_%03d.jpg"));

    for(int i = 0;; ++i)
    {
        lcap >> l; rcap >> r;
        if (l.empty() || r.empty())
            break;

        int ndisp = 128;
        cv::rectangle(l, cv::Rect(255,       0, 829,       l.rows-1), CV_RGB(255, 0, 0));
        cv::rectangle(r, cv::Rect(255,       0, 829,       l.rows-1), CV_RGB(255, 0, 0));
        cv::rectangle(r, cv::Rect(255-ndisp, 0, 829+ndisp ,l.rows-1), CV_RGB(255, 0, 0));
        cv::remap(l, lundist, lmapx, lmapy, cv::INTER_LINEAR);
        cv::remap(r, rundist, rmapx, rmapy, cv::INTER_LINEAR);

        cv::Mat rectification = mergeRectification(lundist, rundist, 0.75);

        cv::Mat correct = cv::imread(combine_format(folder, "test_rectify/rectification_AB_%03d.png", i));
        if (correct.empty())
            cv::imwrite(combine_format(folder, "test_rectify/rectification_AB_%03d.png", i), rectification);
        else
            EXPECT_MAT_NEAR(correct, rectification, 1e-15);
    }
}





