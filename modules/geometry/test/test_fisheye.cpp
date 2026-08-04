// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/ts/cuda_test.hpp> // EXPECT_MAT_NEAR
#include "opencv2/videoio.hpp"

namespace opencv_test { namespace {

class fisheyeTest : public ::testing::Test {

protected:
    const static cv::Size imageSize;
    const static cv::Matx33d K;
    const static cv::Vec4d D;
    std::string datasets_repository_path;

    virtual void SetUp() {
        datasets_repository_path = combine(cvtest::TS::ptr()->get_data_path(), "cv/cameracalibration/fisheye");
    }

protected:
    std::string combine(const std::string& _item1, const std::string& _item2);
};

const cv::Size fisheyeTest::imageSize(1280, 800);

const cv::Matx33d fisheyeTest::K(558.478087865323,               0, 620.458515360843,
                              0, 560.506767351568, 381.939424848348,
                              0,               0,                1);

const cv::Vec4d fisheyeTest::D(-0.0014613319981768, -0.00329861110580401, 0.00605760088590183, -0.00374209380722371);

std::string fisheyeTest::combine(const std::string& _item1, const std::string& _item2)
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

TEST_F(fisheyeTest, projectPoints)
{
    double cols = this->imageSize.width,
           rows = this->imageSize.height;

    const int N = 20;
    cv::Mat distorted0(1, N*N, CV_64FC2), undist1, undist2, distorted1, distorted2;
    undist2.create(distorted0.size(), CV_MAKETYPE(distorted0.depth(), 3));
    cv::Vec2d* pts = distorted0.ptr<cv::Vec2d>();

    cv::Vec2d c(this->K(0, 2), this->K(1, 2));
    for(int y = 0, k = 0; y < N; ++y)
        for(int x = 0; x < N; ++x)
        {
            cv::Vec2d point(x*cols/(N-1.f), y*rows/(N-1.f));
            pts[k++] = (point - c) * 0.85 + c;
        }

    cv::fisheye::undistortPoints(distorted0, undist1, this->K, this->D);

    cv::Vec2d* u1 = undist1.ptr<cv::Vec2d>();
    cv::Vec3d* u2 = undist2.ptr<cv::Vec3d>();
    for(int i = 0; i  < (int)distorted0.total(); ++i)
        u2[i] = cv::Vec3d(u1[i][0], u1[i][1], 1.0);

    cv::fisheye::distortPoints(undist1, distorted1, this->K, this->D);
    cv::fisheye::projectPoints(undist2, distorted2, cv::Vec3d::all(0), cv::Vec3d::all(0), this->K, this->D);

    EXPECT_MAT_NEAR(distorted0, distorted1, 1e-10);
    EXPECT_MAT_NEAR(distorted0, distorted2, 1e-10);
}

TEST_F(fisheyeTest, distortUndistortPoints)
{
    int width = imageSize.width;
    int height = imageSize.height;

    /* Create test points */
    cv::Mat principalPoints = (cv::Mat_<double>(5, 2) << K(0, 2), K(1, 2), // (cx, cy)
                                                                    /* Image corners */
                                                                    0, 0,
                                                                    0, height,
                                                                    width, 0,
                                                                    width, height
                                                                    );

    /* Random points inside image */
    cv::Mat xy[2] = {};
    xy[0].create(100, 1, CV_64F);
    theRNG().fill(xy[0], cv::RNG::UNIFORM, 0, width); // x
    xy[1].create(100, 1, CV_64F);
    theRNG().fill(xy[1], cv::RNG::UNIFORM, 0, height); // y

    cv::Mat randomPoints;
    merge(xy, 2, randomPoints);

    cv::Mat points0;
    cv::vconcat(principalPoints.reshape(2), randomPoints, points0);

    /* Test with random D set */
    for (size_t i = 0; i < 10; ++i) {
        cv::Mat distortion(1, 4, CV_64F);
        theRNG().fill(distortion, cv::RNG::UNIFORM, -0.00001, 0.00001);

        /* Distort -> Undistort */
        cv::Mat distortedPoints;
        cv::fisheye::distortPoints(points0, distortedPoints, K, distortion);
        cv::Mat undistortedPoints;
        cv::fisheye::undistortPoints(distortedPoints, undistortedPoints, K, distortion);

        EXPECT_MAT_NEAR(points0, undistortedPoints, 1e-8);

        /* Undistort -> Distort */
        cv::fisheye::undistortPoints(points0, undistortedPoints, K, distortion);
        cv::fisheye::distortPoints(undistortedPoints, distortedPoints, K, distortion);

        EXPECT_MAT_NEAR(points0, distortedPoints, 1e-8);
    }
}

TEST_F(fisheyeTest, distortUndistortPointsNewCameraFixed)
{
    int width = imageSize.width;
    int height = imageSize.height;

    /* Random points inside image */
    cv::Mat xy[2] = {};
    xy[0].create(100, 1, CV_64F);
    theRNG().fill(xy[0], cv::RNG::UNIFORM, 0, width); // x
    xy[1].create(100, 1, CV_64F);
    theRNG().fill(xy[1], cv::RNG::UNIFORM, 0, height); // y

    cv::Mat randomPoints;
    merge(xy, 2, randomPoints);

    cv::Mat points0 = randomPoints;
    cv::Mat Reye = cv::Mat::eye(3, 3, CV_64FC1);

    cv::Mat Knew;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, imageSize, Reye,  Knew);

    /* Distort -> Undistort */
    cv::Mat distortedPoints;
    cv::fisheye::distortPoints(points0, distortedPoints, Knew, K, D);
    cv::Mat undistortedPoints;
    cv::fisheye::undistortPoints(distortedPoints, undistortedPoints, K, D, Reye, Knew);

    EXPECT_MAT_NEAR(points0, undistortedPoints, 1e-8);

    /* Undistort -> Distort */
    cv::fisheye::undistortPoints(points0, undistortedPoints, K, D, Reye, Knew);
    cv::fisheye::distortPoints(undistortedPoints, distortedPoints, Knew, K, D);

    EXPECT_MAT_NEAR(points0, distortedPoints, 1e-8);
}

TEST_F(fisheyeTest, distortUndistortPointsNewCameraRandom)
{
    int width = imageSize.width;
    int height = imageSize.height;

    /* Create test points */
    std::vector<cv::Point2d> points0Vector;
    cv::Mat principalPoints = (cv::Mat_<double>(5, 2) << K(0, 2), K(1, 2), // (cx, cy)
                                                                    /* Image corners */
                                                                    0, 0,
                                                                    0, height,
                                                                    width, 0,
                                                                    width, height
                                                                    );

    /* Random points inside image */
    cv::Mat xy[2] = {};
    xy[0].create(100, 1, CV_64F);
    theRNG().fill(xy[0], cv::RNG::UNIFORM, 0, width); // x
    xy[1].create(100, 1, CV_64F);
    theRNG().fill(xy[1], cv::RNG::UNIFORM, 0, height); // y

    cv::Mat randomPoints;
    merge(xy, 2, randomPoints);

    cv::Mat points0;
    cv::Mat Reye = cv::Mat::eye(3, 3, CV_64FC1);
    cv::vconcat(principalPoints.reshape(2), randomPoints, points0);

    /* Test with random D set */
    for (size_t i = 0; i < 10; ++i) {
        cv::Mat distortion(1, 4, CV_64F);
        theRNG().fill(distortion, cv::RNG::UNIFORM, -0.001, 0.001);

        cv::Mat Knew;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, distortion, imageSize, Reye,  Knew);

        /* Distort -> Undistort */
        cv::Mat distortedPoints;
        cv::fisheye::distortPoints(points0, distortedPoints, Knew, K, distortion);
        cv::Mat undistortedPoints;
        cv::fisheye::undistortPoints(distortedPoints, undistortedPoints, K, distortion, Reye, Knew);

        EXPECT_MAT_NEAR(points0, undistortedPoints, 1e-8);

        /* Undistort -> Distort */
        cv::fisheye::undistortPoints(points0, undistortedPoints, K, distortion, Reye, Knew);
        cv::fisheye::distortPoints(undistortedPoints, distortedPoints, Knew, K, distortion);

        EXPECT_MAT_NEAR(points0, distortedPoints, 1e-8);
    }
}

// Reference implementation of the fisheye undistortion solver, kept scalar and independent of the
// library so that the vectorized path can be checked against it. Mirrors the scalar loop in
// modules/geometry/src/fisheye.cpp for the case without rectification/new camera matrix.
static void undistortPointsScalarReference(const cv::Mat& distorted, cv::Mat& undistorted,
                                           const cv::Matx33d& K, const cv::Vec4d& D, int maxCount)
{
    undistorted.create(distorted.size(), distorted.type());

    const double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
    const bool isFloat = distorted.depth() == CV_32F;
    const size_t n = distorted.total();

    for (size_t i = 0; i < n; i++)
    {
        cv::Vec2d pi = isFloat ? (cv::Vec2d)distorted.ptr<cv::Vec2f>()[i]
                               : distorted.ptr<cv::Vec2d>()[i];
        cv::Vec2d pw((pi[0] - cx)/fx, (pi[1] - cy)/fy);

        double theta_d = std::sqrt(pw[0]*pw[0] + pw[1]*pw[1]);
        theta_d = std::min(std::max(-CV_PI/2., theta_d), CV_PI/2.);

        double theta = theta_d;
        for (int j = 0; j < maxCount; j++)
        {
            double theta2 = theta*theta, theta4 = theta2*theta2;
            double theta6 = theta4*theta2, theta8 = theta6*theta2;
            double k0_theta2 = D[0]*theta2, k1_theta4 = D[1]*theta4;
            double k2_theta6 = D[2]*theta6, k3_theta8 = D[3]*theta8;
            double theta_fix = (theta*(1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                               (1 + 3*k0_theta2 + 5*k1_theta4 + 7*k2_theta6 + 9*k3_theta8);
            theta = theta - theta_fix;
        }
        double scale = std::tan(theta) / theta_d;

        bool theta_flipped = ((theta_d < 0 && theta > 0) || (theta_d > 0 && theta < 0));
        cv::Vec2d fi = theta_flipped ? cv::Vec2d(-1000000.0, -1000000.0) : pw*scale;

        if (isFloat)
            undistorted.ptr<cv::Vec2f>()[i] = cv::Vec2f(fi);
        else
            undistorted.ptr<cv::Vec2d>()[i] = fi;
    }
}

// Largest elementwise relative difference, |a - b| / max(|b|, 1). Points far outside the valid
// field of view are ill-conditioned: theta approaches pi/2, where tan() amplifies the last bits of
// theta, so the two implementations are compared in relative rather than absolute terms.
static double maxRelativeDiff(const cv::Mat& a, const cv::Mat& b)
{
    CV_Assert(a.size() == b.size() && a.type() == b.type());
    cv::Mat a64, b64;
    a.reshape(1).convertTo(a64, CV_64F);
    b.reshape(1).convertTo(b64, CV_64F);

    double maxRel = 0.0;
    for (int i = 0; i < a64.rows; i++)
    {
        for (int j = 0; j < a64.cols; j++)
        {
            const double x = a64.at<double>(i, j), y = b64.at<double>(i, j);
            // The principal point yields tan(0)/0, so NaN must appear in the same places in both.
            if (cvIsNaN(x) || cvIsNaN(y))
            {
                if (cvIsNaN(x) != cvIsNaN(y))
                    return std::numeric_limits<double>::infinity();
                continue;
            }
            if (cvIsInf(x) || cvIsInf(y))
            {
                if (x != y)
                    return std::numeric_limits<double>::infinity();
                continue;
            }
            maxRel = std::max(maxRel, std::abs(x - y) / std::max(std::abs(y), 1.0));
        }
    }
    return maxRel;
}

// The vectorized path of undistortPoints must agree with the scalar solver. It is selected by
// MAX_ITER-only criteria with CV_64FC2 input and no rectification, so the test uses exactly that;
// CV_32FC2 stays on the scalar path and is checked as well. The point counts include values that
// are not multiples of the vector width, so the scalar tail is exercised on 2-, 4- and 8-lane
// builds; the largest case also covers the multi-stripe parallel path.
TEST_F(fisheyeTest, undistortPointsFastPathMatchesScalar)
{
    const int counts[] = { 1, 2, 3, 7, 33, 129, 257, 10001 };
    const int maxCount = 10;
    const cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER, maxCount, 0);

    cv::RNG rng(0x8f3a21c5);

    for (size_t ci = 0; ci < sizeof(counts)/sizeof(counts[0]); ci++)
    {
        const int n = counts[ci];

        cv::Mat points64(n, 1, CV_64FC2);
        cv::Vec2d* pts = points64.ptr<cv::Vec2d>();
        for (int i = 0; i < n; i++)
        {
            pts[i] = cv::Vec2d(rng.uniform(0.0, (double)imageSize.width),
                               rng.uniform(0.0, (double)imageSize.height));
        }
        pts[0] = cv::Vec2d(K(0, 2), K(1, 2)); // principal point

        cv::Mat expected, actual;

        undistortPointsScalarReference(points64, expected, K, D, maxCount);
        cv::fisheye::undistortPoints(points64, actual, K, D, cv::noArray(), cv::noArray(), criteria);
        EXPECT_LT(maxRelativeDiff(actual, expected), 1e-12) << "CV_64FC2, n = " << n;

        cv::Mat points32;
        points64.convertTo(points32, CV_32FC2);
        undistortPointsScalarReference(points32, expected, K, D, maxCount);
        cv::fisheye::undistortPoints(points32, actual, K, D, cv::noArray(), cv::noArray(), criteria);
        EXPECT_LT(maxRelativeDiff(actual, expected), 1e-6) << "CV_32FC2, n = " << n;
    }
}

// Points whose theta flips sign during the iteration are replaced by a sentinel value. Check that
// the vectorized path marks exactly the same points as the scalar solver.
TEST_F(fisheyeTest, undistortPointsSentinelMatchesScalar)
{
    const int maxCount = 10;
    const cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER, maxCount, 0);
    // Strong negative distortion makes the iteration diverge for large radii.
    const cv::Vec4d strongD(-0.5, 0.05, -0.01, 0.002);

    const int n = 129;
    cv::Mat points(n, 1, CV_64FC2);
    cv::Vec2d* pts = points.ptr<cv::Vec2d>();
    for (int i = 0; i < n; i++)
    {
        // Sweep the radius from inside the image out to far beyond the valid field of view.
        const double radius = 40.0*(i + 1);
        const double angle = 0.37*i;
        pts[i] = cv::Vec2d(K(0, 2) + radius*std::cos(angle),
                           K(1, 2) + radius*std::sin(angle));
    }

    cv::Mat expected, actual;
    undistortPointsScalarReference(points, expected, K, strongD, maxCount);
    cv::fisheye::undistortPoints(points, actual, K, strongD, cv::noArray(), cv::noArray(), criteria);

    int sentinels = 0;
    for (int i = 0; i < n; i++)
        if (expected.ptr<cv::Vec2d>()[i][0] == -1000000.0)
            sentinels++;

    // The test is only meaningful if the sentinel branch is actually taken.
    EXPECT_GT(sentinels, 0);
    EXPECT_LT(sentinels, n);

    // Both paths must mark exactly the same points, and agree on the remaining ones.
    for (int i = 0; i < n; i++)
    {
        const bool expectedSentinel = expected.ptr<cv::Vec2d>()[i][0] == -1000000.0;
        const bool actualSentinel = actual.ptr<cv::Vec2d>()[i][0] == -1000000.0;
        ASSERT_EQ(expectedSentinel, actualSentinel) << "point " << i;
    }
    // Strong distortion makes the non-sentinel points numerically ill-conditioned. The sentinel
    // mask above is exact; use a relative check only to catch large deviations in the other points.
    EXPECT_LT(maxRelativeDiff(actual, expected), 1e-8);
}

TEST_F(fisheyeTest, solvePnP)
{
    const int n = 16;

    const cv::Matx33d R_mat ( 9.9756700084424932e-01, 6.9698277640183867e-02, 1.4929569991321144e-03,
                            -6.9711825162322980e-02, 9.9748249845531767e-01, 1.2997180766418455e-02,
                            -5.8331736398316541e-04,-1.3069635393884985e-02, 9.9991441852366736e-01);

    const cv::Vec3d T(-9.9217369356044638e-02, 3.1741831972356663e-03, 1.8551007952921010e-04);

    cv::Mat obj_points(1, n, CV_64FC3);
    theRNG().fill(obj_points, cv::RNG::NORMAL, 2, 1);
    obj_points = cv::abs(obj_points) * 10;

    cv::Mat R;
    cv::Rodrigues(R_mat, R);
    cv::Mat img_points;
    cv::fisheye::projectPoints(obj_points, img_points, R, T, this->K, this->D);

    cv::Mat rvec_pred;
    cv::Mat tvec_pred;
    bool converged = cv::fisheye::solvePnP(obj_points, img_points, this->K, this->D, rvec_pred, tvec_pred);
    EXPECT_MAT_NEAR(R, rvec_pred, 1e-6);
    EXPECT_MAT_NEAR(T, tvec_pred, 1e-6);

    ASSERT_TRUE(converged);
}

TEST_F(fisheyeTest, solvePnPRansac)
{
    const int inliers_n = 16;
    const int outliers_n = 4;
    const bool use_extrinsic_guess = false;
    const int iterations_count = 100;
    const float reprojection_error = 1.0;
    const double confidence = 0.99;

    const cv::Matx33d R_mat ( 9.9756700084424932e-01, 6.9698277640183867e-02, 1.4929569991321144e-03,
                              -6.9711825162322980e-02, 9.9748249845531767e-01, 1.2997180766418455e-02,
                              -5.8331736398316541e-04,-1.3069635393884985e-02, 9.9991441852366736e-01);

    const cv::Vec3d T(-9.9217369356044638e-02, 3.1741831972356663e-03, 1.8551007952921010e-04);

    cv::Mat rvec;
    cv::Rodrigues(R_mat, rvec);

    // inliers
    cv::Mat inlier_obj_points(1, inliers_n, CV_64FC3);
    theRNG().fill(inlier_obj_points, cv::RNG::NORMAL, 2, 1);
    inlier_obj_points = cv::abs(inlier_obj_points) * 10;
    cv::Mat inlier_img_points;
    cv::fisheye::projectPoints(inlier_obj_points, inlier_img_points, rvec, T, this->K, this->D);

    // outliers
    cv::Mat outlier_obj_points(1, outliers_n, CV_64FC3);
    theRNG().fill(outlier_obj_points, cv::RNG::NORMAL, 2, 1);
    outlier_obj_points = cv::abs(outlier_obj_points) * 10;
    cv::Mat outlier_img_points;
    cv::fisheye::projectPoints(outlier_obj_points, outlier_img_points, rvec, (T * 10), this->K, this->D);

    cv::Mat obj_points;
    cv::hconcat(outlier_obj_points, inlier_obj_points, obj_points);

    cv::Mat img_points;
    cv::hconcat(outlier_img_points, inlier_img_points, img_points);

    cv::Mat rvec_pred;
    cv::Mat tvec_pred;
    cv::Mat inliers_pred;

    bool converged = cv::fisheye::solvePnPRansac(obj_points, img_points, this->K, this->D,
                                                 rvec_pred, tvec_pred, use_extrinsic_guess,
                                                 iterations_count, reprojection_error, confidence, inliers_pred);

    EXPECT_MAT_NEAR(rvec, rvec_pred, 1e-5);
    EXPECT_MAT_NEAR(T, tvec_pred, 1e-5);
    EXPECT_EQ(inliers_pred.size[0], inliers_n);
    ASSERT_TRUE(converged);
}

TEST_F(fisheyeTest, undistortImage)
{
    // we use it to reduce patch size for images in testdata
    auto throwAwayHalf = [](Mat img)
    {
        int whalf = img.cols / 2, hhalf = img.rows / 2;
        Rect tl(0, 0, whalf, hhalf), br(whalf, hhalf, whalf, hhalf);
        img(tl) = 0;
        img(br) = 0;
    };

    cv::Matx33d theK = this->K;
    cv::Mat theD = cv::Mat(this->D);
    std::string file = combine(datasets_repository_path, "stereo_pair_014.png");
    cv::Matx33d newK = theK;
    cv::Mat distorted = cv::imread(file), undistorted;
    {
        newK(0, 0) = 100;
        newK(1, 1) = 100;
        cv::fisheye::undistortImage(distorted, undistorted, theK, theD, newK);
        std::string imageFilename = combine(datasets_repository_path, "new_f_100.png");
        cv::Mat correct = cv::imread(imageFilename);
        ASSERT_FALSE(correct.empty()) << "Correct image " << imageFilename.c_str() << " can not be read" << std::endl;

        throwAwayHalf(correct);
        throwAwayHalf(undistorted);

        EXPECT_MAT_NEAR(correct, undistorted, 1e-10);
    }
    {
        double balance = 1.0;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(theK, theD, distorted.size(), cv::noArray(), newK, balance);
        cv::fisheye::undistortImage(distorted, undistorted, theK, theD, newK);
        std::string imageFilename = combine(datasets_repository_path, "balance_1.0.png");
        cv::Mat correct = cv::imread(imageFilename);
        ASSERT_FALSE(correct.empty()) << "Correct image " << imageFilename.c_str() << " can not be read" << std::endl;

        throwAwayHalf(correct);
        throwAwayHalf(undistorted);

        EXPECT_MAT_NEAR(correct, undistorted, 1e-10);
    }

    {
        double balance = 0.0;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(theK, theD, distorted.size(), cv::noArray(), newK, balance);
        cv::fisheye::undistortImage(distorted, undistorted, theK, theD, newK);
        std::string imageFilename = combine(datasets_repository_path, "balance_0.0.png");
        cv::Mat correct = cv::imread(imageFilename);
        ASSERT_FALSE(correct.empty()) << "Correct image " << imageFilename.c_str() << " can not be read" << std::endl;

        throwAwayHalf(correct);
        throwAwayHalf(undistorted);

        EXPECT_MAT_NEAR(correct, undistorted, 1e-10);
    }
}

TEST_F(fisheyeTest, undistortAndDistortImage)
{
    cv::Matx33d K_src = this->K;
    cv::Mat D_src = cv::Mat(this->D);
    std::string file = combine(datasets_repository_path, "/calib-3_stereo_from_JY/left/stereo_pair_014.jpg");
    cv::Matx33d K_dst = K_src;
    cv::Mat image = cv::imread(file), image_projected;
    cv::Vec4d D_dst_vec (-1.0, 0.0, 0.0, 0.0);
    cv::Mat D_dst = cv::Mat(D_dst_vec);

    int imageWidth = (int)this->imageSize.width;
    int imageHeight = (int)this->imageSize.height;

    cv::Mat imagePoints(imageHeight, imageWidth, CV_32FC2), undPoints, distPoints;
    cv::Vec2f* pts = imagePoints.ptr<cv::Vec2f>();

    for(int y = 0, k = 0; y < imageHeight; ++y)
    {
        for(int x = 0; x < imageWidth; ++x)
        {
            cv::Vec2f point((float)x, (float)y);
            pts[k++] = point;
        }
    }

    cv::fisheye::undistortPoints(imagePoints, undPoints, K_dst, D_dst);
    cv::fisheye::distortPoints(undPoints, distPoints, K_src, D_src);
    cv::remap(image, image_projected, distPoints, cv::noArray(), cv::INTER_LINEAR);

    float dx, dy, r_sq;
    float R_MAX = 250;
    float imageCenterX = (float)imageWidth / 2;
    float imageCenterY = (float)imageHeight / 2;

    cv::Mat undPointsGt(imageHeight, imageWidth, CV_32FC2);
    cv::Mat imageGt(imageHeight, imageWidth, CV_8UC3);

    for(int y = 0; y < imageHeight; ++y)
    {
        for(int x = 0; x < imageWidth; ++x)
        {
            dx = x - imageCenterX;
            dy = y - imageCenterY;
            r_sq = dy * dy + dx * dx;

            Vec2f & und_vec = undPoints.at<Vec2f>(y,x);
            Vec3b & pixel = image_projected.at<Vec3b>(y,x);

            Vec2f & undist_vec_gt = undPointsGt.at<Vec2f>(y,x);
            Vec3b & pixel_gt = imageGt.at<Vec3b>(y,x);

            if (r_sq > R_MAX * R_MAX)
            {

                undist_vec_gt[0] = -1e6;
                undist_vec_gt[1] = -1e6;

                pixel_gt[0] = 0;
                pixel_gt[1] = 0;
                pixel_gt[2] = 0;
            }
            else
            {
                undist_vec_gt[0] = und_vec[0];
                undist_vec_gt[1] = und_vec[1];

                pixel_gt[0] = pixel[0];
                pixel_gt[1] = pixel[1];
                pixel_gt[2] = pixel[2];
            }

        }
    }

    EXPECT_MAT_NEAR(undPoints, undPointsGt, 1e-10);
    EXPECT_MAT_NEAR(image_projected, imageGt, 1e-10);

    Vec2f dist_point_1 = distPoints.at<Vec2f>(400, 640);
    Vec2f dist_point_1_gt(640.044f, 400.041f);

    Vec2f dist_point_2 = distPoints.at<Vec2f>(400, 440);
    Vec2f dist_point_2_gt(409.731f, 403.029f);

    Vec2f dist_point_3 = distPoints.at<Vec2f>(200, 640);
    Vec2f dist_point_3_gt(643.341f, 168.896f);

    Vec2f dist_point_4 = distPoints.at<Vec2f>(300, 480);
    Vec2f dist_point_4_gt(463.402f, 290.317f);

    Vec2f dist_point_5 = distPoints.at<Vec2f>(550, 750);
    Vec2f dist_point_5_gt(797.51f, 611.637f);

    EXPECT_MAT_NEAR(dist_point_1, dist_point_1_gt, 1e-2);
    EXPECT_MAT_NEAR(dist_point_2, dist_point_2_gt, 1e-2);
    EXPECT_MAT_NEAR(dist_point_3, dist_point_3_gt, 1e-2);
    EXPECT_MAT_NEAR(dist_point_4, dist_point_4_gt, 1e-2);
    EXPECT_MAT_NEAR(dist_point_5, dist_point_5_gt, 1e-2);

    // Add the "--test_debug" to arguments for file output
    if (cvtest::debugLevel > 0)
        cv::imwrite(combine(datasets_repository_path, "new_distortion.png"), image_projected);
}

TEST_F(fisheyeTest, jacobians)
{
    int n = 10;
    cv::Mat X(1, n, CV_64FC3);
    cv::Mat om(3, 1, CV_64F), theT(3, 1, CV_64F);
    cv::Mat f(2, 1, CV_64F), c(2, 1, CV_64F);
    cv::Mat k(4, 1, CV_64F);
    double alpha;

    cv::RNG r;

    r.fill(X, cv::RNG::NORMAL, 2, 1);
    X = cv::abs(X) * 10;

    r.fill(om, cv::RNG::NORMAL, 0, 1);
    om = cv::abs(om);

    r.fill(theT, cv::RNG::NORMAL, 0, 1);
    theT = cv::abs(theT); theT.at<double>(2) = 4; theT *= 10;

    r.fill(f, cv::RNG::NORMAL, 0, 1);
    f = cv::abs(f) * 1000;

    r.fill(c, cv::RNG::NORMAL, 0, 1);
    c = cv::abs(c) * 1000;

    r.fill(k, cv::RNG::NORMAL, 0, 1);
    k*= 0.5;

    alpha = 0.01*r.gaussian(1);

    cv::Mat x1, x2, xpred;
    cv::Matx33d theK(f.at<double>(0), alpha * f.at<double>(0), c.at<double>(0),
                     0,            f.at<double>(1), c.at<double>(1),
                     0,            0,    1);

    cv::Mat jacobians;
    cv::fisheye::projectPoints(X, x1, om, theT, theK, k, alpha, jacobians);

    //test on T:
    cv::Mat dT(3, 1, CV_64FC1);
    r.fill(dT, cv::RNG::NORMAL, 0, 1);
    dT *= 1e-9*cv::norm(theT);
    cv::Mat T2 = theT + dT;
    cv::fisheye::projectPoints(X, x2, om, T2, theK, k, alpha, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(11,14) * dT).reshape(2, 1);
    CV_Assert (cv::norm(x2 - xpred) < 1e-10);

    //test on om:
    cv::Mat dom(3, 1, CV_64FC1);
    r.fill(dom, cv::RNG::NORMAL, 0, 1);
    dom *= 1e-9*cv::norm(om);
    cv::Mat om2 = om + dom;
    cv::fisheye::projectPoints(X, x2, om2, theT, theK, k, alpha, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(8,11) * dom).reshape(2, 1);
    CV_Assert (cv::norm(x2 - xpred) < 1e-10);

    //test on f:
    cv::Mat df(2, 1, CV_64FC1);
    r.fill(df, cv::RNG::NORMAL, 0, 1);
    df *= 1e-9*cv::norm(f);
    cv::Matx33d K2 = theK + cv::Matx33d(df.at<double>(0), df.at<double>(0) * alpha, 0, 0, df.at<double>(1), 0, 0, 0, 0);
    cv::fisheye::projectPoints(X, x2, om, theT, K2, k, alpha, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(0,2) * df).reshape(2, 1);
    CV_Assert (cv::norm(x2 - xpred) < 1e-10);

    //test on c:
    cv::Mat dc(2, 1, CV_64FC1);
    r.fill(dc, cv::RNG::NORMAL, 0, 1);
    dc *= 1e-9*cv::norm(c);
    K2 = theK + cv::Matx33d(0, 0, dc.at<double>(0), 0, 0, dc.at<double>(1), 0, 0, 0);
    cv::fisheye::projectPoints(X, x2, om, theT, K2, k, alpha, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(2,4) * dc).reshape(2, 1);
    CV_Assert (cv::norm(x2 - xpred) < 1e-10);

    //test on k:
    cv::Mat dk(4, 1, CV_64FC1);
    r.fill(dk, cv::RNG::NORMAL, 0, 1);
    dk *= 1e-9*cv::norm(k);
    cv::Mat k2 = k + dk;
    cv::fisheye::projectPoints(X, x2, om, theT, theK, k2, alpha, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.colRange(4,8) * dk).reshape(2, 1);
    CV_Assert (cv::norm(x2 - xpred) < 1e-10);

    //test on alpha:
    cv::Mat dalpha(1, 1, CV_64FC1);
    r.fill(dalpha, cv::RNG::NORMAL, 0, 1);
    dalpha *= 1e-9*cv::norm(f);
    double alpha2 = alpha + dalpha.at<double>(0);
    K2 = theK + cv::Matx33d(0, f.at<double>(0) * dalpha.at<double>(0), 0, 0, 0, 0, 0, 0, 0);
    cv::fisheye::projectPoints(X, x2, om, theT, theK, k, alpha2, cv::noArray());
    xpred = x1 + cv::Mat(jacobians.col(14) * dalpha).reshape(2, 1);
    CV_Assert (cv::norm(x2 - xpred) < 1e-10);
}

}}
