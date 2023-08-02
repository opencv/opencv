// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/ts/cuda_test.hpp> // EXPECT_MAT_NEAR
#include "opencv2/3d.hpp"
#include <opencv2/core/utils/logger.hpp>

namespace opencv_test { namespace {

static bool checkPandROI(const Matx33d& M, const Matx<double, 5, 1>& D,
                          const Mat& R, const Mat& P, Size imgsize, Rect roi)
{
    const double eps = 0.05;
    const int N = 21;
    int x, y, k;
    vector<Point2f> pts, upts;

    // step 1. check that all the original points belong to the destination image
    for( y = 0; y < N; y++ )
        for( x = 0; x < N; x++ )
            pts.push_back(Point2f((float)x*imgsize.width/(N-1), (float)y*imgsize.height/(N-1)));

    undistortPoints(pts, upts, M, D, R, P );
    for( k = 0; k < N*N; k++ )
        if( upts[k].x < -imgsize.width*eps || upts[k].x > imgsize.width*(1+eps) ||
            upts[k].y < -imgsize.height*eps || upts[k].y > imgsize.height*(1+eps) )
        {
            CV_LOG_ERROR(NULL, cv::format("The point (%g, %g) was mapped to (%g, %g) which is out of image\n",
                                          pts[k].x, pts[k].y, upts[k].x, upts[k].y));
            return false;
        }

    // step 2. check that all the points inside ROI belong to the original source image
    Mat temp(imgsize, CV_8U), utemp, map1, map2;
    temp = Scalar::all(1);
    initUndistortRectifyMap(M, D, R, P, imgsize, CV_16SC2, map1, map2);
    remap(temp, utemp, map1, map2, INTER_LINEAR);

    if(roi.x < 0 || roi.y < 0 || roi.x + roi.width > imgsize.width || roi.y + roi.height > imgsize.height)
    {
            CV_LOG_ERROR(NULL, cv::format("The ROI=(%d, %d, %d, %d) is outside of the imge rectangle\n",
                                          roi.x, roi.y, roi.width, roi.height));
            return false;
    }
    double s = sum(utemp(roi))[0];
    if( s > roi.area() || roi.area() - s > roi.area()*(1-eps) )
    {
            CV_LOG_ERROR(NULL, cv::format("The ratio of black pixels inside the valid ROI (~%g%%) is too large\n",
                                           s*100./roi.area()));
            return false;
    }

    return true;
}

TEST(StereoGeometry, stereoRectify)
{
    // camera parameters are extracted from the original calib3d test CV_StereoCalibrationTest::run
    const Matx33d M1(
        530.4643719672913, 0, 319.5,
        0, 529.7477570329314, 239.5,
        0, 0, 1);
    const Matx<double, 5, 1> D1(-0.2982901576925627, 0.1134645765152131, 0, 0, 0);

    const Matx33d M2(
        530.4643719672913, 0, 319.5,
        0, 529.7477570329314, 239.5,
        0, 0, 1);
    const Matx<double, 5, 1> D2(-0.2833068597502156, 0.0944810713984697, 0, 0, 0);

    const Matx33d R(0.9996903750450727, 0.005330951201286465, -0.02430504066096785,
                    -0.004837810799471072, 0.9997821583334892, 0.02030348405319902,
                    0.02440798289310936, -0.02017961439967296, 0.9994983909610711);
    const Matx31d T(-3.328706469151101, 0.05621025406095936, -0.02956576727262086);

    const Size imageSize(640, 480);

    Mat R1, R2, P1, P2, Q;
    Rect roi1, roi2;

    stereoRectify( M1, D1, M2, D2, imageSize, R, T, R1, R2, P1, P2, Q, 0, 1, imageSize, &roi1, &roi2 );

    Mat eye33 = Mat::eye(3,3,CV_64F);
    Mat R1t = R1.t(), R2t = R2.t();

    EXPECT_LE(cvtest::norm(R1t*R1 - eye33, NORM_L2), 0.01) << "R1 is not orthogonal!";
    EXPECT_LE(cvtest::norm(R2t*R2 - eye33, NORM_L2), 0.01) << "R2 is not orthogonal!";

    //check that Tx after rectification is equal to distance between cameras
    double tx = fabs(P2.at<double>(0, 3) / P2.at<double>(0, 0));
    EXPECT_LE(fabs(tx - cvtest::norm(T, NORM_L2)), 1e-5);
    EXPECT_TRUE(checkPandROI(M1, D1, R1, P1, imageSize, roi1));
    EXPECT_TRUE(checkPandROI(M2, D2, R2, P2, imageSize, roi2));

    //check that Q reprojects points before the camera
    double testPoint[4] = {0.0, 0.0, 100.0, 1.0};
    Mat reprojectedTestPoint = Q * Mat_<double>(4, 1, testPoint);
    CV_Assert(reprojectedTestPoint.type() == CV_64FC1);
    EXPECT_GT( reprojectedTestPoint.at<double>(2) / reprojectedTestPoint.at<double>(3), 0 ) << \
        "A point after rectification is reprojected behind the camera";
}

TEST(StereoGeometry, regression_10791)
{
    const Matx33d M1(
        853.1387981631528, 0, 704.154907802121,
        0, 853.6445089162528, 520.3600712930319,
        0, 0, 1
    );
    const Matx33d M2(
        848.6090216909176, 0, 701.6162856852185,
        0, 849.7040162357157, 509.1864036137,
        0, 0, 1
    );
    const Matx<double, 14, 1> D1(-6.463598629567206, 79.00104930508179, -0.0001006144444464403, -0.0005437499822299972,
        12.56900616588467, -6.056719942752855, 76.3842481414836, 45.57460250612659,
        0, 0, 0, 0, 0, 0);
    const Matx<double, 14, 1> D2(0.6123436439798265, -0.4671756923224087, -0.0001261947899033442, -0.000597334584036978,
        -0.05660119809538371, 1.037075740629769, -0.3076042835831711, -0.2502169324283623,
        0, 0, 0, 0, 0, 0);

    const Matx33d R(
        0.9999926627018476, -0.0001095586963765905, 0.003829169539302921,
        0.0001021735876758584, 0.9999981346680941, 0.0019287874145156,
        -0.003829373712065528, -0.001928382022437616, 0.9999908085776333
    );
    const Matx31d T(-58.9161771697128, -0.01581306249996402, -0.8492960216760961);

    const Size imageSize(1280, 960);

    Mat R1, R2, P1, P2, Q;
    Rect roi1, roi2;
    stereoRectify(M1, D1, M2, D2, imageSize, R, T,
                  R1, R2, P1, P2, Q,
                  STEREO_ZERO_DISPARITY, 1, imageSize, &roi1, &roi2);

    EXPECT_GE(roi1.area(), 400*300) << roi1;
    EXPECT_GE(roi2.area(), 400*300) << roi2;
}

TEST(StereoGeometry, regression_11131)
{
    const Matx33d M1(
        1457.572438721727, 0, 1212.945694211622,
        0, 1457.522226502963, 1007.32058848921,
        0, 0, 1
    );
    const Matx33d M2(
        1460.868570835972, 0, 1215.024068023046,
        0, 1460.791367088, 1011.107202932225,
        0, 0, 1
    );
    const Matx<double, 5, 1> D1(0, 0, 0, 0, 0);
    const Matx<double, 5, 1> D2(0, 0, 0, 0, 0);

    const Matx33d R(
        0.9985404059825475, 0.02963547172078553, -0.04515303352041626,
        -0.03103795276460111, 0.9990471552537432, -0.03068268351343364,
        0.04420071389006859, 0.03203935697372317, 0.9985087763742083
    );
    const Matx31d T(0.9995500167379527, 0.0116311595111068, 0.02764923448462666);

    const Size imageSize(2456, 2058);

    Mat R1, R2, P1, P2, Q;
    Rect roi1, roi2;
    stereoRectify(M1, D1, M2, D2, imageSize, R, T,
                  R1, R2, P1, P2, Q,
                  STEREO_ZERO_DISPARITY, 1, imageSize, &roi1, &roi2);

    EXPECT_GT(P1.at<double>(0, 0), 0);
    EXPECT_GT(P2.at<double>(0, 0), 0);
    EXPECT_GT(R1.at<double>(0, 0), 0);
    EXPECT_GT(R2.at<double>(0, 0), 0);
    EXPECT_GE(roi1.area(), 400*300) << roi1;
    EXPECT_GE(roi2.area(), 400*300) << roi2;
}

TEST(StereoGeometry, regression_23305)
{
    const Matx33d M1(
        850, 0, 640,
        0, 850, 640,
        0, 0, 1
    );

    const Matx34d P1_gold(
        850, 0, 640, 0,
        0, 850, 640, 0,
        0, 0, 1, 0
    );

    const Matx33d M2(
        850, 0, 640,
        0, 850, 640,
        0, 0, 1
    );

    const Matx34d P2_gold(
        850, 0, 640, -2*850, // correcponds to T(-2., 0., 0.)
        0, 850, 640, 0,
        0, 0, 1, 0
    );

    const Matx<double, 5, 1> D1(0, 0, 0, 0, 0);
    const Matx<double, 5, 1> D2(0, 0, 0, 0, 0);

    const Matx33d R(
        1., 0., 0.,
        0., 1., 0.,
        0., 0., 1.
    );
    const Matx31d T(-2., 0., 0.);

    const Size imageSize(1280, 1280);

    Mat R1, R2, P1, P2, Q;
    Rect roi1, roi2;
    stereoRectify(M1, D1, M2, D2, imageSize, R, T,
                  R1, R2, P1, P2, Q,
                  STEREO_ZERO_DISPARITY, 0, imageSize, &roi1, &roi2);

    EXPECT_EQ(cv::norm(P1, P1_gold), 0.);
    EXPECT_EQ(cv::norm(P2, P2_gold), 0.);
}

class fisheyeTest : public ::testing::Test {

protected:
    const static cv::Size imageSize;
    const static cv::Matx33d K;
    const static cv::Vec4d D;
    const static cv::Matx33d R;
    const static cv::Vec3d T;
    std::string datasets_repository_path;

    virtual void SetUp() {
        datasets_repository_path = combine(cvtest::TS::ptr()->get_data_path(), "cv/cameracalibration/fisheye");
    }

protected:
    std::string combine(const std::string& _item1, const std::string& _item2);
    static void merge4(const cv::Mat& tl, const cv::Mat& tr, const cv::Mat& bl, const cv::Mat& br, cv::Mat& merged);
};

const cv::Size fisheyeTest::imageSize(1280, 800);

const cv::Matx33d fisheyeTest::K(558.478087865323,               0, 620.458515360843,
                              0, 560.506767351568, 381.939424848348,
                              0,               0,                1);

const cv::Vec4d fisheyeTest::D(-0.0014613319981768, -0.00329861110580401, 0.00605760088590183, -0.00374209380722371);


const cv::Matx33d fisheyeTest::R ( 9.9756700084424932e-01, 6.9698277640183867e-02, 1.4929569991321144e-03,
                            -6.9711825162322980e-02, 9.9748249845531767e-01, 1.2997180766418455e-02,
                            -5.8331736398316541e-04,-1.3069635393884985e-02, 9.9991441852366736e-01);

const cv::Vec3d fisheyeTest::T(-9.9217369356044638e-02, 3.1741831972356663e-03, 1.8551007952921010e-04);

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

void fisheyeTest::merge4(const cv::Mat& tl, const cv::Mat& tr, const cv::Mat& bl, const cv::Mat& br, cv::Mat& merged)
{
    int type = tl.type();
    cv::Size sz = tl.size();
    ASSERT_EQ(type, tr.type()); ASSERT_EQ(type, bl.type()); ASSERT_EQ(type, br.type());
    ASSERT_EQ(sz.width, tr.cols); ASSERT_EQ(sz.width, bl.cols); ASSERT_EQ(sz.width, br.cols);
    ASSERT_EQ(sz.height, tr.rows); ASSERT_EQ(sz.height, bl.rows); ASSERT_EQ(sz.height, br.rows);

    merged.create(cv::Size(sz.width * 2, sz.height * 2), type);
    tl.copyTo(merged(cv::Rect(0, 0, sz.width, sz.height)));
    tr.copyTo(merged(cv::Rect(sz.width, 0, sz.width, sz.height)));
    bl.copyTo(merged(cv::Rect(0, sz.height, sz.width, sz.height)));
    br.copyTo(merged(cv::Rect(sz.width, sz.height, sz.width, sz.height)));
}

TEST_F(fisheyeTest, stereoRectify)
{
    const std::string folder = combine(datasets_repository_path, "calib-3_stereo_from_JY");

    cv::Size calibration_size = this->imageSize, requested_size = calibration_size;
    cv::Matx33d K1 = this->K, K2 = K1;
    cv::Mat D1 = cv::Mat(this->D), D2 = D1;

    cv::Vec3d theT = this->T;
    cv::Matx33d theR = this->R;

    double balance = 0.0, fov_scale = 1.1;
    cv::Mat R1, R2, P1, P2, Q;
    cv::fisheye::stereoRectify(K1, D1, K2, D2, calibration_size, theR, theT, R1, R2, P1, P2, Q,
                      cv::STEREO_ZERO_DISPARITY, requested_size, balance, fov_scale);

    // Collected with these CMake flags: -DWITH_IPP=OFF -DCV_ENABLE_INTRINSICS=OFF -DCV_DISABLE_OPTIMIZATION=ON -DCMAKE_BUILD_TYPE=Debug
    cv::Matx33d R1_ref(
        0.9992853269091279, 0.03779164101000276, -0.0007920188690205426,
        -0.03778569762983931, 0.9992646472015868, 0.006511981857667881,
        0.001037534936357442, -0.006477400933964018, 0.9999784831677112
    );
    cv::Matx33d R2_ref(
        0.9994868963898833, -0.03197579751378937, -0.001868774538573449,
        0.03196298186616116, 0.9994677442608699, -0.0065265589947392,
        0.002076471801477729, 0.006463478587068991, 0.9999769555891836
    );
    cv::Matx34d P1_ref(
        420.9684016542647, 0, 586.3059567784627, 0,
        0, 420.9684016542647, 374.8571836462291, 0,
        0, 0, 1, 0
    );
    cv::Matx34d P2_ref(
        420.9684016542647, 0, 586.3059567784627, -41.78881938824554,
        0, 420.9684016542647, 374.8571836462291, 0,
        0, 0, 1, 0
    );
    cv::Matx44d Q_ref(
        1, 0, 0, -586.3059567784627,
        0, 1, 0, -374.8571836462291,
        0, 0, 0, 420.9684016542647,
        0, 0, 10.07370889670733, -0
    );

    const double eps = 1e-10;
    EXPECT_MAT_NEAR(R1_ref, R1, eps);
    EXPECT_MAT_NEAR(R2_ref, R2, eps);
    EXPECT_MAT_NEAR(P1_ref, P1, eps);
    EXPECT_MAT_NEAR(P2_ref, P2, eps);
    EXPECT_MAT_NEAR(Q_ref, Q, eps);

    if (::testing::Test::HasFailure())
    {
        std::cout << "Actual values are:" << std::endl
            << "R1 =" << std::endl << R1 << std::endl
            << "R2 =" << std::endl << R2 << std::endl
            << "P1 =" << std::endl << P1 << std::endl
            << "P2 =" << std::endl << P2 << std::endl
            << "Q =" << std::endl << Q << std::endl;
    }

    if (cvtest::debugLevel == 0)
        return;
    // DEBUG code is below

    cv::Mat lmapx, lmapy, rmapx, rmapy;
    //rewrite for fisheye
    cv::fisheye::initUndistortRectifyMap(K1, D1, R1, P1, requested_size, CV_32F, lmapx, lmapy);
    cv::fisheye::initUndistortRectifyMap(K2, D2, R2, P2, requested_size, CV_32F, rmapx, rmapy);

    cv::Mat l, r, lundist, rundist;
    for (int i = 0; i < 34; ++i)
    {
        SCOPED_TRACE(cv::format("image %d", i));
        l = imread(combine(folder, cv::format("left/stereo_pair_%03d.jpg", i)), cv::IMREAD_COLOR);
        r = imread(combine(folder, cv::format("right/stereo_pair_%03d.jpg", i)), cv::IMREAD_COLOR);
        ASSERT_FALSE(l.empty());
        ASSERT_FALSE(r.empty());

        int ndisp = 128;
        cv::rectangle(l, cv::Rect(255,       0, 829,       l.rows-1), cv::Scalar(0, 0, 255));
        cv::rectangle(r, cv::Rect(255,       0, 829,       l.rows-1), cv::Scalar(0, 0, 255));
        cv::rectangle(r, cv::Rect(255-ndisp, 0, 829+ndisp ,l.rows-1), cv::Scalar(0, 0, 255));
        cv::remap(l, lundist, lmapx, lmapy, cv::INTER_LINEAR);
        cv::remap(r, rundist, rmapx, rmapy, cv::INTER_LINEAR);

        for (int ii = 0; ii < lundist.rows; ii += 20)
        {
            cv::line(lundist, cv::Point(0, ii), cv::Point(lundist.cols, ii), cv::Scalar(0, 255, 0));
            cv::line(rundist, cv::Point(0, ii), cv::Point(lundist.cols, ii), cv::Scalar(0, 255, 0));
        }

        cv::Mat rectification;
        merge4(l, r, lundist, rundist, rectification);

        // Add the "--test_debug" to arguments for file output
        if (cvtest::debugLevel > 0)
            cv::imwrite(cv::format("fisheye_rectification_AB_%03d.png", i), rectification);
    }
}

}}
