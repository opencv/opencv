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

#include "test_precomp.hpp"

namespace opencv_test {
namespace {

PARAM_TEST_CASE(Video_ECC, int, bool)
{
    int motionType;
    bool usePyramids;
    virtual void SetUp()
    {
        motionType = GET_PARAM(0);
        usePyramids = GET_PARAM(1);
    }
};

class CV_ECC_Test : public cvtest::BaseTest {
   public:
    CV_ECC_Test(int motionType, bool usePyramids);
    virtual ~CV_ECC_Test();

   protected:
    int motionType;
    double MAX_RMS;  // upper bound for RMS error

    double computeRMS(const Mat& mat1, const Mat& mat2);
    bool isMapCorrect(const Mat& mat);

    virtual bool test(const Mat img);
    bool testAllTypes(const Mat img);               // run test for all supported data types (U8, U16, F32, F64)
    bool testAllChNum(const Mat img);               // run test for all supported channels count (gray, RGB)

    void run(int);

    bool checkMap(const Mat& map, const Mat& ground);

    int ntests;          // number of tests per motion type
    int ECC_iterations;  // number of iterations for ECC
    double ECC_epsilon;  // we choose a negative value, so that
    // ECC_iterations are always executed
    TermCriteria criteria;
    bool usePyramids;       // use version of findTransformECC with pyramids
};


CV_ECC_Test::CV_ECC_Test(int a_motionType, bool a_usePyramids) : motionType(a_motionType)
    , MAX_RMS(0.1)
    , ntests(3)
    , ECC_iterations(50)
    , ECC_epsilon(-1)
    , criteria(TermCriteria::COUNT + TermCriteria::EPS, ECC_iterations, ECC_epsilon)
    , usePyramids(a_usePyramids)
    {}


CV_ECC_Test::~CV_ECC_Test() {}

bool CV_ECC_Test::isMapCorrect(const Mat& map) {
    bool tr = true;
    float mapVal;
    for (int i = 0; i < map.rows; i++)
        for (int j = 0; j < map.cols; j++) {
            mapVal = map.at<float>(i, j);
            tr = tr & (!cvIsNaN(mapVal) && (fabs(mapVal) < 1e9));
        }

    return tr;
}

double CV_ECC_Test::computeRMS(const Mat& mat1, const Mat& mat2) {
    CV_Assert(mat1.rows == mat2.rows);
    CV_Assert(mat1.cols == mat2.cols);

    Mat errorMat;
    subtract(mat1, mat2, errorMat);

    return sqrt(errorMat.dot(errorMat) / (mat1.rows * mat1.cols * mat1.channels()));
}

bool CV_ECC_Test::checkMap(const Mat& map, const Mat& ground) {
    if (!isMapCorrect(map)) {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        return false;
    }

    if (computeRMS(map, ground) > MAX_RMS) {
        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        ts->printf(ts->LOG, "RMS = %f", computeRMS(map, ground));
        return false;
    }
    return true;
}

bool CV_ECC_Test::test(const Mat img)
{
    cv::RNG rng = ts->get_rng();

    int progress = 0;

    for (int k = 0; k < ntests; k++) {
        ts->update_context(this, k, true);
        progress = update_progress(progress, k, ntests, 0);

        Mat groundMap;
        switch(motionType)
        {
            case MOTION_TRANSLATION:
                groundMap = (Mat_<float>(2, 3) << 1, 0, (rng.uniform(10.f, 20.f)), 0, 1, (rng.uniform(10.f, 20.f)));
                break;
            case MOTION_EUCLIDEAN:
            {
                double angle = CV_PI / 30 + CV_PI * rng.uniform((double)-2.f, (double)2.f) / 180;
                groundMap = (Mat_<float>(2, 3) << cos(angle), -sin(angle), (rng.uniform(10.f, 20.f)), sin(angle),
                            cos(angle), (rng.uniform(10.f, 20.f)));
                break;
            }
            case MOTION_AFFINE:
                groundMap = (Mat_<float>(2, 3) << (1 - rng.uniform(-0.05f, 0.05f)), (rng.uniform(-0.03f, 0.03f)),
                            (rng.uniform(10.f, 20.f)), (rng.uniform(-0.03f, 0.03f)), (1 - rng.uniform(-0.05f, 0.05f)),
                            (rng.uniform(10.f, 20.f)));
                break;
            case MOTION_HOMOGRAPHY:
                groundMap =
                    (Mat_<float>(3, 3) << (1 - rng.uniform(-0.05f, 0.05f)), (rng.uniform(-0.03f, 0.03f)),
                    (rng.uniform(10.f, 20.f)), (rng.uniform(-0.03f, 0.03f)), (1 - rng.uniform(-0.05f, 0.05f)),
                    (rng.uniform(10.f, 20.f)), (rng.uniform(0.0001f, 0.0003f)), (rng.uniform(0.0001f, 0.0003f)), 1.f);
                break;
            default:
                CV_Error(Error::StsBadArg, "Incorrect motion type");
                break;
        }

        Mat warpedImage;

        Mat foundMap;
        if(motionType == MOTION_HOMOGRAPHY)
        {
            warpPerspective(img, warpedImage, groundMap, Size(200, 200), INTER_LINEAR + WARP_INVERSE_MAP);
            foundMap = Mat::eye(3, 3, CV_32F);
        }
        else
        {
            warpAffine(img, warpedImage, groundMap, Size(200, 200), INTER_LINEAR + WARP_INVERSE_MAP);
            foundMap = Mat((Mat_<float>(2, 3) << 1, 0, 0, 0, 1, 0));
        }


        if(usePyramids)
        {
            ECCParameters params;
            params.criteria = criteria;
            params.motionType = motionType;
            findTransformECCMultiScale(warpedImage, img, foundMap, params);
        }
        else
            findTransformECC(warpedImage, img, foundMap, motionType, criteria);

        if (!checkMap(foundMap, groundMap))
            return false;
    }
    return true;
}

bool CV_ECC_Test::testAllTypes(const Mat img) {
    auto types = {CV_8U, CV_16U, CV_32F, CV_64F};
    for (auto type : types) {
        Mat timg;
        img.convertTo(timg, type);
        if (!test(timg))
            return false;
    }
    return true;
}

bool CV_ECC_Test::testAllChNum(const Mat img) {
    if(!usePyramids)
        if (!testAllTypes(img))
            return false;

    Mat gray;
    cvtColor(img, gray, COLOR_RGB2GRAY);
    if (!testAllTypes(gray))
        return false;

    return true;
}

void CV_ECC_Test::run(int) {
    Mat img = imread(string(ts->get_data_path()) + "shared/fruits.png");
    if (img.empty()) {
        ts->printf(ts->LOG, "test image can not be read");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }

    Mat testImg;
    resize(img, testImg, Size(216, 216), 0, 0, INTER_LINEAR_EXACT);

    testAllChNum(testImg);

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST_P(Video_ECC, accuracy) {
    CV_ECC_Test test(motionType, usePyramids);
    test.safe_run();
}

INSTANTIATE_TEST_CASE_P(ECCfixtures, Video_ECC,
    testing::Values(testing::make_tuple(MOTION_TRANSLATION, false),
                    testing::make_tuple(MOTION_TRANSLATION, true),
                    testing::make_tuple(MOTION_EUCLIDEAN, false),
                    testing::make_tuple(MOTION_EUCLIDEAN, true),
                    testing::make_tuple(MOTION_AFFINE, false),
                    testing::make_tuple(MOTION_AFFINE, true),
                    testing::make_tuple(MOTION_HOMOGRAPHY, false),
                    testing::make_tuple(MOTION_HOMOGRAPHY, true)));

class CV_ECC_Test_Mask : public CV_ECC_Test {
   public:
    CV_ECC_Test_Mask();

   protected:
    bool test(const Mat);
};

CV_ECC_Test_Mask::CV_ECC_Test_Mask():CV_ECC_Test(MOTION_TRANSLATION, false) {}

bool CV_ECC_Test_Mask::test(const Mat testImg) {
    cv::RNG rng = ts->get_rng();

    int progress = 0;

    for (int k = 0; k < ntests; k++) {
        ts->update_context(this, k, true);
        progress = update_progress(progress, k, ntests, 0);

        Mat translationGround = (Mat_<float>(2, 3) << 1, 0, (rng.uniform(10.f, 20.f)), 0, 1, (rng.uniform(10.f, 20.f)));

        Mat warpedImage;

        warpAffine(testImg, warpedImage, translationGround, Size(200, 200), INTER_LINEAR + WARP_INVERSE_MAP);

        Mat mapTranslation = (Mat_<float>(2, 3) << 1, 0, 0, 0, 1, 0);

        Mat_<unsigned char> mask = Mat_<unsigned char>::ones(testImg.rows, testImg.cols);
        Rect region(testImg.cols * 2 / 3, testImg.rows * 2 / 3, testImg.cols / 3, testImg.rows / 3);

        rectangle(testImg, region, Scalar::all(0), FILLED);
        rectangle(mask, region, Scalar(0), FILLED);

        findTransformECC(warpedImage, testImg, mapTranslation, 0, criteria, mask);

        if (!checkMap(mapTranslation, translationGround))
            return false;

        // Test with non-default gaussian blur.
        findTransformECC(warpedImage, testImg, mapTranslation, 0, criteria, mask, 1);

        if (!checkMap(mapTranslation, translationGround))
            return false;

        // Test with template mask.
        Mat_<unsigned char> warpedMask = Mat_<unsigned char>::ones(warpedImage.rows, warpedImage.cols);
        for (int i=warpedImage.rows*1/3; i<warpedImage.rows*2/3; i++) {
            for (int j=warpedImage.cols*1/3; j<warpedImage.cols*2/3; j++) {
                warpedMask(i, j) = 0;
            }
        }

        findTransformECCWithMask(warpedImage, testImg, warpedMask, mask, mapTranslation, 0,
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, ECC_iterations, ECC_epsilon));

        if (!checkMap(mapTranslation, translationGround))
            return false;

        // Test with non-default gaussian blur.
        findTransformECCWithMask(warpedImage, testImg, warpedMask,  mask, mapTranslation, 0, criteria, 1);

        if (!checkMap(mapTranslation, translationGround))
            return false;
    }
    return true;
}

class CV_ECC_BigPictureTest : public CV_ECC_Test {
   public:
    CV_ECC_BigPictureTest(bool a_maskedVersion) : CV_ECC_Test(MOTION_HOMOGRAPHY, true), maskedVersion(a_maskedVersion) {}
    virtual ~CV_ECC_BigPictureTest() {}
   protected:
    void run(int);
    bool maskedVersion;
};

void CV_ECC_BigPictureTest::run(int)
{
    Mat largeGray0 = imread(string(ts->get_data_path()) + "shared/halmosh0.jpg", IMREAD_GRAYSCALE);
    Mat largeGray1;
    Mat roiMask0;
    Mat roiMask1;
    Mat expectedRes;
    bool readError = false;
    if(maskedVersion)
    {
        largeGray1 = imread(string(ts->get_data_path()) + "shared/halmosh2.jpg", IMREAD_GRAYSCALE);
        roiMask0 = imread(string(ts->get_data_path()) + "shared/halmosh0mask.png", IMREAD_GRAYSCALE);
        roiMask1 = imread(string(ts->get_data_path()) + "shared/halmosh2mask.png", IMREAD_GRAYSCALE);
        readError = largeGray0.empty() || largeGray1.empty() || roiMask0.empty() || roiMask1.empty();
        expectedRes = (Mat_<float>(3, 3) << 1.0225, 0.0606, -28.6452, -0.0475, 1.0314, 11.819, 8.21e-06, -3.65e-07, 1);
    }
    else
    {
        largeGray1 = imread(string(ts->get_data_path()) + "shared/halmosh1.jpg", IMREAD_GRAYSCALE);
        readError = largeGray0.empty() || largeGray1.empty();
        expectedRes = (Mat_<float>(3, 3) << 0.9756, -0.0319, 24.685, 0.013, 0.9808, 7.7453, -2.35e-05, -9.12e-06, 1);
    }

    if(readError)
    {
        ts->printf(ts->LOG, "test image can not be read");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }

    cv::Mat found = cv::Mat::eye(3, 3, CV_32F);
    constexpr int N_ITERS = 20;
    constexpr double TERMINATION_EPS = 1e-6;
    ECCParameters params;
    params.criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, N_ITERS, TERMINATION_EPS);
    params.motionType = MOTION_HOMOGRAPHY;
    params.nlevels = 5;
    params.itersPerLevel = {5, 10, 300, 300, 1000};
    findTransformECCMultiScale(largeGray0, largeGray1, found, params, roiMask0, roiMask1);
    ASSERT_EQ(checkMap(found, expectedRes), true);
    ts->set_failed_test_info(cvtest::TS::OK);
}

void testECCProperties(Mat x, float eps) {
    // The channels are independent
    Mat y = x.t();
    Mat Z = Mat::zeros(x.size(), y.type());
    Mat O = Mat::ones(x.size(), y.type());

    EXPECT_NEAR(computeECC(x, y), 0.0, eps);
    if (x.type() != CV_8U && x.type() != CV_8U) {
        EXPECT_NEAR(computeECC(x + y, x - y), 0.0, eps);
    }

    EXPECT_NEAR(computeECC(x, x), 1.0, eps);

    Mat R, G, B, X, Y;
    cv::merge(std::vector<cv::Mat>({O, Z, Z}), R);
    cv::merge(std::vector<cv::Mat>({Z, O, Z}), G);
    cv::merge(std::vector<cv::Mat>({Z, Z, O}), B);
    cv::merge(std::vector<cv::Mat>({x, x, x}), X);
    cv::merge(std::vector<cv::Mat>({y, y, y}), Y);

    // 1. The channels are orthogonal and independent
    EXPECT_NEAR(computeECC(X.mul(R), X.mul(G)), 0, eps);
    EXPECT_NEAR(computeECC(X.mul(R), X.mul(B)), 0, eps);
    EXPECT_NEAR(computeECC(X.mul(B), X.mul(G)), 0, eps);

    EXPECT_NEAR(computeECC(X.mul(R) + Y.mul(B), X.mul(B) + Y.mul(R)), 0, eps);

    EXPECT_NEAR(computeECC(X.mul(R) + Y.mul(G) + (X + Y).mul(B), Y.mul(R) + X.mul(G) + (X - Y).mul(B)), 0, eps);

    // 2. Each channel contribute equally
    EXPECT_NEAR(computeECC(X.mul(R) + Y.mul(G + B), X), 1.0 / 3, eps);
    EXPECT_NEAR(computeECC(X.mul(G) + Y.mul(R + B), X), 1.0 / 3, eps);
    EXPECT_NEAR(computeECC(X.mul(B) + Y.mul(G + R), X), 1.0 / 3, eps);

    // 3. The coefficient is invariant with respect to the offset of channels
    EXPECT_NEAR(computeECC(X - R + 2 * G + B, X), 1.0, eps);
    if (x.type() != CV_8U && x.type() != CV_8U) {
        EXPECT_NEAR(computeECC(X + R - 2 * G + B, Y), 0.0, eps);
    }

    // The channels are independent. Check orthogonal combinations
    // full squares norm = sum of squared norms
    EXPECT_NEAR(computeECC(X, Y + X), 1.0 / sqrt(2.0), eps);
    EXPECT_NEAR(computeECC(X, 2 * Y + X), 1.0 / sqrt(5.0), eps);
}

TEST(Video_ECC_Test_Compute, properties) {
    Mat xline(1, 100, CV_32F), x;
    for (int i = 0; i < xline.cols; ++i) xline.at<float>(0, i) =  (float)i;

    repeat(xline, xline.cols, 1, x);

    Mat x_f64, x_u8, x_u16;
    x.convertTo(x_f64, CV_64F);
    x.convertTo(x_u8, CV_8U);
    x.convertTo(x_u16, CV_16U);

    testECCProperties(x, 1e-5f);
    testECCProperties(x_f64, 1e-5f);
    testECCProperties(x_u8, 1);
    testECCProperties(x_u16, 1);
}

TEST(Video_ECC_Test_Compute, accuracy) {
    Mat testImg = (Mat_<float>(3, 3) << 1, 0, 0, 1, 0, 0, 1, 0, 0);
    Mat warpedImage = (Mat_<float>(3, 3) << 0, 1, 0, 0, 1, 0, 0, 1, 0);
    Mat_<unsigned char> mask = Mat_<unsigned char>::ones(testImg.rows, testImg.cols);
    double ecc = computeECC(warpedImage, testImg, mask);

    EXPECT_NEAR(ecc, -0.5f, 1e-5f);
}

TEST(Video_ECC_Test_Compute, bug_14657) {
    /*
     * Simple test case - a 2 x 2 matrix with 10, 10, 10, 6. When the mean (36 / 4 = 9) is subtracted,
     * it results in 1, 1, 1, 0 for the unsigned int case - compare to  1, 1, 1, -3 in the signed case.
     * For this reason, when the same matrix was provided as the input and the template, we didn't get 1 as expected.
     */
    Mat img = (Mat_<uint8_t>(2, 2) << 10, 10, 10, 6);
    EXPECT_NEAR(computeECC(img, img), 1.0f, 1e-5f);
}

TEST(Video_ECC_Mask, accuracy) {
    CV_ECC_Test_Mask test;
    test.safe_run();
}
TEST(Video_ECC_BigMS, accuracy) {
    CV_ECC_BigPictureTest test(false);
    test.safe_run();
}
TEST(Video_ECC_BigMS_Mask, accuracy) {
    CV_ECC_BigPictureTest test(true);
    test.safe_run();
}
}  // namespace
}  // namespace opencv_test
