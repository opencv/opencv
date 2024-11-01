// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_JPEGXL

typedef testing::TestWithParam<perf::MatType> Imgcodecs_JpegXL_MatType;

TEST_P(Imgcodecs_JpegXL_MatType, write_read)
{
    const int matType  = GetParam();

    cv::Scalar col;
    // Jpeg XL is lossy compression.
    // There may be small differences in decoding results by environments.
    double th;

    switch( CV_MAT_DEPTH(matType) )
    {
        case CV_16U:
            col = cv::Scalar(124 * 255, 76 * 255, 42 * 255 );
            th = 656; // = 65535 / 100;
            break;
        case CV_32F : 
            col = cv::Scalar(0.486, 0.298, 0.165);
            th = 1.0 / 100.0;
            break;
        default:
        case CV_8U:
            col = cv::Scalar(124,76,42);
            th = 3; // = 255 / 100 (1%);
            break;
    }

    bool ret = false;
    string tmp_fname = cv::tempfile(".jxl");
    Mat img_org(320, 480, matType, col);
    vector<int> param;
    EXPECT_NO_THROW(ret = imwrite(tmp_fname, img_org, param));
    EXPECT_TRUE(ret);
    Mat img_decoded;
    EXPECT_NO_THROW(img_decoded = imread(tmp_fname, IMREAD_UNCHANGED));
    EXPECT_FALSE(img_decoded.empty());

    EXPECT_LE(cvtest::norm(img_org, img_decoded, NORM_INF), th);

    EXPECT_EQ(0, remove(tmp_fname.c_str()));
}

TEST_P(Imgcodecs_JpegXL_MatType, encode_decode)
{
    const int matType  = GetParam();

    cv::Scalar col;
    // Jpeg XL is lossy compression.
    // There may be small differences in decoding results by environments.
    double th;

    switch( CV_MAT_DEPTH(matType) )
    {
        case CV_16U:
            col = cv::Scalar(124 * 255, 76 * 255, 42 * 255 );
            th = 656; // = 65535 / 100;
            break;
        case CV_32F : 
            col = cv::Scalar(0.486, 0.298, 0.165);
            th = 1.0 / 100.0;
            break;
        default:
        case CV_8U:
            col = cv::Scalar(124,76,42);
            th = 3; // = 255 / 100 (1%);
            break;
    }

    bool ret = false;
    vector<uchar> buff;
    Mat img_org(320, 480, matType, col);
    vector<int> param;
    EXPECT_NO_THROW(ret = imencode(".jxl", img_org, buff, param));
    EXPECT_TRUE(ret);
    Mat img_decoded;
    EXPECT_NO_THROW(img_decoded = imdecode(buff, IMREAD_UNCHANGED));
    EXPECT_FALSE(img_decoded.empty());

    EXPECT_LE(cvtest::norm(img_org, img_decoded, NORM_INF), th);
}

INSTANTIATE_TEST_CASE_P(
    /**/,
    Imgcodecs_JpegXL_MatType,
    testing::Values(
        CV_8UC1,  CV_8UC3,  // CV_8UC4,
        CV_16UC1, CV_16UC3, // CV_16UC4,
        CV_32FC1, CV_32FC3  // CV_32FC4,
    ) );

#endif  // HAVE_JPEGXL

}  // namespace
}  // namespace opencv_test
