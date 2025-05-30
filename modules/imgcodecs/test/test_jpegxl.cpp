// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_JPEGXL

typedef tuple<perf::MatType, int> MatType_and_Distance;
typedef testing::TestWithParam<MatType_and_Distance> Imgcodecs_JpegXL_MatType;

TEST_P(Imgcodecs_JpegXL_MatType, write_read)
{
    const int matType  = get<0>(GetParam());
    const int distanceParam = get<1>(GetParam());

    cv::Scalar col;
    // Jpeg XL is lossy compression.
    // There may be small differences in decoding results by environments.
    double th;

    switch( CV_MAT_DEPTH(matType) )
    {
        case CV_16U:
            col = cv::Scalar(124 * 256, 76 * 256, 42 * 256, 192 * 256 );
            th = 656; // = 65535 / 100;
            break;
        case CV_32F:
            col = cv::Scalar(0.486, 0.298, 0.165, 0.75);
            th = 1.0 / 100.0;
            break;
        default:
        case CV_8U:
            col = cv::Scalar(124, 76, 42, 192);
            th = 3; // = 255 / 100 (1%);
            break;
    }

    // If increasing distanceParam, threshold should be increased.
    th *= (distanceParam >= 25) ? 5 : ( distanceParam > 2 ) ? 3 : (distanceParam == 2) ? 2: 1;

    bool ret = false;
    string tmp_fname = cv::tempfile(".jxl");
    Mat img_org(320, 480, matType, col);
    vector<int> param;
    param.push_back(IMWRITE_JPEGXL_DISTANCE);
    param.push_back(distanceParam);
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
    const int matType  = get<0>(GetParam());
    const int distanceParam  = get<1>(GetParam());

    cv::Scalar col;
    // Jpeg XL is lossy compression.
    // There may be small differences in decoding results by environments.
    double th;

    // If alpha=0, libjxl modify color channels(BGR). So do not set it.
    switch( CV_MAT_DEPTH(matType) )
    {
        case CV_16U:
            col = cv::Scalar(124 * 256, 76 * 256, 42 * 256, 192 * 256 );
            th = 656; // = 65535 / 100;
            break;
        case CV_32F:
            col = cv::Scalar(0.486, 0.298, 0.165, 0.75);
            th = 1.0 / 100.0;
            break;
        default:
        case CV_8U:
            col = cv::Scalar(124, 76, 42, 192);
            th = 3; // = 255 / 100 (1%);
            break;
    }

    // If increasing distanceParam, threshold should be increased.
    th *= (distanceParam >= 25) ? 5 : ( distanceParam > 2 ) ? 3 : (distanceParam == 2) ? 2: 1;

    bool ret = false;
    vector<uchar> buff;
    Mat img_org(320, 480, matType, col);
    vector<int> param;
    param.push_back(IMWRITE_JPEGXL_DISTANCE);
    param.push_back(distanceParam);
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
    testing::Combine(
        testing::Values(
            CV_8UC1,  CV_8UC3,  CV_8UC4,
            CV_16UC1, CV_16UC3, CV_16UC4,
            CV_32FC1, CV_32FC3, CV_32FC4
        ),
        testing::Values( // Distance
            0, // Lossless
            1, // Default
            3, // Recomended Lossy Max
            25 // Specification Max
        )
) );


typedef tuple<int, int> Effort_and_Decoding_speed;
typedef testing::TestWithParam<Effort_and_Decoding_speed> Imgcodecs_JpegXL_Effort_DecodingSpeed;

TEST_P(Imgcodecs_JpegXL_Effort_DecodingSpeed, encode_decode)
{
    const int effort = get<0>(GetParam());
    const int speed  = get<1>(GetParam());

    cv::Scalar col = cv::Scalar(124,76,42);
    // Jpeg XL is lossy compression.
    // There may be small differences in decoding results by environments.
    double th = 3; // = 255 / 100 (1%);

    bool ret = false;
    vector<uchar> buff;
    Mat img_org(320, 480, CV_8UC3, col);
    vector<int> param;
    param.push_back(IMWRITE_JPEGXL_EFFORT);
    param.push_back(effort);
    param.push_back(IMWRITE_JPEGXL_DECODING_SPEED);
    param.push_back(speed);
    EXPECT_NO_THROW(ret = imencode(".jxl", img_org, buff, param));
    EXPECT_TRUE(ret);
    Mat img_decoded;
    EXPECT_NO_THROW(img_decoded = imdecode(buff, IMREAD_UNCHANGED));
    EXPECT_FALSE(img_decoded.empty());

    EXPECT_LE(cvtest::norm(img_org, img_decoded, NORM_INF), th);
}

INSTANTIATE_TEST_CASE_P(
    /**/,
    Imgcodecs_JpegXL_Effort_DecodingSpeed,
    testing::Combine(
        testing::Values( // Effort
            1,  // fastest
            7,  // default
            9   // slowest
        ),
        testing::Values( // Decoding Speed
            0,  // default, slowest, and best quality/density
            2,
            4   // fastest, at the cost of some qulity/density
        )
) );

TEST(Imgcodecs_JpegXL, encode_from_uncontinued_image)
{
    cv::Mat src(100, 100, CV_8UC1, Scalar(40,50,10));
    cv::Mat roi = src(cv::Rect(10,20,30,50));
    EXPECT_FALSE(roi.isContinuous()); // uncontinued image

    vector<uint8_t> buff;
    vector<int> param;
    bool ret = false;
    EXPECT_NO_THROW(ret = cv::imencode(".jxl", roi, buff, param));
    EXPECT_TRUE(ret);
}

// See https://github.com/opencv/opencv/issues/26767

typedef tuple<perf::MatType, ImreadModes> MatType_and_ImreadFlag;
typedef testing::TestWithParam<MatType_and_ImreadFlag> Imgcodecs_JpegXL_MatType_ImreadFlag;

TEST_P(Imgcodecs_JpegXL_MatType_ImreadFlag, all_imreadFlags)
{
    string tmp_fname = cv::tempfile(".jxl");
    const int matType  = get<0>(GetParam());
    const int imreadFlag  = get<1>(GetParam());

    Mat img(240, 320, matType);
    randu(img, Scalar(0, 0, 0, 255), Scalar(255, 255, 255, 255));

    vector<int> param;
    param.push_back(IMWRITE_JPEGXL_DISTANCE);
    param.push_back(0 /* Lossless */);
    EXPECT_NO_THROW(imwrite(tmp_fname, img, param));

    Mat img_decoded;
    EXPECT_NO_THROW(img_decoded = imread(tmp_fname, imreadFlag));
    EXPECT_FALSE(img_decoded.empty());

    switch( imreadFlag )
    {
        case IMREAD_UNCHANGED:
            EXPECT_EQ( img.type(), img_decoded.type() );
            break;
        case IMREAD_GRAYSCALE:
            EXPECT_EQ( img_decoded.depth(), CV_8U );
            EXPECT_EQ( img_decoded.channels(), 1 );
            break;
        case IMREAD_COLOR:
        case IMREAD_COLOR_RGB:
            EXPECT_EQ( img_decoded.depth(), CV_8U );
            EXPECT_EQ( img_decoded.channels(), 3 );
            break;
        case IMREAD_ANYDEPTH:
            EXPECT_EQ( img_decoded.depth(), img.depth() );
            EXPECT_EQ( img_decoded.channels(), 1 );
            break;
        case IMREAD_ANYCOLOR:
            EXPECT_EQ( img_decoded.depth(), CV_8U ) ;
            EXPECT_EQ( img_decoded.channels(), img.channels() == 1 ? 1 : 3 ); // Alpha channel will be dropped.
            break;
    }
    remove(tmp_fname.c_str());
}

INSTANTIATE_TEST_CASE_P(
    /**/,
    Imgcodecs_JpegXL_MatType_ImreadFlag,
    testing::Combine(
        testing::Values(
            CV_8UC1,  CV_8UC3,  CV_8UC4,
            CV_16UC1, CV_16UC3, CV_16UC4,
            CV_32FC1, CV_32FC3, CV_32FC4
        ),
        testing::Values(
            IMREAD_UNCHANGED,
            IMREAD_GRAYSCALE,
            IMREAD_COLOR,
            IMREAD_COLOR_RGB,
            IMREAD_ANYDEPTH,
            IMREAD_ANYCOLOR
        )
) );

TEST(Imgcodecs_JpegXL, imdecode_truncated_stream)
{
    cv::Mat src(100, 100, CV_8UC1, Scalar(40,50,10));
    vector<uint8_t> buff;
    vector<int> param;

    bool ret = false;
    EXPECT_NO_THROW(ret = cv::imencode(".jxl", src, buff, param));
    EXPECT_TRUE(ret);

    // Try to decode non-truncated image.
    cv::Mat decoded;
    EXPECT_NO_THROW(decoded = cv::imdecode(buff, cv::IMREAD_COLOR));
    EXPECT_FALSE(decoded.empty());

    // Try to decode truncated image.
    buff.resize(buff.size() - 1 );
    EXPECT_NO_THROW(decoded = cv::imdecode(buff, cv::IMREAD_COLOR));
    EXPECT_TRUE(decoded.empty());
}

TEST(Imgcodecs_JpegXL, imread_truncated_stream)
{
    string tmp_fname = cv::tempfile(".jxl");
    cv::Mat src(100, 100, CV_8UC1, Scalar(40,50,10));
    vector<uint8_t> buff;
    vector<int> param;

    bool ret = false;
    EXPECT_NO_THROW(ret = cv::imencode(".jxl", src, buff, param));
    EXPECT_TRUE(ret);

    // Try to decode non-truncated image.
    FILE *fp = nullptr;

    fp = fopen(tmp_fname.c_str(), "wb");
    EXPECT_TRUE(fp != nullptr);
    fwrite(&buff[0], sizeof(uint8_t), buff.size(), fp);
    fclose(fp);

    cv::Mat decoded;
    EXPECT_NO_THROW(decoded = cv::imread(tmp_fname, cv::IMREAD_COLOR));
    EXPECT_FALSE(decoded.empty());

    // Try to decode truncated image.
    fp = fopen(tmp_fname.c_str(), "wb");
    EXPECT_TRUE(fp != nullptr);
    fwrite(&buff[0], sizeof(uint8_t), buff.size() - 1, fp);
    fclose(fp);

    EXPECT_NO_THROW(decoded = cv::imread(tmp_fname, cv::IMREAD_COLOR));
    EXPECT_TRUE(decoded.empty());

    // Delete temporary file
    remove(tmp_fname.c_str());
}

// See https://github.com/opencv/opencv/issues/27382
TEST(Imgcodecs_JpegXL, imencode_regression27382)
{
    cv::Mat image(1024, 1024, CV_16U);
    cv::RNG rng(1024);
    rng.fill(image, cv::RNG::NORMAL, 0, 65535);

    std::vector<unsigned char> buffer;
    std::vector<int> params = {cv::IMWRITE_JPEGXL_DISTANCE, 0}; // lossless

    EXPECT_NO_THROW(cv::imencode(".jxl", image, buffer, params));

    cv::Mat decoded;
    EXPECT_NO_THROW(decoded = cv::imdecode(buffer, cv::IMREAD_UNCHANGED));
    EXPECT_FALSE(decoded.empty());

    cv::Mat diff;
    cv::absdiff(image, decoded, diff);
    double max_diff = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_diff);
    EXPECT_EQ(max_diff, 0 );
}
TEST(Imgcodecs_JpegXL, imencode_regression27382_2)
{
    cv::Mat image(1024, 1024, CV_16U);
    cv::RNG rng(1024);
    rng.fill(image, cv::RNG::NORMAL, 0, 65535);

    std::vector<unsigned char> buffer;
    std::vector<int> params = {cv::IMWRITE_JPEGXL_QUALITY, 100}; // lossless

    EXPECT_NO_THROW(cv::imencode(".jxl", image, buffer, params));

    cv::Mat decoded;
    EXPECT_NO_THROW(decoded = cv::imdecode(buffer, cv::IMREAD_UNCHANGED));
    EXPECT_FALSE(decoded.empty());

    cv::Mat diff;
    cv::absdiff(image, decoded, diff);
    double max_diff = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_diff);
    EXPECT_EQ(max_diff, 0 );
}


#endif  // HAVE_JPEGXL

}  // namespace
}  // namespace opencv_test
