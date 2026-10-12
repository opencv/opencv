// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/objdetect/barcode.hpp"

namespace opencv_test { namespace {

TEST(Objdetect_Barcode_Encode, reference_patterns_ean13)
{
    barcode::BarcodeEncoder::Params params;
    params.type = barcode::BarcodeEncoder::EAN_13;
    params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
    params.module_width = 1;
    params.bar_height = 40;

    Ptr<barcode::BarcodeEncoder> encoder = barcode::BarcodeEncoder::create(params);
    ASSERT_FALSE(encoder.empty());

    Mat img;
    encoder->encode("629104150021", img); // Expected check digit: 3 -> full code: 6291041500213

    ASSERT_EQ(CV_8UC1, img.type());
    ASSERT_EQ(40, img.rows);
    ASSERT_EQ(113, img.cols); // 11 (left quiet) + 95 (symbol) + 7 (right quiet)

    // Left quiet zone: 11 white modules (255)
    for (int col = 0; col < 11; ++col)
    {
        EXPECT_EQ(255, img.at<uchar>(20, col)) << "col " << col;
    }

    // Start pattern: 101 (black, white, black)
    EXPECT_EQ(0, img.at<uchar>(20, 11));
    EXPECT_EQ(255, img.at<uchar>(20, 12));
    EXPECT_EQ(0, img.at<uchar>(20, 13));

    // Center guard pattern at module 11 + 3 + 42 = 56: 01010
    EXPECT_EQ(255, img.at<uchar>(20, 56));
    EXPECT_EQ(0, img.at<uchar>(20, 57));
    EXPECT_EQ(255, img.at<uchar>(20, 58));
    EXPECT_EQ(0, img.at<uchar>(20, 59));
    EXPECT_EQ(255, img.at<uchar>(20, 60));

    // Stop pattern at module 11 + 3 + 42 + 5 + 42 = 103: 101
    EXPECT_EQ(0, img.at<uchar>(20, 103));
    EXPECT_EQ(255, img.at<uchar>(20, 104));
    EXPECT_EQ(0, img.at<uchar>(20, 105));

    // Right quiet zone: 7 white modules (106 to 112)
    for (int col = 106; col < 113; ++col)
    {
        EXPECT_EQ(255, img.at<uchar>(20, col)) << "col " << col;
    }

    // Also test VALIDATE_CHECKSUM with 13 digits
    params.checksum_policy = barcode::BarcodeEncoder::VALIDATE_CHECKSUM;
    Ptr<barcode::BarcodeEncoder> encoder_val = barcode::BarcodeEncoder::create(params);
    Mat img_val;
    encoder_val->encode("6291041500213", img_val);

    Mat diff;
    absdiff(img, img_val, diff);
    EXPECT_EQ(0, countNonZero(diff));
}

TEST(Objdetect_Barcode_Encode, reference_patterns_ean8)
{
    barcode::BarcodeEncoder::Params params;
    params.type = barcode::BarcodeEncoder::EAN_8;
    params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
    params.module_width = 1;
    params.bar_height = 40;

    Ptr<barcode::BarcodeEncoder> encoder = barcode::BarcodeEncoder::create(params);
    ASSERT_FALSE(encoder.empty());

    Mat img;
    encoder->encode("4017072", img); // Expected check digit: 5 -> full code: 40170725

    ASSERT_EQ(CV_8UC1, img.type());
    ASSERT_EQ(40, img.rows);
    ASSERT_EQ(81, img.cols); // 7 (left quiet) + 67 (symbol) + 7 (right quiet)

    // Left quiet zone: 7 white modules
    for (int col = 0; col < 7; ++col)
    {
        EXPECT_EQ(255, img.at<uchar>(20, col)) << "col " << col;
    }

    // Start pattern: 101
    EXPECT_EQ(0, img.at<uchar>(20, 7));
    EXPECT_EQ(255, img.at<uchar>(20, 8));
    EXPECT_EQ(0, img.at<uchar>(20, 9));

    // Center guard pattern at 7 + 3 + 28 = 38: 01010
    EXPECT_EQ(255, img.at<uchar>(20, 38));
    EXPECT_EQ(0, img.at<uchar>(20, 39));
    EXPECT_EQ(255, img.at<uchar>(20, 40));
    EXPECT_EQ(0, img.at<uchar>(20, 41));
    EXPECT_EQ(255, img.at<uchar>(20, 42));

    // Stop pattern at 7 + 3 + 28 + 5 + 28 = 71: 101
    EXPECT_EQ(0, img.at<uchar>(20, 71));
    EXPECT_EQ(255, img.at<uchar>(20, 72));
    EXPECT_EQ(0, img.at<uchar>(20, 73));

    // Right quiet zone: 7 white modules (74 to 80)
    for (int col = 74; col < 81; ++col)
    {
        EXPECT_EQ(255, img.at<uchar>(20, col)) << "col " << col;
    }

    // Test VALIDATE_CHECKSUM with 8 digits
    params.checksum_policy = barcode::BarcodeEncoder::VALIDATE_CHECKSUM;
    Ptr<barcode::BarcodeEncoder> encoder_val = barcode::BarcodeEncoder::create(params);
    Mat img_val;
    encoder_val->encode("40170725", img_val);

    Mat diff;
    absdiff(img, img_val, diff);
    EXPECT_EQ(0, countNonZero(diff));
}

TEST(Objdetect_Barcode_Encode, reference_patterns_upca)
{
    barcode::BarcodeEncoder::Params params;
    params.type = barcode::BarcodeEncoder::UPC_A;
    params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
    params.module_width = 1;
    params.bar_height = 40;

    Ptr<barcode::BarcodeEncoder> encoder = barcode::BarcodeEncoder::create(params);
    ASSERT_FALSE(encoder.empty());

    Mat img;
    encoder->encode("01234567890", img); // Expected check digit: 5 -> full code: 012345678905

    ASSERT_EQ(CV_8UC1, img.type());
    ASSERT_EQ(40, img.rows);
    ASSERT_EQ(113, img.cols); // 9 (left quiet) + 95 (symbol) + 9 (right quiet)

    // Left quiet zone: 9 white modules
    for (int col = 0; col < 9; ++col)
    {
        EXPECT_EQ(255, img.at<uchar>(20, col)) << "col " << col;
    }

    // Start pattern: 101
    EXPECT_EQ(0, img.at<uchar>(20, 9));
    EXPECT_EQ(255, img.at<uchar>(20, 10));
    EXPECT_EQ(0, img.at<uchar>(20, 11));

    // Center guard pattern at 9 + 3 + 42 = 54: 01010
    EXPECT_EQ(255, img.at<uchar>(20, 54));
    EXPECT_EQ(0, img.at<uchar>(20, 55));
    EXPECT_EQ(255, img.at<uchar>(20, 56));
    EXPECT_EQ(0, img.at<uchar>(20, 57));
    EXPECT_EQ(255, img.at<uchar>(20, 58));

    // Stop pattern at 9 + 3 + 42 + 5 + 42 = 101: 101
    EXPECT_EQ(0, img.at<uchar>(20, 101));
    EXPECT_EQ(255, img.at<uchar>(20, 102));
    EXPECT_EQ(0, img.at<uchar>(20, 103));

    // Right quiet zone: 9 white modules (104 to 112)
    for (int col = 104; col < 113; ++col)
    {
        EXPECT_EQ(255, img.at<uchar>(20, col)) << "col " << col;
    }

    // Test VALIDATE_CHECKSUM with 12 digits
    params.checksum_policy = barcode::BarcodeEncoder::VALIDATE_CHECKSUM;
    Ptr<barcode::BarcodeEncoder> encoder_val = barcode::BarcodeEncoder::create(params);
    Mat img_val;
    encoder_val->encode("012345678905", img_val);

    Mat diff;
    absdiff(img, img_val, diff);
    EXPECT_EQ(0, countNonZero(diff));
}

TEST(Objdetect_Barcode_Encode, leading_zeros)
{
    // EAN-13 with leading zeros
    {
        barcode::BarcodeEncoder::Params params;
        params.type = barcode::BarcodeEncoder::EAN_13;
        params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
        auto encoder = barcode::BarcodeEncoder::create(params);
        Mat img;
        EXPECT_NO_THROW(encoder->encode("001234567890", img));
    }

    // UPC-A with leading zeros
    {
        barcode::BarcodeEncoder::Params params;
        params.type = barcode::BarcodeEncoder::UPC_A;
        params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
        auto encoder = barcode::BarcodeEncoder::create(params);
        Mat img;
        EXPECT_NO_THROW(encoder->encode("00123456789", img));
    }

    // EAN-8 with leading zeros
    {
        barcode::BarcodeEncoder::Params params;
        params.type = barcode::BarcodeEncoder::EAN_8;
        params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
        auto encoder = barcode::BarcodeEncoder::create(params);
        Mat img;
        EXPECT_NO_THROW(encoder->encode("0001234", img));
    }
}

TEST(Objdetect_Barcode_Encode, invalid_inputs)
{
    barcode::BarcodeEncoder::Params params;
    params.type = barcode::BarcodeEncoder::EAN_13;
    params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
    auto encoder = barcode::BarcodeEncoder::create(params);
    Mat img;

    // Empty input
    EXPECT_THROW(encoder->encode("", img), cv::Exception);

    // Non-numeric characters
    EXPECT_THROW(encoder->encode("62910415002a", img), cv::Exception);
    EXPECT_THROW(encoder->encode("6291-4150021", img), cv::Exception);
    EXPECT_THROW(encoder->encode(" 62910415002", img), cv::Exception);

    // Invalid length for APPEND_CHECKSUM (expected 12 for EAN-13)
    EXPECT_THROW(encoder->encode("62910415002", img), cv::Exception); // 11
    EXPECT_THROW(encoder->encode("6291041500213", img), cv::Exception); // 13

    // Invalid length for VALIDATE_CHECKSUM (expected 13 for EAN-13)
    params.checksum_policy = barcode::BarcodeEncoder::VALIDATE_CHECKSUM;
    auto encoder_val = barcode::BarcodeEncoder::create(params);
    EXPECT_THROW(encoder_val->encode("629104150021", img), cv::Exception); // 12
    EXPECT_THROW(encoder_val->encode("62910415002133", img), cv::Exception); // 14

    // Mismatched checksum in VALIDATE_CHECKSUM (expected 3, gave 9)
    EXPECT_THROW(encoder_val->encode("6291041500219", img), cv::Exception);

    // EAN-8 invalid lengths
    params.type = barcode::BarcodeEncoder::EAN_8;
    params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
    auto enc_ean8 = barcode::BarcodeEncoder::create(params);
    EXPECT_THROW(enc_ean8->encode("123456", img), cv::Exception); // 6
    EXPECT_THROW(enc_ean8->encode("12345678", img), cv::Exception); // 8

    params.checksum_policy = barcode::BarcodeEncoder::VALIDATE_CHECKSUM;
    auto enc_ean8_val = barcode::BarcodeEncoder::create(params);
    EXPECT_THROW(enc_ean8_val->encode("40170729", img), cv::Exception); // wrong checksum

    // UPC-A invalid lengths
    params.type = barcode::BarcodeEncoder::UPC_A;
    params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
    auto enc_upca = barcode::BarcodeEncoder::create(params);
    EXPECT_THROW(enc_upca->encode("1234567890", img), cv::Exception); // 10
    EXPECT_THROW(enc_upca->encode("123456789012", img), cv::Exception); // 12

    params.checksum_policy = barcode::BarcodeEncoder::VALIDATE_CHECKSUM;
    auto enc_upca_val = barcode::BarcodeEncoder::create(params);
    EXPECT_THROW(enc_upca_val->encode("012345678909", img), cv::Exception); // wrong checksum

    // Invalid module_width and bar_height
    params.module_width = 0;
    auto enc_bad_mw = barcode::BarcodeEncoder::create(params);
    EXPECT_THROW(enc_bad_mw->encode("012345678905", img), cv::Exception);

    params.module_width = 2;
    params.bar_height = -5;
    auto enc_bad_bh = barcode::BarcodeEncoder::create(params);
    EXPECT_THROW(enc_bad_bh->encode("012345678905", img), cv::Exception);

    // Invalid type
    params.bar_height = 50;
    params.type = barcode::BarcodeEncoder::NONE;
    auto enc_none = barcode::BarcodeEncoder::create(params);
    EXPECT_THROW(enc_none->encode("012345678905", img), cv::Exception);
}

TEST(Objdetect_Barcode_Encode, scaling_and_geometry)
{
    barcode::BarcodeEncoder::Params params;
    params.type = barcode::BarcodeEncoder::EAN_13;
    params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;

    for (int mw : {1, 2, 3, 5})
    {
        for (int bh : {25, 60, 90})
        {
            params.module_width = mw;
            params.bar_height = bh;
            auto encoder = barcode::BarcodeEncoder::create(params);
            Mat img;
            encoder->encode("629104150021", img);

            EXPECT_EQ(CV_8UC1, img.type());
            EXPECT_EQ(bh, img.rows);
            EXPECT_EQ(113 * mw, img.cols);
        }
    }
}

TEST(Objdetect_Barcode_Encode, decode_roundtrip_corners)
{
    barcode::BarcodeDetector detector;

    // EAN-13
    {
        barcode::BarcodeEncoder::Params params;
        params.type = barcode::BarcodeEncoder::EAN_13;
        params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
        params.module_width = 2;
        params.bar_height = 80;
        auto encoder = barcode::BarcodeEncoder::create(params);

        Mat img;
        encoder->encode("629104150021", img);

        std::vector<Point2f> corners = {
            Point2f(0.f, static_cast<float>(img.rows - 1)),
            Point2f(0.f, 0.f),
            Point2f(static_cast<float>(img.cols - 1), 0.f),
            Point2f(static_cast<float>(img.cols - 1), static_cast<float>(img.rows - 1))
        };
        std::vector<std::string> decoded_info, decoded_type;
        bool ok = detector.decodeWithType(img, corners, decoded_info, decoded_type);

        ASSERT_TRUE(ok);
        ASSERT_EQ(1u, decoded_info.size());
        EXPECT_EQ("6291041500213", decoded_info[0]);
        EXPECT_EQ("EAN_13", decoded_type[0]);
    }

    // EAN-8
    {
        barcode::BarcodeEncoder::Params params;
        params.type = barcode::BarcodeEncoder::EAN_8;
        params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
        params.module_width = 2;
        params.bar_height = 80;
        auto encoder = barcode::BarcodeEncoder::create(params);

        Mat img;
        encoder->encode("4017072", img);

        std::vector<Point2f> corners = {
            Point2f(0.f, static_cast<float>(img.rows - 1)),
            Point2f(0.f, 0.f),
            Point2f(static_cast<float>(img.cols - 1), 0.f),
            Point2f(static_cast<float>(img.cols - 1), static_cast<float>(img.rows - 1))
        };
        std::vector<std::string> decoded_info, decoded_type;
        bool ok = detector.decodeWithType(img, corners, decoded_info, decoded_type);

        ASSERT_TRUE(ok);
        ASSERT_EQ(1u, decoded_info.size());
        EXPECT_EQ("40170725", decoded_info[0]);
        EXPECT_EQ("EAN_8", decoded_type[0]);
    }

    // UPC-A
    {
        barcode::BarcodeEncoder::Params params;
        params.type = barcode::BarcodeEncoder::UPC_A;
        params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
        params.module_width = 2;
        params.bar_height = 80;
        auto encoder = barcode::BarcodeEncoder::create(params);

        Mat img;
        encoder->encode("01234567890", img);

        std::vector<Point2f> corners = {
            Point2f(0.f, static_cast<float>(img.rows - 1)),
            Point2f(0.f, 0.f),
            Point2f(static_cast<float>(img.cols - 1), 0.f),
            Point2f(static_cast<float>(img.cols - 1), static_cast<float>(img.rows - 1))
        };
        std::vector<std::string> decoded_info, decoded_type;
        bool ok = detector.decodeWithType(img, corners, decoded_info, decoded_type);

        ASSERT_TRUE(ok);
        ASSERT_EQ(1u, decoded_info.size());
        EXPECT_EQ("012345678905", decoded_info[0]);
        EXPECT_EQ("UPC_A", decoded_type[0]);
    }
}

}} // opencv_test::<anonymous>

