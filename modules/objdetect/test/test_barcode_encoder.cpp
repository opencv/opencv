// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/objdetect/barcode.hpp"
#include <climits>

namespace opencv_test { namespace {

static std::string barcodeMatToBitString(const Mat& img)
{
    CV_Assert(!img.empty() && img.type() == CV_8UC1);
    std::string bits;
    bits.reserve(img.cols);
    for (int col = 0; col < img.cols; ++col)
    {
        // In generated barcode: 0 is black bar ('1'), 255 is white space ('0')
        uchar val = img.at<uchar>(0, col);
        bits += (val == 0) ? '1' : '0';
    }
    // Verify vertical consistency across all rows
    for (int row = 1; row < img.rows; ++row)
    {
        for (int col = 0; col < img.cols; ++col)
        {
            CV_Assert(img.at<uchar>(row, col) == img.at<uchar>(0, col));
        }
    }
    return bits;
}

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

    // Assert the complete known GS1 module bitstream (113 modules) independently verified:
    // First digit 6 -> parity ABBBAA
    // Left:  2(B: 0011011), 9(B: 0010111), 1(B: 0110011), 0(B: 0100111), 4(A: 0100011), 1(A: 0011001)
    // Right: 5(C: 1001110), 0(C: 1110010), 0(C: 1110010), 2(C: 1101100), 1(C: 1100110), 3(C: 1000010)
    const std::string expected_ean13_bits =
        "00000000000" // 11 quiet
        "101"         // start guard
        "0010011"     // '2' (B)
        "0010111"     // '9' (B)
        "0110011"     // '1' (B)
        "0100111"     // '0' (B)
        "0100011"     // '4' (A)
        "0011001"     // '1' (A)
        "01010"       // center guard
        "1001110"     // '5' (C)
        "1110010"     // '0' (C)
        "1110010"     // '0' (C)
        "1101100"     // '2' (C)
        "1100110"     // '1' (C)
        "1000010"     // '3' (C, check digit)
        "101"         // stop guard
        "0000000";    // 7 quiet
    EXPECT_EQ(expected_ean13_bits, barcodeMatToBitString(img));

    // Also assert second independent reference EAN-13 code (9787115279460)
    Mat img_book;
    encoder->encode("978711527946", img_book); // Check digit 0
    const std::string expected_ean13_book_bits =
        "00000000000" // 11 quiet
        "101"         // start guard
        "0111011"     // '7' (A parity for first digit 9)
        "0001001"     // '8' (B)
        "0010001"     // '7' (B)
        "0011001"     // '1' (A)
        "0110011"     // '1' (B)
        "0110001"     // '5' (A)
        "01010"       // center guard
        "1101100"     // '2' (C)
        "1000100"     // '7' (C)
        "1110100"     // '9' (C)
        "1011100"     // '4' (C)
        "1010000"     // '6' (C)
        "1110010"     // '0' (C, check digit)
        "101"         // stop guard
        "0000000";    // 7 quiet
    EXPECT_EQ(expected_ean13_book_bits, barcodeMatToBitString(img_book));

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

    // Assert the complete known GS1 module bitstream (81 modules) independently verified:
    // Left (Set A): 4(0100011), 0(0001101), 1(0011001), 7(0111011)
    // Right (Set C): 0(1110010), 7(1000100), 2(1101100), 5(1001110)
    const std::string expected_ean8_bits =
        "0000000"  // 7 quiet
        "101"      // start guard
        "0100011"  // '4' (A)
        "0001101"  // '0' (A)
        "0011001"  // '1' (A)
        "0111011"  // '7' (A)
        "01010"    // center guard
        "1110010"  // '0' (C)
        "1000100"  // '7' (C)
        "1101100"  // '2' (C)
        "1001110"  // '5' (C, check digit)
        "101"      // stop guard
        "0000000"; // 7 quiet
    EXPECT_EQ(expected_ean8_bits, barcodeMatToBitString(img));

    // Assert second independent reference EAN-8 (96385074)
    Mat img_ean8_2;
    encoder->encode("9638507", img_ean8_2); // Check digit 4
    const std::string expected_ean8_2_bits =
        "0000000"  // 7 quiet
        "101"      // start guard
        "0001011"  // '9' (A)
        "0101111"  // '6' (A)
        "0111101"  // '3' (A)
        "0110111"  // '8' (A)
        "01010"    // center guard
        "1001110"  // '5' (C)
        "1110010"  // '0' (C)
        "1000100"  // '7' (C)
        "1011100"  // '4' (C, check digit)
        "101"      // stop guard
        "0000000"; // 7 quiet
    EXPECT_EQ(expected_ean8_2_bits, barcodeMatToBitString(img_ean8_2));

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

    // Assert the complete known GS1 module bitstream (113 modules) independently verified:
    // Left (Set A):  0(0001101), 1(0011001), 2(0010011), 3(0111101), 4(0100011), 5(0110001)
    // Right (Set C): 6(1010000), 7(1000100), 8(1001000), 9(1110100), 0(1110010), 5(1001110)
    const std::string expected_upca_bits =
        "000000000" // 9 quiet
        "101"       // start guard
        "0001101"   // '0' (A)
        "0011001"   // '1' (A)
        "0010011"   // '2' (A)
        "0111101"   // '3' (A)
        "0100011"   // '4' (A)
        "0110001"   // '5' (A)
        "01010"     // center guard
        "1010000"   // '6' (C)
        "1000100"   // '7' (C)
        "1001000"   // '8' (C)
        "1110100"   // '9' (C)
        "1110010"   // '0' (C)
        "1001110"   // '5' (C, check digit)
        "101"       // stop guard
        "000000000";// 9 quiet
    EXPECT_EQ(expected_upca_bits, barcodeMatToBitString(img));

    // Assert second independent reference UPC-A (036000291452 - Wrigley's chewing gum)
    Mat img_upca_2;
    encoder->encode("03600029145", img_upca_2); // Check digit 2
    const std::string expected_upca_2_bits =
        "000000000" // 9 quiet
        "101"       // start guard
        "0001101"   // '0' (A)
        "0111101"   // '3' (A)
        "0101111"   // '6' (A)
        "0001101"   // '0' (A)
        "0001101"   // '0' (A)
        "0001101"   // '0' (A)
        "01010"     // center guard
        "1101100"   // '2' (C)
        "1110100"   // '9' (C)
        "1100110"   // '1' (C)
        "1011100"   // '4' (C)
        "1001110"   // '5' (C)
        "1101100"   // '2' (C, check digit)
        "101"       // stop guard
        "000000000";// 9 quiet
    EXPECT_EQ(expected_upca_2_bits, barcodeMatToBitString(img_upca_2));

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

TEST(Objdetect_Barcode_Encode, module_width_overflow_boundary)
{
    barcode::BarcodeEncoder::Params params;
    params.type = barcode::BarcodeEncoder::EAN_13;
    params.checksum_policy = barcode::BarcodeEncoder::APPEND_CHECKSUM;
    params.bar_height = 10;

    // EAN-13 has 113 total modules.
    // Boundary value: (INT_MAX / 113) + 1 must throw StsBadArg
    params.module_width = (INT_MAX / 113) + 1;
    auto encoder_ean13 = barcode::BarcodeEncoder::create(params);
    Mat img;
    EXPECT_THROW(encoder_ean13->encode("629104150021", img), cv::Exception);

    // INT_MAX must also throw StsBadArg
    params.module_width = INT_MAX;
    encoder_ean13 = barcode::BarcodeEncoder::create(params);
    EXPECT_THROW(encoder_ean13->encode("629104150021", img), cv::Exception);

    // UPC-A has 113 total modules
    params.type = barcode::BarcodeEncoder::UPC_A;
    params.module_width = (INT_MAX / 113) + 1;
    auto encoder_upca = barcode::BarcodeEncoder::create(params);
    EXPECT_THROW(encoder_upca->encode("01234567890", img), cv::Exception);

    // EAN-8 has 81 total modules
    params.type = barcode::BarcodeEncoder::EAN_8;
    params.module_width = (INT_MAX / 81) + 1;
    auto encoder_ean8 = barcode::BarcodeEncoder::create(params);
    EXPECT_THROW(encoder_ean8->encode("4017072", img), cv::Exception);
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

