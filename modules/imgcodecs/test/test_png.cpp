// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"
#include "test_common.hpp"

namespace opencv_test { namespace {

#if defined(HAVE_PNG) || defined(HAVE_SPNG)

TEST(Imgcodecs_Png, write_big)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/read.png";
    const string dst_file = cv::tempfile(".png");
    Mat img;
    ASSERT_NO_THROW(img = imread(filename));
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(13043, img.cols);
    EXPECT_EQ(13917, img.rows);
    ASSERT_NO_THROW(imwrite(dst_file, img));
    EXPECT_EQ(0, remove(dst_file.c_str()));
}

TEST(Imgcodecs_Png, encode)
{
    vector<uchar> buff;
    Mat img_gt = Mat::zeros(1000, 1000, CV_8U);
    vector<int> param;
    param.push_back(IMWRITE_PNG_COMPRESSION);
    param.push_back(3); //default(3) 0-9.
    EXPECT_NO_THROW(imencode(".png", img_gt, buff, param));
    Mat img;
    EXPECT_NO_THROW(img = imdecode(buff, IMREAD_ANYDEPTH)); // hang
    EXPECT_FALSE(img.empty());
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img, img_gt);
}

TEST(Imgcodecs_Png, regression_ImreadVSCvtColor)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string imgName = root + "../cv/shared/lena.png";
    Mat original_image = imread(imgName);
    Mat gray_by_codec = imread(imgName, IMREAD_GRAYSCALE);
    Mat gray_by_cvt;
    cvtColor(original_image, gray_by_cvt, COLOR_BGR2GRAY);

    Mat diff;
    absdiff(gray_by_codec, gray_by_cvt, diff);
    EXPECT_LT(cvtest::mean(diff)[0], 1.);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(10, 0), gray_by_codec, gray_by_cvt);
}

// Test OpenCV issue 3075 is solved
TEST(Imgcodecs_Png, read_color_palette_with_alpha)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    Mat img;

    // First Test : Read PNG with alpha, imread flag -1
    img = imread(root + "readwrite/color_palette_alpha.png", IMREAD_UNCHANGED);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 4);

    // pixel is red in BGRA
    EXPECT_EQ(img.at<Vec4b>(0, 0), Vec4b(0, 0, 255, 255));
    EXPECT_EQ(img.at<Vec4b>(0, 1), Vec4b(0, 0, 255, 255));

    // Second Test : Read PNG without alpha, imread flag -1
    img = imread(root + "readwrite/color_palette_no_alpha.png", IMREAD_UNCHANGED);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 3);

    // pixel is red in BGR
    EXPECT_EQ(img.at<Vec3b>(0, 0), Vec3b(0, 0, 255));
    EXPECT_EQ(img.at<Vec3b>(0, 1), Vec3b(0, 0, 255));

    // Third Test : Read PNG with alpha, imread flag 1
    img = imread(root + "readwrite/color_palette_alpha.png", IMREAD_COLOR);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 3);

    // pixel is red in BGR
    EXPECT_EQ(img.at<Vec3b>(0, 0), Vec3b(0, 0, 255));
    EXPECT_EQ(img.at<Vec3b>(0, 1), Vec3b(0, 0, 255));

    img = imread(root + "readwrite/color_palette_alpha.png", IMREAD_COLOR_RGB);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 3);

    // pixel is red in RGB
    EXPECT_EQ(img.at<Vec3b>(0, 0), Vec3b(255, 0, 0));
    EXPECT_EQ(img.at<Vec3b>(0, 1), Vec3b(255, 0, 0));

    // Fourth Test : Read PNG without alpha, imread flag 1
    img = imread(root + "readwrite/color_palette_no_alpha.png", IMREAD_COLOR);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 3);

    // pixel is red in BGR
    EXPECT_EQ(img.at<Vec3b>(0, 0), Vec3b(0, 0, 255));
    EXPECT_EQ(img.at<Vec3b>(0, 1), Vec3b(0, 0, 255));

    img = imread(root + "readwrite/color_palette_no_alpha.png", IMREAD_COLOR_RGB);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 3);

    // pixel is red in RGB
    EXPECT_EQ(img.at<Vec3b>(0, 0), Vec3b(255, 0, 0));
    EXPECT_EQ(img.at<Vec3b>(0, 1), Vec3b(255, 0, 0));
}

typedef testing::TestWithParam<string> Imgcodecs_Png_PngSuite;

// Parameterized test for decoding PNG files from the PNGSuite test set
TEST_P(Imgcodecs_Png_PngSuite, decode)
{
    // Construct full paths for the PNG image and corresponding ground truth XML file
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/" + GetParam() + ".png";
    const string xml_filename = root + "pngsuite/" + GetParam() + ".xml";

    // Load the XML file containing the ground truth data
    FileStorage fs(xml_filename, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened()); // Ensure the file was opened successfully

    // Load the image using IMREAD_UNCHANGED to preserve original format
    Mat src = imread(filename, IMREAD_UNCHANGED);
    ASSERT_FALSE(src.empty()); // Ensure the image was loaded successfully

    // Load the ground truth matrix from XML
    Mat gt;
    fs.getFirstTopLevelNode() >> gt;

    // Compare the image loaded with IMREAD_UNCHANGED to the ground truth
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), src, gt);

    // Declare matrices for ground truth in different imread flag combinations
    Mat gt_0, gt_1, gt_2, gt_3, gt_256, gt_258;

    // Handle grayscale 8-bit and 16-bit images
    if (gt.channels() == 1)
    {
        gt.copyTo(gt_2); // For IMREAD_ANYDEPTH
        if (gt.depth() == CV_16U)
            gt_2.convertTo(gt_0, CV_8U, 1. / 256);
        else
        gt_0 = gt_2; // For IMREAD_GRAYSCALE
        cvtColor(gt_0, gt_1, COLOR_GRAY2BGR);  // For IMREAD_COLOR
        cvtColor(gt, gt_3, COLOR_GRAY2BGR); // For IMREAD_COLOR | IMREAD_ANYDEPTH
        gt_256 = gt_1; // For IMREAD_COLOR_RGB
        gt_258 = gt_3; // For IMREAD_COLOR_RGB | IMREAD_ANYDEPTH
    }

    // Handle color images (3 or 4 channels) with 8-bit and 16-bit depth
    if (gt.channels() > 1)
    {
        // Convert to grayscale
        cvtColor(gt, gt_2, COLOR_BGRA2GRAY);
        if (gt.depth() == CV_16U)
            gt_2.convertTo(gt_0, CV_8U, 1. / 256);
        else
           gt_0 = gt_2;

        // Convert to 3-channel BGR
        if (gt.channels() == 3)
            gt.copyTo(gt_3);
        else
            cvtColor(gt, gt_3, COLOR_BGRA2BGR);

        if (gt.depth() == CV_16U)
            gt_3.convertTo(gt_1, CV_8U, 1. / 256);
        else
            gt_1 = gt_3;

        // Convert to RGB for IMREAD_COLOR_RGB variants
        cvtColor(gt_3, gt_256, COLOR_BGR2RGB);
        gt_258 = gt_256;
    }

    // Perform comparisons with different imread flags
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(2, 0), imread(filename, IMREAD_GRAYSCALE), gt_0);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(1, 0), imread(filename, IMREAD_COLOR), gt_1);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(5, 0), imread(filename, IMREAD_ANYDEPTH), gt_2);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), imread(filename, IMREAD_COLOR | IMREAD_ANYDEPTH), gt_3);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(1, 0), imread(filename, IMREAD_COLOR_RGB), gt_256);
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), imread(filename, IMREAD_COLOR_RGB | IMREAD_ANYDEPTH), gt_258);
}

const string pngsuite_files[] =
{
    "basi0g01",
    "basi0g02",
    "basi0g04",
    "basi0g08",
    "basi0g16",
    "basi2c08",
    "basi2c16",
    "basi3p01",
    "basi3p02",
    "basi3p04",
    "basi3p08",
    "basi4a08",
    "basi4a16",
    "basi6a08",
    "basi6a16",
    "basn0g01",
    "basn0g02",
    "basn0g04",
    "basn0g08",
    "basn0g16",
    "basn2c08",
    "basn2c16",
    "basn3p01",
    "basn3p02",
    "basn3p04",
    "basn3p08",
    "basn4a08",
    "basn4a16",
    "basn6a08",
    "basn6a16",
    "bgai4a08",
    "bgai4a16",
    "bgan6a08",
    "bgan6a16",
    "bgbn4a08",
    "bggn4a16",
    "bgwn6a08",
    "bgyn6a16",
    "ccwn2c08",
    "ccwn3p08",
    "cdfn2c08",
    "cdhn2c08",
    "cdsn2c08",
    "cdun2c08",
    "ch1n3p04",
    "ch2n3p08",
    "cm0n0g04",
    "cm7n0g04",
    "cm9n0g04",
    "cs3n2c16",
    "cs3n3p08",
    "cs5n2c08",
    "cs5n3p08",
    "cs8n2c08",
    "cs8n3p08",
    "ct0n0g04",
    "ct1n0g04",
    "cten0g04",
    "ctfn0g04",
    "ctgn0g04",
    "cthn0g04",
    "ctjn0g04",
    "ctzn0g04",
    "exif2c08",
    "f00n0g08",
    "f00n2c08",
    "f01n0g08",
    "f01n2c08",
    "f02n0g08",
    "f02n2c08",
    "f03n0g08",
    "f03n2c08",
    "f04n0g08",
    "f04n2c08",
    "f99n0g04",
    "g03n0g16",
    "g04n0g16",
    "g05n0g16",
    "g07n0g16",
    "g10n0g16",
    "g10n2c08",
    "g10n3p04",
    "g25n0g16",
    "oi1n0g16",
    "oi1n2c16",
    "oi2n0g16",
    "oi2n2c16",
    "oi4n0g16",
    "oi4n2c16",
    "oi9n0g16",
    "oi9n2c16",
    "pp0n2c16",
    "pp0n6a08",
    "ps1n0g08",
    "ps1n2c16",
    "ps2n0g08",
    "ps2n2c16",
    "s01i3p01",
    "s01n3p01",
    "s02i3p01",
    "s02n3p01",
    "s03i3p01",
    "s03n3p01",
    "s04i3p01",
    "s04n3p01",
    "s05i3p02",
    "s05n3p02",
    "s06i3p02",
    "s06n3p02",
    "s07i3p02",
    "s07n3p02",
    "s08i3p02",
    "s08n3p02",
    "s09i3p02",
    "s09n3p02",
    "s32i3p04",
    "s32n3p04",
    "s33i3p04",
    "s33n3p04",
    "s34i3p04",
    "s34n3p04",
    "s35i3p04",
    "s35n3p04",
    "s36i3p04",
    "s36n3p04",
    "s37i3p04",
    "s37n3p04",
    "s38i3p04",
    "s38n3p04",
    "s39i3p04",
    "s39n3p04",
    "s40i3p04",
    "s40n3p04",
    "tbbn0g04",
    "tbbn2c16",
    "tbbn3p08",
    "tbgn2c16",
    "tbgn3p08",
    "tbrn2c08",
    "tbwn0g16",
    "tbwn3p08",
    "tbyn3p08",
    "tm3n3p02",
    "tp0n0g08",
    "tp0n2c08",
    "tp0n3p08",
    "tp1n3p08",
    "z00n2c08",
    "z03n2c08",
    "z06n2c08",
    "z09n2c08",
};

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgcodecs_Png_PngSuite,
                        testing::ValuesIn(pngsuite_files));

typedef testing::TestWithParam<string> Imgcodecs_Png_PngSuite_Gamma;

// Parameterized test for decoding PNG files from the PNGSuite test set
TEST_P(Imgcodecs_Png_PngSuite_Gamma, decode)
{
    // Construct full paths for the PNG image and corresponding ground truth XML file
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/" + GetParam() + ".png";
    const string xml_filename = root + "pngsuite/" + GetParam() + ".xml";

    // Load the XML file containing the ground truth data
    FileStorage fs(xml_filename, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened()); // Ensure the file was opened successfully

    // Load the image using IMREAD_UNCHANGED to preserve original format
    Mat src = imread(filename, IMREAD_UNCHANGED);
    ASSERT_FALSE(src.empty()); // Ensure the image was loaded successfully

    // Load the ground truth matrix from XML
    Mat gt;
    fs.getFirstTopLevelNode() >> gt;

    // Compare the image loaded with IMREAD_UNCHANGED to the ground truth
    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), src, gt);
}

const string pngsuite_files_gamma[] =
{
    "g03n2c08",
    "g03n3p04",
    "g04n2c08",
    "g04n3p04",
    "g05n2c08",
    "g05n3p04",
    "g07n2c08",
    "g07n3p04",
    "g25n2c08",
    "g25n3p04"
};

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgcodecs_Png_PngSuite_Gamma,
                        testing::ValuesIn(pngsuite_files_gamma));

typedef testing::TestWithParam<string> Imgcodecs_Png_PngSuite_Corrupted;

TEST_P(Imgcodecs_Png_PngSuite_Corrupted, decode)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/" + GetParam() + ".png";

    Mat src = imread(filename, IMREAD_UNCHANGED);

    // Corrupted files should not be read
    EXPECT_TRUE(src.empty());
}

const string pngsuite_files_corrupted[] = {
    "xc1n0g08",
    "xc9n2c08",
    "xcrn0g04",
    "xcsn0g01",
    "xd0n2c08",
    "xd3n2c08",
    "xd9n2c08",
    "xdtn0g01",
    "xhdn0g08",
    "xlfn0g04",
    "xs1n0g01",
    "xs2n0g01",
    "xs4n0g01",
    "xs7n0g01",
};

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgcodecs_Png_PngSuite_Corrupted,
                        testing::ValuesIn(pngsuite_files_corrupted));

CV_ENUM(PNGStrategy, IMWRITE_PNG_STRATEGY_DEFAULT, IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, IMWRITE_PNG_STRATEGY_RLE, IMWRITE_PNG_STRATEGY_FIXED);
CV_ENUM(PNGFilters, IMWRITE_PNG_FILTER_NONE, IMWRITE_PNG_FILTER_SUB, IMWRITE_PNG_FILTER_UP, IMWRITE_PNG_FILTER_AVG, IMWRITE_PNG_FILTER_PAETH, IMWRITE_PNG_FAST_FILTERS, IMWRITE_PNG_ALL_FILTERS);

typedef testing::TestWithParam<testing::tuple<string, PNGStrategy, PNGFilters, int>> Imgcodecs_Png_Encode;

TEST_P(Imgcodecs_Png_Encode, params)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/" + get<0>(GetParam());

    const int strategy = get<1>(GetParam());
    const int filter = get<2>(GetParam());
    const int compression_level = get<3>(GetParam());

    std::vector<uchar> file_buf;
    readFileBytes(filename, file_buf);
    Mat src = imdecode(file_buf, IMREAD_UNCHANGED);
    EXPECT_FALSE(src.empty()) << "Cannot decode test image " << filename;

    vector<uchar> buf;
    imencode(".png", src, buf, { IMWRITE_PNG_COMPRESSION, compression_level, IMWRITE_PNG_STRATEGY, strategy, IMWRITE_PNG_FILTER, filter });
    EXPECT_EQ(buf.size(), file_buf.size());
}

INSTANTIATE_TEST_CASE_P(/**/,
    Imgcodecs_Png_Encode,
    testing::Values(
        make_tuple("f00n0g08.png", IMWRITE_PNG_STRATEGY_DEFAULT, IMWRITE_PNG_FILTER_NONE, 6),
        make_tuple("f00n2c08.png", IMWRITE_PNG_STRATEGY_DEFAULT, IMWRITE_PNG_FILTER_NONE, 6),
        make_tuple("f01n0g08.png", IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_FILTER_SUB, 6),
        make_tuple("f01n2c08.png", IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_FILTER_SUB, 6),
        make_tuple("f02n0g08.png", IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_FILTER_UP, 6),
        make_tuple("f02n2c08.png", IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_FILTER_UP, 6),
        make_tuple("f03n0g08.png", IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_FILTER_AVG, 6),
        make_tuple("f03n2c08.png", IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_FILTER_AVG, 6),
        make_tuple("f04n0g08.png", IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_FILTER_PAETH, 6),
        make_tuple("f04n2c08.png", IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_FILTER_PAETH, 6),
        make_tuple("z03n2c08.png", IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_ALL_FILTERS, 3),
        make_tuple("z06n2c08.png", IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_ALL_FILTERS, 6),
        make_tuple("z09n2c08.png", IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_ALL_FILTERS, 9)));

typedef testing::TestWithParam<testing::tuple<string, int, size_t>> Imgcodecs_Png_ImwriteFlags;

TEST_P(Imgcodecs_Png_ImwriteFlags, compression_level)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + get<0>(GetParam());

    const int compression_level = get<1>(GetParam());
    const size_t compression_level_output_size = get<2>(GetParam());

    Mat src = imread(filename, IMREAD_UNCHANGED);
    EXPECT_FALSE(src.empty()) << "Cannot read test image " << filename;

    vector<uchar> buf;
    imencode(".png", src, buf, { IMWRITE_PNG_COMPRESSION, compression_level });
    EXPECT_EQ(buf.size(), compression_level_output_size);
}

INSTANTIATE_TEST_CASE_P(/**/,
    Imgcodecs_Png_ImwriteFlags,
    testing::Values(
        make_tuple("../perf/512x512.png", 0, 788279),
        make_tuple("../perf/512x512.png", 1, 179503),
        make_tuple("../perf/512x512.png", 2, 176007),
        make_tuple("../perf/512x512.png", 3, 170497),
        make_tuple("../perf/512x512.png", 4, 163357),
        make_tuple("../perf/512x512.png", 5, 159190),
        make_tuple("../perf/512x512.png", 6, 156621),
        make_tuple("../perf/512x512.png", 7, 155696),
        make_tuple("../perf/512x512.png", 8, 153708),
        make_tuple("../perf/512x512.png", 9, 152181)));

#endif // HAVE_PNG

}} // namespace
