// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

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

TEST_P(Imgcodecs_Png_PngSuite, decode)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "pngsuite/" + GetParam() + ".png";
    const string xml_filename = root + "pngsuite/" + GetParam() + ".xml";
    FileStorage fs(xml_filename, FileStorage::READ);
    EXPECT_TRUE(fs.isOpened());

    Mat src = imread(filename, IMREAD_UNCHANGED);
    Mat gt;
    fs.getFirstTopLevelNode() >> gt;

    EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), src, gt);
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
    "g03n2c08",
    "g03n3p04",
    "g04n0g16",
    "g04n2c08",
    "g04n3p04",
    "g05n0g16",
    "g05n2c08",
    "g05n3p04",
    "g07n0g16",
    "g07n2c08",
    "g07n3p04",
    "g10n0g16",
    "g10n2c08",
    "g10n3p04",
    "g25n0g16",
    "g25n2c08",
    "g25n3p04",
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

typedef testing::TestWithParam<testing::tuple<string, int, size_t>> Imgcodecs_Png_Default;

TEST_P(Imgcodecs_Png_Default, compression_level)
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
    Imgcodecs_Png_Default,
    testing::Values(
        make_tuple("pngsuite/z09n2c08.png", 0, 3172),
        make_tuple("pngsuite/z09n2c08.png", 1, 232),
        make_tuple("pngsuite/z09n2c08.png", 2, 232),
        make_tuple("pngsuite/z09n2c08.png", 3, 232),
        make_tuple("pngsuite/z09n2c08.png", 4, 286),
        make_tuple("pngsuite/z09n2c08.png", 5, 242),
        make_tuple("pngsuite/z09n2c08.png", 6, 224),
        make_tuple("pngsuite/z09n2c08.png", 7, 224),
        make_tuple("pngsuite/z09n2c08.png", 8, 224),
        make_tuple("pngsuite/z09n2c08.png", 9, 224),
        make_tuple("pngsuite/PngSuite.png", 0, 197245),
        make_tuple("pngsuite/PngSuite.png", 1, 4014),
        make_tuple("pngsuite/PngSuite.png", 2, 3863),
        make_tuple("pngsuite/PngSuite.png", 3, 3553),
        make_tuple("pngsuite/PngSuite.png", 4, 3248),
        make_tuple("pngsuite/PngSuite.png", 5, 3199),
        make_tuple("pngsuite/PngSuite.png", 6, 2264),
        make_tuple("pngsuite/PngSuite.png", 7, 2582),
        make_tuple("pngsuite/PngSuite.png", 8, 1609),
        make_tuple("pngsuite/PngSuite.png", 9, 1585)));

#endif // HAVE_PNG

}} // namespace
