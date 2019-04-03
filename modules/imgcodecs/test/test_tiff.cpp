// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_TIFF

// these defines are used to resolve conflict between tiff.h and opencv2/core/types_c.h
#define uint64 uint64_hack_
#define int64 int64_hack_
#include "tiff.h"

#ifdef __ANDROID__
// Test disabled as it uses a lot of memory.
// It is killed with SIGKILL by out of memory killer.
TEST(Imgcodecs_Tiff, DISABLED_decode_tile16384x16384)
#else
TEST(Imgcodecs_Tiff, decode_tile16384x16384)
#endif
{
    // see issue #2161
    cv::Mat big(16384, 16384, CV_8UC1, cv::Scalar::all(0));
    string file3 = cv::tempfile(".tiff");
    string file4 = cv::tempfile(".tiff");

    std::vector<int> params;
    params.push_back(TIFFTAG_ROWSPERSTRIP);
    params.push_back(big.rows);
    EXPECT_NO_THROW(cv::imwrite(file4, big, params));
    EXPECT_NO_THROW(cv::imwrite(file3, big.colRange(0, big.cols - 1), params));
    big.release();

    try
    {
        cv::imread(file3, IMREAD_UNCHANGED);
        EXPECT_NO_THROW(cv::imread(file4, IMREAD_UNCHANGED));
    }
    catch(const std::bad_alloc&)
    {
        // not enough memory
    }

    EXPECT_EQ(0, remove(file3.c_str()));
    EXPECT_EQ(0, remove(file4.c_str()));
}

TEST(Imgcodecs_Tiff, write_read_16bit_big_little_endian)
{
    // see issue #2601 "16-bit Grayscale TIFF Load Failures Due to Buffer Underflow and Endianness"

    // Setup data for two minimal 16-bit grayscale TIFF files in both endian formats
    uchar tiff_sample_data[2][86] = { {
        // Little endian
        0x49, 0x49, 0x2a, 0x00, 0x0c, 0x00, 0x00, 0x00, 0xad, 0xde, 0xef, 0xbe, 0x06, 0x00, 0x00, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x01, 0x03, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x06, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x11, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x17, 0x01, 0x04, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00 }, {
        // Big endian
        0x4d, 0x4d, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x0c, 0xde, 0xad, 0xbe, 0xef, 0x00, 0x06, 0x01, 0x00,
        0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x01, 0x01, 0x00, 0x03, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x10,
        0x00, 0x00, 0x01, 0x06, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x11,
        0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x01, 0x17, 0x00, 0x04, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00, 0x00, 0x04 }
        };

    // Test imread() for both a little endian TIFF and big endian TIFF
    for (int i = 0; i < 2; i++)
    {
        string filename = cv::tempfile(".tiff");

        // Write sample TIFF file
        FILE* fp = fopen(filename.c_str(), "wb");
        ASSERT_TRUE(fp != NULL);
        ASSERT_EQ((size_t)1, fwrite(tiff_sample_data[i], 86, 1, fp));
        fclose(fp);

        Mat img = imread(filename, IMREAD_UNCHANGED);

        EXPECT_EQ(1, img.rows);
        EXPECT_EQ(2, img.cols);
        EXPECT_EQ(CV_16U, img.type());
        EXPECT_EQ(sizeof(ushort), img.elemSize());
        EXPECT_EQ(1, img.channels());
        EXPECT_EQ(0xDEAD, img.at<ushort>(0,0));
        EXPECT_EQ(0xBEEF, img.at<ushort>(0,1));

        EXPECT_EQ(0, remove(filename.c_str()));
    }
}

TEST(Imgcodecs_Tiff, decode_tile_remainder)
{
    /* see issue #3472 - dealing with tiled images where the tile size is
     * not a multiple of image size.
     * The tiled images were created with 'convert' from ImageMagick,
     * using the command 'convert <input> -define tiff:tile-geometry=128x128 -depth [8|16] <output>
     * Note that the conversion to 16 bits expands the range from 0-255 to 0-255*255,
     * so the test converts back but rounding errors cause small differences.
     */
    const string root = cvtest::TS::ptr()->get_data_path();
    cv::Mat img = imread(root + "readwrite/non_tiled.tif",-1);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 3);
    cv::Mat tiled8 = imread(root + "readwrite/tiled_8.tif", -1);
    ASSERT_FALSE(tiled8.empty());
    ASSERT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img, tiled8);
    cv::Mat tiled16 = imread(root + "readwrite/tiled_16.tif", -1);
    ASSERT_FALSE(tiled16.empty());
    ASSERT_TRUE(tiled16.elemSize() == 6);
    tiled16.convertTo(tiled8, CV_8UC3, 1./256.);
    ASSERT_PRED_FORMAT2(cvtest::MatComparator(2, 0), img, tiled8);
    // What about 32, 64 bit?
}

TEST(Imgcodecs_Tiff, decode_infinite_rowsperstrip)
{
    const uchar sample_data[142] = {
        0x49, 0x49, 0x2a, 0x00, 0x10, 0x00, 0x00, 0x00, 0x56, 0x54,
        0x56, 0x5a, 0x59, 0x55, 0x5a, 0x00, 0x0a, 0x00, 0x00, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x07, 0x00,
        0x00, 0x00, 0x02, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x08, 0x00, 0x00, 0x00, 0x03, 0x01, 0x03, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x01, 0x03, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x11, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
        0x15, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x16, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00,
        0xff, 0xff, 0xff, 0xff, 0x17, 0x01, 0x04, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x1c, 0x01, 0x03, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00
    };

    const string filename = cv::tempfile(".tiff");
    std::ofstream outfile(filename.c_str(), std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(sample_data), sizeof sample_data);
    outfile.close();

    EXPECT_NO_THROW(cv::imread(filename, IMREAD_UNCHANGED));

    EXPECT_EQ(0, remove(filename.c_str()));
}

TEST(Imgcodecs_Tiff, readWrite_32FC1)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC1.tiff";
    const string filenameOutput = cv::tempfile(".tiff");
    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC1,img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(),img.type());
    ASSERT_EQ(img2.size(),img.size());
    EXPECT_GE(1e-3, cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE));
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

//==================================================================================================

typedef testing::TestWithParam<int> Imgcodecs_Tiff_Modes;

TEST_P(Imgcodecs_Tiff_Modes, decode_multipage)
{
    const int mode = GetParam();
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/multipage.tif";
    const string page_files[] = {
        "readwrite/multipage_p1.tif",
        "readwrite/multipage_p2.tif",
        "readwrite/multipage_p3.tif",
        "readwrite/multipage_p4.tif",
        "readwrite/multipage_p5.tif",
        "readwrite/multipage_p6.tif"
    };
    const size_t page_count = sizeof(page_files)/sizeof(page_files[0]);
    vector<Mat> pages;
    bool res = imreadmulti(filename, pages, mode);
    ASSERT_TRUE(res == true);
    ASSERT_EQ(page_count, pages.size());
    for (size_t i = 0; i < page_count; i++)
    {
        const Mat page = imread(root + page_files[i], mode);
        EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), page, pages[i]);
    }
}

const int all_modes[] =
{
    IMREAD_UNCHANGED,
    IMREAD_GRAYSCALE,
    IMREAD_COLOR,
    IMREAD_ANYDEPTH,
    IMREAD_ANYCOLOR
};

INSTANTIATE_TEST_CASE_P(AllModes, Imgcodecs_Tiff_Modes, testing::ValuesIn(all_modes));

//==================================================================================================

TEST(Imgcodecs_Tiff_Modes, write_multipage)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/multipage.tif";
    const string page_files[] = {
        "readwrite/multipage_p1.tif",
        "readwrite/multipage_p2.tif",
        "readwrite/multipage_p3.tif",
        "readwrite/multipage_p4.tif",
        "readwrite/multipage_p5.tif",
        "readwrite/multipage_p6.tif"
    };
    const size_t page_count = sizeof(page_files) / sizeof(page_files[0]);
    vector<Mat> pages;
    for (size_t i = 0; i < page_count; i++)
    {
        const Mat page = imread(root + page_files[i]);
        pages.push_back(page);
    }

    string tmp_filename = cv::tempfile(".tiff");
    bool res = imwrite(tmp_filename, pages);
    ASSERT_TRUE(res);

    vector<Mat> read_pages;
    imreadmulti(tmp_filename, read_pages);
    for (size_t i = 0; i < page_count; i++)
    {
        EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), read_pages[i], pages[i]);
    }
}

//==================================================================================================

TEST(Imgcodecs_Tiff, imdecode_no_exception_temporary_file_removed)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(filename);
    ASSERT_FALSE(img.empty());
    std::vector<uchar> buf;
    EXPECT_NO_THROW(cv::imencode(".tiff", img, buf));
    EXPECT_NO_THROW(cv::imdecode(buf, IMREAD_UNCHANGED));
}


TEST(Imgcodecs_Tiff, decode_black_and_write_image_pr12989)
{
    const string filename = cvtest::findDataFile("readwrite/bitsperpixel1.tiff");
    cv::Mat img;
    ASSERT_NO_THROW(img = cv::imread(filename, IMREAD_GRAYSCALE));
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(64, img.cols);
    EXPECT_EQ(64, img.rows);
    EXPECT_EQ(CV_8UC1, img.type()) << cv::typeToString(img.type());
    // Check for 0/255 values only: 267 + 3829 = 64*64
    EXPECT_EQ(267, countNonZero(img == 0));
    EXPECT_EQ(3829, countNonZero(img == 255));
}

TEST(Imgcodecs_Tiff, decode_black_and_write_image_pr12989_default)
{
    const string filename = cvtest::findDataFile("readwrite/bitsperpixel1.tiff");
    cv::Mat img;
    ASSERT_NO_THROW(img = cv::imread(filename));  // by default image type is CV_8UC3
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(64, img.cols);
    EXPECT_EQ(64, img.rows);
    EXPECT_EQ(CV_8UC3, img.type()) << cv::typeToString(img.type());
}

#endif

}} // namespace
