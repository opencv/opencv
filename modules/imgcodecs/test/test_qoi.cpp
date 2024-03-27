// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_QOI

typedef testing::TestWithParam<string> Imgcodecs_Qoi_basic;

TEST_P(Imgcodecs_Qoi_basic, imwrite_imread)
{
    // (1) read from test data
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + GetParam();
    Mat img_from_extra;
    ASSERT_NO_THROW( img_from_extra = imread( filename, IMREAD_COLOR ) );

    // (2) write to temp file
    string tmp_file_name = tempfile( ".qoi" );
    ASSERT_NO_THROW( imwrite( tmp_file_name, img_from_extra) );

    // (3) read from temp file
    Mat img_from_temp;
    ASSERT_NO_THROW( img_from_temp = imread( tmp_file_name, IMREAD_COLOR ) );

    // (4) compare (1) and (3)
    EXPECT_EQ(cvtest::norm( img_from_extra, img_from_temp, NORM_INF), 0 );

    EXPECT_EQ(0, remove( tmp_file_name.c_str() ) );
}

TEST_P(Imgcodecs_Qoi_basic, imencode_imdecode)
{
    // (1) read from test data
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + GetParam();
    Mat img_from_extra;
    ASSERT_NO_THROW( img_from_extra = imread( filename, IMREAD_COLOR ) );

    // (2) write to memory
    std::vector<uint8_t> buf;
    ASSERT_NO_THROW( imencode( ".qoi", img_from_extra, buf ) );

    // (3) read from memory
    Mat img_from_memory;
    ASSERT_NO_THROW( img_from_memory = imdecode( buf, IMREAD_COLOR ) );

    // (4) compare (1) and (3)
    EXPECT_EQ( cvtest::norm( img_from_extra, img_from_memory, NORM_INF), 0 );
}

const string qoi_basic_files[] =
{
    "readwrite/test_1_c3.qoi",
    "readwrite/test_1_c4.qoi",
    "readwrite/320x260.qoi",
    "readwrite/400x320.qoi",
    "readwrite/640x512.qoi",
    "readwrite/800x600.qoi",
    "readwrite/1280x1024.qoi",
    "readwrite/1680x1050.qoi",
    "readwrite/1920x1080.qoi",
    "readwrite/2560x1600.qoi"
};

INSTANTIATE_TEST_CASE_P(, Imgcodecs_Qoi_basic,
                        testing::ValuesIn(qoi_basic_files));


TEST(Imgcodecs_Qoi, suppport_IMREAD_GRAYSCALE)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/test_1_c4.qoi";

    // (1) read from test data
    Mat img_from_extra;
    EXPECT_NO_THROW( img_from_extra = imread( filename, IMREAD_GRAYSCALE ) );
    EXPECT_EQ( img_from_extra.empty(), false );
    EXPECT_EQ( img_from_extra.channels(), 1 );

    // (2) write to memory
    std::vector<uint8_t> buf;
    ASSERT_NO_THROW( imencode(".qoi", img_from_extra, buf ) );

    // (3) read from memory
    Mat img_from_memory;
    ASSERT_NO_THROW( img_from_memory = imdecode( buf, IMREAD_GRAYSCALE ) );

    // (4) compare (1) and (3)
    EXPECT_EQ( cvtest::norm( img_from_extra, img_from_memory, NORM_INF ), 0 );
}

TEST(Imgcodecs_Qoi, suppport_IMREAD_UNCHANGED)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/test_1_c4.qoi";

    // (1) read from test data
    Mat img_from_extra;
    EXPECT_NO_THROW( img_from_extra = imread( filename, IMREAD_UNCHANGED ) );
    EXPECT_EQ( img_from_extra.empty(), false );
    EXPECT_EQ( img_from_extra.channels(), 4 );

    // (2) write to memory
    std::vector<uint8_t> buf;
    ASSERT_NO_THROW( imencode(".qoi", img_from_extra, buf ) );

    // (3) read from memory
    Mat img_from_memory;
    ASSERT_NO_THROW( img_from_memory = imdecode( buf, IMREAD_UNCHANGED ) );

    // (4) compare (1) and (3)
    EXPECT_EQ( cvtest::norm( img_from_extra, img_from_memory, NORM_INF ), 0 );
}

TEST(Imgcodecs_Qoi, broken_decode)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "readwrite/test_1_c4.qoi";

    // (1) read from test data
    Mat img_from_extra;
    EXPECT_NO_THROW( img_from_extra = imread( filename, IMREAD_COLOR ) );
    EXPECT_EQ( img_from_extra.empty(), false );
    EXPECT_EQ( img_from_extra.channels(), 3 );

    // (2) write to memory
    std::vector<uint8_t> buf;
    ASSERT_NO_THROW( imencode(".qoi", img_from_extra, buf ) );

    /// [TEST.1] break padding data
    {
        // (3) last data modify.
        buf[buf.size() - 1] += 1;

        // (4) read from memory
        Mat img_from_memory;
        ASSERT_NO_THROW( img_from_memory = imdecode( buf, IMREAD_COLOR ) );
        EXPECT_EQ( img_from_memory.empty(), true );
    }

    /// [TEST.2] only signature
    {
        // (3) trancate Qio data
        buf.resize( 4 );

        // (4) read from memory
        Mat img_from_memory;
        ASSERT_NO_THROW( img_from_memory = imdecode( buf, IMREAD_COLOR ) );
        EXPECT_EQ( img_from_memory.empty(), true );
    }
}
#endif // HAVE_QOI

}} // namespace
