// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/core/utils/configuration.private.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_TIFF

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
    params.push_back(IMWRITE_TIFF_ROWSPERSTRIP);
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

//==================================================================================================
// See https://github.com/opencv/opencv/issues/22388

/**
 * Dummy enum to show combination of IMREAD_*.
 */
enum ImreadMixModes
{
    IMREAD_MIX_UNCHANGED                   = IMREAD_UNCHANGED                                     ,
    IMREAD_MIX_GRAYSCALE                   = IMREAD_GRAYSCALE                                     ,
    IMREAD_MIX_COLOR                       = IMREAD_COLOR     | IMREAD_COLOR_RGB                  ,
    IMREAD_MIX_GRAYSCALE_ANYDEPTH          = IMREAD_GRAYSCALE | IMREAD_ANYDEPTH                   ,
    IMREAD_MIX_GRAYSCALE_ANYCOLOR          = IMREAD_GRAYSCALE                    | IMREAD_ANYCOLOR,
    IMREAD_MIX_GRAYSCALE_ANYDEPTH_ANYCOLOR = IMREAD_GRAYSCALE | IMREAD_ANYDEPTH  | IMREAD_ANYCOLOR,
    IMREAD_MIX_COLOR_ANYDEPTH              = IMREAD_COLOR     | IMREAD_ANYDEPTH                   ,
    IMREAD_MIX_COLOR_ANYCOLOR              = IMREAD_COLOR                        | IMREAD_ANYCOLOR,
    IMREAD_MIX_COLOR_ANYDEPTH_ANYCOLOR     = IMREAD_COLOR     | IMREAD_ANYDEPTH  | IMREAD_ANYCOLOR
};

typedef tuple< uint64_t, tuple<string, int>, ImreadMixModes > Bufsize_and_Type;
typedef testing::TestWithParam<Bufsize_and_Type> Imgcodecs_Tiff_decode_Huge;

static inline
void PrintTo(const ImreadMixModes& val, std::ostream* os)
{
    PrintTo( static_cast<ImreadModes>(val), os );
}

TEST_P(Imgcodecs_Tiff_decode_Huge, regression)
{
    // Get test parameters
    const uint64_t buffer_size   = get<0>(GetParam());
    const string mat_type_string =   get<0>(get<1>(GetParam()));
    const int mat_type           =   get<1>(get<1>(GetParam()));
    const int imread_mode        = get<2>(GetParam());

    // Detect data file
    const string req_filename = cv::format("readwrite/huge-tiff/%s_%zu.tif", mat_type_string.c_str(), (size_t)buffer_size);
    const string filename = findDataFile( req_filename );

    // Preparation process for test
    {
        // Convert from mat_type and buffer_size to tiff file information.
        const uint64_t width  = 32768;
        int ncn          = CV_MAT_CN(mat_type);
        int depth        = ( CV_MAT_DEPTH(mat_type) == CV_16U) ? 2 : 1; // 16bit or 8 bit
        const uint64_t height = (uint64_t) buffer_size / width / ncn / depth;
        const uint64_t base_scanline_size  = (uint64_t) width * ncn  * depth;
        const uint64_t base_strip_size     = (uint64_t) base_scanline_size * height;

        // To avoid exception about pixel size, check it.
        static const size_t CV_IO_MAX_IMAGE_PIXELS = utils::getConfigurationParameterSizeT("OPENCV_IO_MAX_IMAGE_PIXELS", 1 << 30);
        uint64_t pixels = (uint64_t) width * height;
        if ( pixels > CV_IO_MAX_IMAGE_PIXELS )
        {
            throw SkipTestException( cv::format("Test is skipped( pixels(%zu) > CV_IO_MAX_IMAGE_PIXELS(%zu) )",
                (size_t)pixels, CV_IO_MAX_IMAGE_PIXELS) );
        }

        // If buffer_size >= 1GB * 95%, TIFFReadScanline() is used.
        const uint64_t BUFFER_SIZE_LIMIT_FOR_READS_CANLINE = (uint64_t) 1024*1024*1024*95/100;
        const bool doReadScanline = ( base_strip_size >= BUFFER_SIZE_LIMIT_FOR_READS_CANLINE );

        // Update ncn and depth for destination Mat.
        switch ( imread_mode )
        {
            case IMREAD_UNCHANGED:
                break;
            case IMREAD_GRAYSCALE:
                ncn = 1;
                depth = 1;
                break;
            case IMREAD_GRAYSCALE | IMREAD_ANYDEPTH:
                ncn = 1;
                break;
            case IMREAD_GRAYSCALE | IMREAD_ANYCOLOR:
                ncn = (ncn == 1)?1:3;
                depth = 1;
                break;
            case IMREAD_GRAYSCALE | IMREAD_ANYCOLOR | IMREAD_ANYDEPTH:
                ncn = (ncn == 1)?1:3;
                break;
            case IMREAD_COLOR | IMREAD_COLOR_RGB:
                ncn = 3;
                depth = 1;
                break;
            case IMREAD_COLOR | IMREAD_ANYDEPTH:
                ncn = 3;
                break;
            case IMREAD_COLOR | IMREAD_ANYCOLOR:
                ncn = 3;
                depth = 1;
                break;
            case IMREAD_COLOR | IMREAD_ANYDEPTH | IMREAD_ANYCOLOR:
                ncn = 3;
                break;
            default:
                break;
        }

        // Memory usage for Destination Mat
        const uint64_t memory_usage_cvmat = (uint64_t) width * ncn * depth * height;

        // Memory usage for Work memory in libtiff.
        uint64_t memory_usage_tiff = 0;
        if ( ( depth == 1 ) && ( !doReadScanline ) )
        {
            // TIFFReadRGBA*() request to allocate RGBA(32bit) buffer.
            memory_usage_tiff = (uint64_t)
                width *
                4 *      // ncn     = RGBA
                1 *      // dst_bpp = 8 bpp
                height;
        }
        else
        {
            // TIFFReadEncodedStrip() or TIFFReadScanline() request to allocate strip memory.
            memory_usage_tiff = base_strip_size;
        }

        // Memory usage for Work memory in imgcodec/grfmt_tiff.cpp
        const uint64_t memory_usage_work =
            ( doReadScanline ) ? base_scanline_size // for TIFFReadScanline()
                               : base_strip_size;   // for TIFFReadRGBA*() or TIFFReadEncodedStrip()

        // Total memory usage.
        const uint64_t memory_usage_total =
            memory_usage_cvmat + // Destination Mat
            memory_usage_tiff  + // Work memory in libtiff
            memory_usage_work;   // Work memory in imgcodecs

        // Output memory usage log.
        CV_LOG_DEBUG(NULL, cv::format("OpenCV TIFF-test: memory usage info : mat(%zu), libtiff(%zu), work(%zu) -> total(%zu)",
                     (size_t)memory_usage_cvmat, (size_t)memory_usage_tiff, (size_t)memory_usage_work, (size_t)memory_usage_total) );

        // Add test tags.
        if ( memory_usage_total >= (uint64_t) 6144 * 1024 * 1024 )
        {
            applyTestTag( CV_TEST_TAG_MEMORY_14GB, CV_TEST_TAG_VERYLONG );
        }
        else if ( memory_usage_total >= (uint64_t) 2048 * 1024 * 1024 )
        {
            applyTestTag( CV_TEST_TAG_MEMORY_6GB, CV_TEST_TAG_VERYLONG );
        }
        else if ( memory_usage_total >= (uint64_t) 1024  * 1024 * 1024 )
        {
            applyTestTag( CV_TEST_TAG_MEMORY_2GB, CV_TEST_TAG_LONG );
        }
        else if ( memory_usage_total >= (uint64_t)  512  * 1024 * 1024 )
        {
            applyTestTag( CV_TEST_TAG_MEMORY_1GB );
        }
        else if ( memory_usage_total >= (uint64_t)  200  * 1024 * 1024 )
        {
            applyTestTag( CV_TEST_TAG_MEMORY_512MB );
        }
        else
        {
            // do nothing.
        }
    }

    // TEST Main

    cv::Mat img;
    ASSERT_NO_THROW( img = cv::imread(filename, imread_mode) );
    ASSERT_FALSE(img.empty());

    /**
     * Test marker pixels at each corners.
     *
     *   0xAn,0x00 ... 0x00, 0xBn
     *   0x00,0x00 ... 0x00, 0x00
     *   :    :         :     :
     *   0x00,0x00 ... 0x00, 0x00
     *   0xCn,0x00 .., 0x00, 0xDn
     *
     */

#define MAKE_FLAG(from_type, to_type) (((uint64_t)from_type << 32 ) | to_type )

    switch ( MAKE_FLAG(mat_type, img.type() ) )
    {
    // GRAY TO GRAY
    case MAKE_FLAG(CV_8UC1, CV_8UC1):
    case MAKE_FLAG(CV_16UC1, CV_8UC1):
        EXPECT_EQ( 0xA0,   img.at<uchar>(0,          0)          );
        EXPECT_EQ( 0xB0,   img.at<uchar>(0,          img.cols-1) );
        EXPECT_EQ( 0xC0,   img.at<uchar>(img.rows-1, 0)          );
        EXPECT_EQ( 0xD0,   img.at<uchar>(img.rows-1, img.cols-1) );
        break;

    // RGB/RGBA TO BGR
    case MAKE_FLAG(CV_8UC3, CV_8UC3):
    case MAKE_FLAG(CV_8UC4, CV_8UC3):
    case MAKE_FLAG(CV_16UC3, CV_8UC3):
    case MAKE_FLAG(CV_16UC4, CV_8UC3):
        EXPECT_EQ( 0xA2,   img.at<Vec3b>(0,          0)         [0] );
        EXPECT_EQ( 0xA1,   img.at<Vec3b>(0,          0)         [1] );
        EXPECT_EQ( 0xA0,   img.at<Vec3b>(0,          0)         [2] );
        EXPECT_EQ( 0xB2,   img.at<Vec3b>(0,          img.cols-1)[0] );
        EXPECT_EQ( 0xB1,   img.at<Vec3b>(0,          img.cols-1)[1] );
        EXPECT_EQ( 0xB0,   img.at<Vec3b>(0,          img.cols-1)[2] );
        EXPECT_EQ( 0xC2,   img.at<Vec3b>(img.rows-1, 0)         [0] );
        EXPECT_EQ( 0xC1,   img.at<Vec3b>(img.rows-1, 0)         [1] );
        EXPECT_EQ( 0xC0,   img.at<Vec3b>(img.rows-1, 0)         [2] );
        EXPECT_EQ( 0xD2,   img.at<Vec3b>(img.rows-1, img.cols-1)[0] );
        EXPECT_EQ( 0xD1,   img.at<Vec3b>(img.rows-1, img.cols-1)[1] );
        EXPECT_EQ( 0xD0,   img.at<Vec3b>(img.rows-1, img.cols-1)[2] );
        break;

    // RGBA TO BGRA
    case MAKE_FLAG(CV_8UC4, CV_8UC4):
    case MAKE_FLAG(CV_16UC4, CV_8UC4):
        EXPECT_EQ( 0xA2,   img.at<Vec4b>(0,          0)         [0] );
        EXPECT_EQ( 0xA1,   img.at<Vec4b>(0,          0)         [1] );
        EXPECT_EQ( 0xA0,   img.at<Vec4b>(0,          0)         [2] );
        EXPECT_EQ( 0xA3,   img.at<Vec4b>(0,          0)         [3] );
        EXPECT_EQ( 0xB2,   img.at<Vec4b>(0,          img.cols-1)[0] );
        EXPECT_EQ( 0xB1,   img.at<Vec4b>(0,          img.cols-1)[1] );
        EXPECT_EQ( 0xB0,   img.at<Vec4b>(0,          img.cols-1)[2] );
        EXPECT_EQ( 0xB3,   img.at<Vec4b>(0,          img.cols-1)[3] );
        EXPECT_EQ( 0xC2,   img.at<Vec4b>(img.rows-1, 0)         [0] );
        EXPECT_EQ( 0xC1,   img.at<Vec4b>(img.rows-1, 0)         [1] );
        EXPECT_EQ( 0xC0,   img.at<Vec4b>(img.rows-1, 0)         [2] );
        EXPECT_EQ( 0xC3,   img.at<Vec4b>(img.rows-1, 0)         [3] );
        EXPECT_EQ( 0xD2,   img.at<Vec4b>(img.rows-1, img.cols-1)[0] );
        EXPECT_EQ( 0xD1,   img.at<Vec4b>(img.rows-1, img.cols-1)[1] );
        EXPECT_EQ( 0xD0,   img.at<Vec4b>(img.rows-1, img.cols-1)[2] );
        EXPECT_EQ( 0xD3,   img.at<Vec4b>(img.rows-1, img.cols-1)[3] );
        break;

    // RGB/RGBA to GRAY
    case MAKE_FLAG(CV_8UC3, CV_8UC1):
    case MAKE_FLAG(CV_8UC4, CV_8UC1):
    case MAKE_FLAG(CV_16UC3, CV_8UC1):
    case MAKE_FLAG(CV_16UC4, CV_8UC1):
        EXPECT_LE( 0xA0,   img.at<uchar>(0,          0)          );
        EXPECT_GE( 0xA2,   img.at<uchar>(0,          0)          );
        EXPECT_LE( 0xB0,   img.at<uchar>(0,          img.cols-1) );
        EXPECT_GE( 0xB2,   img.at<uchar>(0,          img.cols-1) );
        EXPECT_LE( 0xC0,   img.at<uchar>(img.rows-1, 0)          );
        EXPECT_GE( 0xC2,   img.at<uchar>(img.rows-1, 0)          );
        EXPECT_LE( 0xD0,   img.at<uchar>(img.rows-1, img.cols-1) );
        EXPECT_GE( 0xD2,   img.at<uchar>(img.rows-1, img.cols-1) );
        break;

    // GRAY to BGR
    case MAKE_FLAG(CV_8UC1, CV_8UC3):
    case MAKE_FLAG(CV_16UC1, CV_8UC3):
        EXPECT_EQ( 0xA0,   img.at<Vec3b>(0,          0)         [0] );
        EXPECT_EQ( 0xB0,   img.at<Vec3b>(0,          img.cols-1)[0] );
        EXPECT_EQ( 0xC0,   img.at<Vec3b>(img.rows-1, 0)         [0] );
        EXPECT_EQ( 0xD0,   img.at<Vec3b>(img.rows-1, img.cols-1)[0] );
        // R==G==B
        EXPECT_EQ( img.at<Vec3b>(0,          0)          [0], img.at<Vec3b>(0,          0)         [1] );
        EXPECT_EQ( img.at<Vec3b>(0,          0)          [0], img.at<Vec3b>(0,          0)         [2] );
        EXPECT_EQ( img.at<Vec3b>(0,          img.cols-1) [0], img.at<Vec3b>(0,          img.cols-1)[1] );
        EXPECT_EQ( img.at<Vec3b>(0,          img.cols-1) [0], img.at<Vec3b>(0,          img.cols-1)[2] );
        EXPECT_EQ( img.at<Vec3b>(img.rows-1,          0) [0], img.at<Vec3b>(img.rows-1, 0)         [1] );
        EXPECT_EQ( img.at<Vec3b>(img.rows-1,          0) [0], img.at<Vec3b>(img.rows-1, 0)         [2] );
        EXPECT_EQ( img.at<Vec3b>(img.rows-1, img.cols-1) [0], img.at<Vec3b>(img.rows-1, img.cols-1)[1] );
        EXPECT_EQ( img.at<Vec3b>(img.rows-1, img.cols-1) [0], img.at<Vec3b>(img.rows-1, img.cols-1)[2] );
        break;

    // GRAY TO GRAY
    case MAKE_FLAG(CV_16UC1, CV_16UC1):
        EXPECT_EQ( 0xA090, img.at<ushort>(0,          0)          );
        EXPECT_EQ( 0xB080, img.at<ushort>(0,          img.cols-1) );
        EXPECT_EQ( 0xC070, img.at<ushort>(img.rows-1, 0)          );
        EXPECT_EQ( 0xD060, img.at<ushort>(img.rows-1, img.cols-1) );
        break;

    // RGB/RGBA TO BGR
    case MAKE_FLAG(CV_16UC3, CV_16UC3):
    case MAKE_FLAG(CV_16UC4, CV_16UC3):
        EXPECT_EQ( 0xA292, img.at<Vec3w>(0,          0)         [0] );
        EXPECT_EQ( 0xA191, img.at<Vec3w>(0,          0)         [1] );
        EXPECT_EQ( 0xA090, img.at<Vec3w>(0,          0)         [2] );
        EXPECT_EQ( 0xB282, img.at<Vec3w>(0,          img.cols-1)[0] );
        EXPECT_EQ( 0xB181, img.at<Vec3w>(0,          img.cols-1)[1] );
        EXPECT_EQ( 0xB080, img.at<Vec3w>(0,          img.cols-1)[2] );
        EXPECT_EQ( 0xC272, img.at<Vec3w>(img.rows-1, 0)         [0] );
        EXPECT_EQ( 0xC171, img.at<Vec3w>(img.rows-1, 0)         [1] );
        EXPECT_EQ( 0xC070, img.at<Vec3w>(img.rows-1, 0)         [2] );
        EXPECT_EQ( 0xD262, img.at<Vec3w>(img.rows-1, img.cols-1)[0] );
        EXPECT_EQ( 0xD161, img.at<Vec3w>(img.rows-1, img.cols-1)[1] );
        EXPECT_EQ( 0xD060, img.at<Vec3w>(img.rows-1, img.cols-1)[2] );
        break;

    // RGBA TO RGBA
    case MAKE_FLAG(CV_16UC4, CV_16UC4):
        EXPECT_EQ( 0xA292, img.at<Vec4w>(0,          0)         [0] );
        EXPECT_EQ( 0xA191, img.at<Vec4w>(0,          0)         [1] );
        EXPECT_EQ( 0xA090, img.at<Vec4w>(0,          0)         [2] );
        EXPECT_EQ( 0xA393, img.at<Vec4w>(0,          0)         [3] );
        EXPECT_EQ( 0xB282, img.at<Vec4w>(0,          img.cols-1)[0] );
        EXPECT_EQ( 0xB181, img.at<Vec4w>(0,          img.cols-1)[1] );
        EXPECT_EQ( 0xB080, img.at<Vec4w>(0,          img.cols-1)[2] );
        EXPECT_EQ( 0xB383, img.at<Vec4w>(0,          img.cols-1)[3] );
        EXPECT_EQ( 0xC272, img.at<Vec4w>(img.rows-1, 0)         [0] );
        EXPECT_EQ( 0xC171, img.at<Vec4w>(img.rows-1, 0)         [1] );
        EXPECT_EQ( 0xC070, img.at<Vec4w>(img.rows-1, 0)         [2] );
        EXPECT_EQ( 0xC373, img.at<Vec4w>(img.rows-1, 0)         [3] );
        EXPECT_EQ( 0xD262, img.at<Vec4w>(img.rows-1,img.cols-1) [0] );
        EXPECT_EQ( 0xD161, img.at<Vec4w>(img.rows-1,img.cols-1) [1] );
        EXPECT_EQ( 0xD060, img.at<Vec4w>(img.rows-1,img.cols-1) [2] );
        EXPECT_EQ( 0xD363, img.at<Vec4w>(img.rows-1,img.cols-1) [3] );
        break;

    // RGB/RGBA to GRAY
    case MAKE_FLAG(CV_16UC3, CV_16UC1):
    case MAKE_FLAG(CV_16UC4, CV_16UC1):
        EXPECT_LE( 0xA090, img.at<ushort>(0,          0) );
        EXPECT_GE( 0xA292, img.at<ushort>(0,          0) );
        EXPECT_LE( 0xB080, img.at<ushort>(0,          img.cols-1) );
        EXPECT_GE( 0xB282, img.at<ushort>(0,          img.cols-1) );
        EXPECT_LE( 0xC070, img.at<ushort>(img.rows-1, 0) );
        EXPECT_GE( 0xC272, img.at<ushort>(img.rows-1, 0) );
        EXPECT_LE( 0xD060, img.at<ushort>(img.rows-1, img.cols-1) );
        EXPECT_GE( 0xD262, img.at<ushort>(img.rows-1, img.cols-1) );
        break;

    // GRAY to RGB
    case MAKE_FLAG(CV_16UC1, CV_16UC3):
        EXPECT_EQ( 0xA090,   img.at<Vec3w>(0,          0)         [0] );
        EXPECT_EQ( 0xB080,   img.at<Vec3w>(0,          img.cols-1)[0] );
        EXPECT_EQ( 0xC070,   img.at<Vec3w>(img.rows-1, 0)         [0] );
        EXPECT_EQ( 0xD060,   img.at<Vec3w>(img.rows-1, img.cols-1)[0] );
        // R==G==B
        EXPECT_EQ( img.at<Vec3w>(0,          0)          [0], img.at<Vec3w>(0,          0)         [1] );
        EXPECT_EQ( img.at<Vec3w>(0,          0)          [0], img.at<Vec3w>(0,          0)         [2] );
        EXPECT_EQ( img.at<Vec3w>(0,          img.cols-1) [0], img.at<Vec3w>(0,          img.cols-1)[1] );
        EXPECT_EQ( img.at<Vec3w>(0,          img.cols-1) [0], img.at<Vec3w>(0,          img.cols-1)[2] );
        EXPECT_EQ( img.at<Vec3w>(img.rows-1,          0) [0], img.at<Vec3w>(img.rows-1, 0)         [1] );
        EXPECT_EQ( img.at<Vec3w>(img.rows-1,          0) [0], img.at<Vec3w>(img.rows-1, 0)         [2] );
        EXPECT_EQ( img.at<Vec3w>(img.rows-1, img.cols-1) [0], img.at<Vec3w>(img.rows-1, img.cols-1)[1] );
        EXPECT_EQ( img.at<Vec3w>(img.rows-1, img.cols-1) [0], img.at<Vec3w>(img.rows-1, img.cols-1)[2] );
        break;

    // No supported.
    // (1) 8bit to 16bit
    case MAKE_FLAG(CV_8UC1, CV_16UC1):
    case MAKE_FLAG(CV_8UC1, CV_16UC3):
    case MAKE_FLAG(CV_8UC1, CV_16UC4):
    case MAKE_FLAG(CV_8UC3, CV_16UC1):
    case MAKE_FLAG(CV_8UC3, CV_16UC3):
    case MAKE_FLAG(CV_8UC3, CV_16UC4):
    case MAKE_FLAG(CV_8UC4, CV_16UC1):
    case MAKE_FLAG(CV_8UC4, CV_16UC3):
    case MAKE_FLAG(CV_8UC4, CV_16UC4):
    // (2) GRAY/RGB TO RGBA
    case MAKE_FLAG(CV_8UC1, CV_8UC4):
    case MAKE_FLAG(CV_8UC3, CV_8UC4):
    case MAKE_FLAG(CV_16UC1, CV_8UC4):
    case MAKE_FLAG(CV_16UC3, CV_8UC4):
    case MAKE_FLAG(CV_16UC1, CV_16UC4):
    case MAKE_FLAG(CV_16UC3, CV_16UC4):
    default:
        FAIL() << cv::format("Unknown test pattern: from = %d ( %d, %d) to = %d ( %d, %d )",
                              mat_type,   (int)CV_MAT_CN(mat_type   ), ( CV_MAT_DEPTH(mat_type   )==CV_16U)?16:8,
                              img.type(), (int)CV_MAT_CN(img.type() ), ( CV_MAT_DEPTH(img.type() )==CV_16U)?16:8);
        break;
    }

#undef MAKE_FLAG
}

// Basic Test
const Bufsize_and_Type Imgcodecs_Tiff_decode_Huge_list_basic[] =
{
    make_tuple<uint64_t, tuple<string,int>,ImreadMixModes>( 1073479680ull, make_tuple<string,int>("CV_8UC1",  CV_8UC1),  IMREAD_MIX_COLOR ),
    make_tuple<uint64_t, tuple<string,int>,ImreadMixModes>( 2147483648ull, make_tuple<string,int>("CV_16UC4", CV_16UC4), IMREAD_MIX_COLOR ),
};

INSTANTIATE_TEST_CASE_P(Imgcodecs_Tiff, Imgcodecs_Tiff_decode_Huge,
        testing::ValuesIn( Imgcodecs_Tiff_decode_Huge_list_basic )
);

// Full Test

/**
 * Test lists for combination of IMREAD_*.
 */
const ImreadMixModes all_modes_Huge_Full[] =
{
    IMREAD_MIX_UNCHANGED,
    IMREAD_MIX_GRAYSCALE,
    IMREAD_MIX_GRAYSCALE_ANYDEPTH,
    IMREAD_MIX_GRAYSCALE_ANYCOLOR,
    IMREAD_MIX_GRAYSCALE_ANYDEPTH_ANYCOLOR,
    IMREAD_MIX_COLOR,
    IMREAD_MIX_COLOR_ANYDEPTH,
    IMREAD_MIX_COLOR_ANYCOLOR,
    IMREAD_MIX_COLOR_ANYDEPTH_ANYCOLOR,
};

const uint64_t huge_buffer_sizes_decode_Full[] =
{
    1048576ull,    // 1 * 1024 * 1024
    1073479680ull, // 1024 * 1024 * 1024 - 32768 * 4 * 2
    1073741824ull, // 1024 * 1024 * 1024
    2147483648ull, // 2048 * 1024 * 1024
};

const tuple<string, int> mat_types_Full[] =
{
    make_tuple<string, int>("CV_8UC1",  CV_8UC1),  // 8bit  GRAY
    make_tuple<string, int>("CV_8UC3",  CV_8UC3),  // 24bit RGB
    make_tuple<string, int>("CV_8UC4",  CV_8UC4),  // 32bit RGBA
    make_tuple<string, int>("CV_16UC1", CV_16UC1), // 16bit GRAY
    make_tuple<string, int>("CV_16UC3", CV_16UC3), // 48bit RGB
    make_tuple<string, int>("CV_16UC4", CV_16UC4), // 64bit RGBA
};

INSTANTIATE_TEST_CASE_P(DISABLED_Imgcodecs_Tiff_Full, Imgcodecs_Tiff_decode_Huge,
        testing::Combine(
            testing::ValuesIn(huge_buffer_sizes_decode_Full),
            testing::ValuesIn(mat_types_Full),
            testing::ValuesIn(all_modes_Huge_Full)
        )
);


//==================================================================================================

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

TEST(Imgcodecs_Tiff, decode_10_12_14)
{
    /* see issue #21700
    */
    const string root = cvtest::TS::ptr()->get_data_path();

    const double maxDiff = 256;//samples do not have the exact same values because of the tool that created them
    cv::Mat tmp;
    double diff = 0;

    cv::Mat img8UC1 = imread(root + "readwrite/pattern_8uc1.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img8UC1.empty());
    ASSERT_EQ(img8UC1.type(), CV_8UC1);

    cv::Mat img8UC3 = imread(root + "readwrite/pattern_8uc3.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img8UC3.empty());
    ASSERT_EQ(img8UC3.type(), CV_8UC3);

    cv::Mat img8UC4 = imread(root + "readwrite/pattern_8uc4.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img8UC4.empty());
    ASSERT_EQ(img8UC4.type(), CV_8UC4);

    cv::Mat img16UC1 = imread(root + "readwrite/pattern_16uc1.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img16UC1.empty());
    ASSERT_EQ(img16UC1.type(), CV_16UC1);
    ASSERT_EQ(img8UC1.size(), img16UC1.size());
    img8UC1.convertTo(tmp, img16UC1.type(), (1U<<(16-8)));
    diff = cv::norm(tmp.reshape(1), img16UC1.reshape(1), cv::NORM_INF);
    ASSERT_LE(diff, maxDiff);

    cv::Mat img16UC3 = imread(root + "readwrite/pattern_16uc3.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img16UC3.empty());
    ASSERT_EQ(img16UC3.type(), CV_16UC3);
    ASSERT_EQ(img8UC3.size(), img16UC3.size());
    img8UC3.convertTo(tmp, img16UC3.type(), (1U<<(16-8)));
    diff = cv::norm(tmp.reshape(1), img16UC3.reshape(1), cv::NORM_INF);
    ASSERT_LE(diff, maxDiff);

    cv::Mat img16UC4 = imread(root + "readwrite/pattern_16uc4.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img16UC4.empty());
    ASSERT_EQ(img16UC4.type(), CV_16UC4);
    ASSERT_EQ(img8UC4.size(), img16UC4.size());
    img8UC4.convertTo(tmp, img16UC4.type(), (1U<<(16-8)));
    diff = cv::norm(tmp.reshape(1), img16UC4.reshape(1), cv::NORM_INF);
    ASSERT_LE(diff, maxDiff);

    cv::Mat img10UC1 = imread(root + "readwrite/pattern_10uc1.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img10UC1.empty());
    ASSERT_EQ(img10UC1.type(), CV_16UC1);
    ASSERT_EQ(img10UC1.size(), img16UC1.size());
    diff = cv::norm(img10UC1.reshape(1), img16UC1.reshape(1), cv::NORM_INF);
    ASSERT_LE(diff, maxDiff);

    cv::Mat img10UC3 = imread(root + "readwrite/pattern_10uc3.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img10UC3.empty());
    ASSERT_EQ(img10UC3.type(), CV_16UC3);
    ASSERT_EQ(img10UC3.size(), img16UC3.size());
    diff = cv::norm(img10UC3.reshape(1), img16UC3.reshape(1), cv::NORM_INF);
    ASSERT_LE(diff, maxDiff);

    cv::Mat img10UC4 = imread(root + "readwrite/pattern_10uc4.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img10UC4.empty());
    ASSERT_EQ(img10UC4.type(), CV_16UC4);
    ASSERT_EQ(img10UC4.size(), img16UC4.size());
    diff = cv::norm(img10UC4.reshape(1), img16UC4.reshape(1), cv::NORM_INF);
    ASSERT_LE(diff, maxDiff);

    cv::Mat img12UC1 = imread(root + "readwrite/pattern_12uc1.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img12UC1.empty());
    ASSERT_EQ(img12UC1.type(), CV_16UC1);
    ASSERT_EQ(img12UC1.size(), img16UC1.size());
    diff = cv::norm(img12UC1.reshape(1), img16UC1.reshape(1), cv::NORM_INF);
    ASSERT_LE(diff, maxDiff);

    cv::Mat img12UC3 = imread(root + "readwrite/pattern_12uc3.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img12UC3.empty());
    ASSERT_EQ(img12UC3.type(), CV_16UC3);
    ASSERT_EQ(img12UC3.size(), img16UC3.size());
    diff = cv::norm(img12UC3.reshape(1), img16UC3.reshape(1), cv::NORM_INF);
    ASSERT_LE(diff, maxDiff);

    cv::Mat img12UC4 = imread(root + "readwrite/pattern_12uc4.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img12UC4.empty());
    ASSERT_EQ(img12UC4.type(), CV_16UC4);
    ASSERT_EQ(img12UC4.size(), img16UC4.size());
    diff = cv::norm(img12UC4.reshape(1), img16UC4.reshape(1), cv::NORM_INF);
    ASSERT_LE(diff, maxDiff);

    cv::Mat img14UC1 = imread(root + "readwrite/pattern_14uc1.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img14UC1.empty());
    ASSERT_EQ(img14UC1.type(), CV_16UC1);
    ASSERT_EQ(img14UC1.size(), img16UC1.size());
    diff = cv::norm(img14UC1.reshape(1), img16UC1.reshape(1), cv::NORM_INF);
    ASSERT_LE(diff, maxDiff);

    cv::Mat img14UC3 = imread(root + "readwrite/pattern_14uc3.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img14UC3.empty());
    ASSERT_EQ(img14UC3.type(), CV_16UC3);
    ASSERT_EQ(img14UC3.size(), img16UC3.size());
    diff = cv::norm(img14UC3.reshape(1), img16UC3.reshape(1), cv::NORM_INF);
    ASSERT_LE(diff, maxDiff);

    cv::Mat img14UC4 = imread(root + "readwrite/pattern_14uc4.tif", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(img14UC4.empty());
    ASSERT_EQ(img14UC4.type(), CV_16UC4);
    ASSERT_EQ(img14UC4.size(), img16UC4.size());
    diff = cv::norm(img14UC4.reshape(1), img16UC4.reshape(1), cv::NORM_INF);
    ASSERT_LE(diff, maxDiff);
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

TEST(Imgcodecs_Tiff, readWrite_unsigned)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/gray_8u.tif";
    const string filenameOutput = cv::tempfile(".tiff");

    Mat img;
    ASSERT_NO_THROW(img = cv::imread(filenameInput, IMREAD_UNCHANGED));
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_8UC1, img.type());

    Mat matS8;
    img.convertTo(matS8, CV_8SC1);

    bool ret_imwrite = false;
    ASSERT_NO_THROW(ret_imwrite = cv::imwrite(filenameOutput, matS8));
    ASSERT_TRUE(ret_imwrite);

    Mat img2;
    ASSERT_NO_THROW(img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED));
    ASSERT_FALSE(img2.empty());
    ASSERT_EQ(img2.type(), matS8.type());
    ASSERT_EQ(img2.size(), matS8.size());
    EXPECT_LE(cvtest::norm(matS8, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_Tiff, readWrite_32FC1)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC1.tiff";
    const string filenameOutput = cv::tempfile(".tiff");

    Mat img;
    ASSERT_NO_THROW(img = cv::imread(filenameInput, IMREAD_UNCHANGED));
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC1,img.type());

    bool ret_imwrite = false;
    ASSERT_NO_THROW(ret_imwrite = cv::imwrite(filenameOutput, img));
    ASSERT_TRUE(ret_imwrite);

    Mat img2;
    ASSERT_NO_THROW(img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED));
    ASSERT_FALSE(img2.empty());
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_Tiff, readWrite_64FC1)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test64FC1.tiff";
    const string filenameOutput = cv::tempfile(".tiff");

    Mat img;
    ASSERT_NO_THROW(img = cv::imread(filenameInput, IMREAD_UNCHANGED));
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_64FC1, img.type());

    bool ret_imwrite = false;
    ASSERT_NO_THROW(ret_imwrite = cv::imwrite(filenameOutput, img));
    ASSERT_TRUE(ret_imwrite);

    Mat img2;
    ASSERT_NO_THROW(img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED));
    ASSERT_FALSE(img2.empty());
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_Tiff, readWrite_32FC3_SGILOG)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC3_sgilog.tiff";
    const string filenameOutput = cv::tempfile(".tiff");

    Mat img;
    ASSERT_NO_THROW(img = cv::imread(filenameInput, IMREAD_UNCHANGED));
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC3, img.type());

    bool ret_imwrite = false;
    ASSERT_NO_THROW(ret_imwrite = cv::imwrite(filenameOutput, img));
    ASSERT_TRUE(ret_imwrite);

    Mat img2;
    ASSERT_NO_THROW(img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED));
    ASSERT_FALSE(img2.empty());
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 0.01);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_Tiff, readWrite_32FC3_RAW)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC3_raw.tiff";
    const string filenameOutput = cv::tempfile(".tiff");

    Mat img;
    ASSERT_NO_THROW(img = cv::imread(filenameInput, IMREAD_UNCHANGED));
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC3, img.type());

    std::vector<int> params;
    params.push_back(IMWRITE_TIFF_COMPRESSION);
    params.push_back(IMWRITE_TIFF_COMPRESSION_NONE);

    bool ret_imwrite = false;
    ASSERT_NO_THROW(ret_imwrite = cv::imwrite(filenameOutput, img, params));
    ASSERT_TRUE(ret_imwrite);

    Mat img2;
    ASSERT_NO_THROW(img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED));
    ASSERT_FALSE(img2.empty());
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_Tiff, read_palette_color_image)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_palette_color_image.tif";

    Mat img;
    ASSERT_NO_THROW(img = cv::imread(filenameInput, IMREAD_UNCHANGED));
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_8UC3, img.type());
}

TEST(Imgcodecs_Tiff, read_palette_color_image_rgb_and_bgr)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test_palette_color_image.tif";

    Mat img_rgb, img_bgr;
    ASSERT_NO_THROW(img_rgb = cv::imread(filenameInput, IMREAD_COLOR_RGB));
    ASSERT_NO_THROW(img_bgr = cv::imread(filenameInput, IMREAD_COLOR_BGR));
    ASSERT_FALSE(img_rgb.empty());
    ASSERT_EQ(CV_8UC3, img_rgb.type());

    ASSERT_FALSE(img_bgr.empty());
    ASSERT_EQ(CV_8UC3, img_bgr.type());

    EXPECT_EQ(img_rgb.at<Vec3b>(32, 24), Vec3b(255, 0, 0));
    EXPECT_EQ(img_bgr.at<Vec3b>(32, 24), Vec3b(0, 0, 255));
}

TEST(Imgcodecs_Tiff, read_4_bit_palette_color_image)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/4-bit_palette_color.tif";

    Mat img;
    ASSERT_NO_THROW(img = cv::imread(filenameInput, IMREAD_UNCHANGED));
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_8UC3, img.type());
}

TEST(Imgcodecs_Tiff, readWrite_predictor)
{
    /* see issue #21871
     */
    const uchar sample_data[160] = {
        0xff, 0xff, 0xff, 0xff, 0x88, 0x88, 0xff, 0xff, 0x88, 0x88, 0xff, 0xff, 0xff, 0xff, 0xff, 0x88,
        0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00,
        0xff, 0x00, 0x00, 0x44, 0xff, 0xff, 0x88, 0xff, 0x33, 0x00, 0x66, 0xff, 0xff, 0x88, 0x00, 0x44,
        0x88, 0x00, 0x44, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x44, 0xff, 0xff, 0x11, 0x00, 0xff,
        0x11, 0x00, 0x88, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xff,
        0x11, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x33, 0x00, 0x88, 0xff, 0x00, 0x66, 0xff,
        0x11, 0x00, 0x66, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x44, 0x33, 0x00, 0xff, 0xff,
        0x88, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff,
        0xff, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0x33, 0x00, 0x00, 0x66, 0xff, 0xff,
        0xff, 0xff, 0x88, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0xff
    };

    cv::Mat mat(10, 16, CV_8UC1, (void*)sample_data);
    int methods[] = {
        IMWRITE_TIFF_COMPRESSION_NONE,     IMWRITE_TIFF_COMPRESSION_LZW,
        IMWRITE_TIFF_COMPRESSION_PACKBITS, IMWRITE_TIFF_COMPRESSION_DEFLATE,
        IMWRITE_TIFF_COMPRESSION_ADOBE_DEFLATE
    };
    for (size_t i = 0; i < sizeof(methods) / sizeof(int); i++)
    {
        string out = cv::tempfile(".tif");

        std::vector<int> params;
        params.push_back(IMWRITE_TIFF_COMPRESSION);
        params.push_back(methods[i]);
        params.push_back(IMWRITE_TIFF_PREDICTOR);
        params.push_back(IMWRITE_TIFF_PREDICTOR_HORIZONTAL);

        bool ret_imwrite = false;
        ASSERT_NO_THROW(ret_imwrite = cv::imwrite(out, mat, params));
        ASSERT_TRUE(ret_imwrite);

        Mat img;
        ASSERT_NO_THROW(img = cv::imread(out, IMREAD_UNCHANGED));
        ASSERT_FALSE(img.empty());

        ASSERT_EQ(0, cv::norm(mat, img, cv::NORM_INF));

        EXPECT_EQ(0, remove(out.c_str()));
    }
}

// See https://github.com/opencv/opencv/issues/23416

typedef std::pair<perf::MatType,bool> Imgcodes_Tiff_TypeAndComp;
typedef testing::TestWithParam< Imgcodes_Tiff_TypeAndComp > Imgcodecs_Tiff_Types;

TEST_P(Imgcodecs_Tiff_Types, readWrite_alltypes)
{
    const int mat_types = static_cast<int>(get<0>(GetParam()));
    const bool isCompAvailable = get<1>(GetParam());

    // Create a test image.
    const Mat src = cv::Mat::zeros( 120, 160, mat_types );
    {
        // Add noise to test compression.
        cv::Mat roi = cv::Mat(src, cv::Rect(0, 0, src.cols, src.rows/2));
        cv::randu(roi, cv::Scalar(0), cv::Scalar(256));
    }

    // Try to encode/decode the test image with LZW compression.
    std::vector<uchar> bufLZW;
    {
        std::vector<int> params;
        params.push_back(IMWRITE_TIFF_COMPRESSION);
        params.push_back(IMWRITE_TIFF_COMPRESSION_LZW);
        ASSERT_NO_THROW(cv::imencode(".tiff", src, bufLZW, params));

        Mat dstLZW;
        ASSERT_NO_THROW(cv::imdecode( bufLZW, IMREAD_UNCHANGED, &dstLZW));
        ASSERT_EQ(dstLZW.type(), src.type());
        ASSERT_EQ(dstLZW.size(), src.size());
        ASSERT_LE(cvtest::norm(dstLZW, src, NORM_INF | NORM_RELATIVE), 1e-3);
    }

    // Try to encode/decode the test image with RAW.
    std::vector<uchar> bufRAW;
    {
        std::vector<int> params;
        params.push_back(IMWRITE_TIFF_COMPRESSION);
        params.push_back(IMWRITE_TIFF_COMPRESSION_NONE);
        ASSERT_NO_THROW(cv::imencode(".tiff", src, bufRAW, params));

        Mat dstRAW;
        ASSERT_NO_THROW(cv::imdecode( bufRAW, IMREAD_UNCHANGED, &dstRAW));
        ASSERT_EQ(dstRAW.type(), src.type());
        ASSERT_EQ(dstRAW.size(), src.size());
        ASSERT_LE(cvtest::norm(dstRAW, src, NORM_INF | NORM_RELATIVE), 1e-3);
    }

    // Compare LZW and RAW streams.
    EXPECT_EQ(bufLZW == bufRAW, !isCompAvailable);
}

Imgcodes_Tiff_TypeAndComp all_types[] = {
    { CV_8UC1,  true  }, { CV_8UC3,  true  }, { CV_8UC4,  true  },
    { CV_8SC1,  true  }, { CV_8SC3,  true  }, { CV_8SC4,  true  },
    { CV_16UC1, true  }, { CV_16UC3, true  }, { CV_16UC4, true  },
    { CV_16SC1, true  }, { CV_16SC3, true  }, { CV_16SC4, true  },
    { CV_32SC1, true  }, { CV_32SC3, true  }, { CV_32SC4, true  },
    { CV_32FC1, false }, { CV_32FC3, false }, { CV_32FC4, false }, // No compression
    { CV_64FC1, false }, { CV_64FC3, false }, { CV_64FC4, false }  // No compression
};

INSTANTIATE_TEST_CASE_P(AllTypes, Imgcodecs_Tiff_Types, testing::ValuesIn(all_types));

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

TEST_P(Imgcodecs_Tiff_Modes, decode_multipage_use_memory_buffer_all_pages)
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
    const size_t page_count = sizeof(page_files) / sizeof(page_files[0]);
    vector<Mat> pages;

    FILE* fp = fopen(filename.c_str(), "rb");
    ASSERT_TRUE(fp != NULL);
    fseek(fp, 0, SEEK_END);
    const size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    std::vector<uchar> buf(file_size);
    const size_t actual_read = fread(&buf[0], 1, file_size, fp);
    fclose(fp);
    ASSERT_EQ(file_size, actual_read);
    ASSERT_EQ(file_size, static_cast<size_t>(buf.size()));

    bool res = imdecodemulti(buf, mode, pages);
    ASSERT_TRUE(res == true);
    ASSERT_EQ(page_count, pages.size());
    for (size_t i = 0; i < page_count; i++)
    {
        const Mat page = imread(root + page_files[i], mode);
        EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), page, pages[i]);
    }
}

TEST_P(Imgcodecs_Tiff_Modes, decode_multipage_use_memory_buffer_selected_pages)
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
    const size_t page_count = sizeof(page_files) / sizeof(page_files[0]);

    FILE* fp = fopen(filename.c_str(), "rb");
    ASSERT_TRUE(fp != NULL);
    fseek(fp, 0, SEEK_END);
    const size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    std::vector<uchar> buf(file_size);
    const size_t actual_read = fread(&buf[0], 1, file_size, fp);
    fclose(fp);
    ASSERT_EQ(file_size, actual_read);
    ASSERT_EQ(file_size, static_cast<size_t>(buf.size()));

    const Range range(1, page_count - 1);
    ASSERT_GE(range.size(), 1);

    vector<Mat> middle_pages_from_imread;
    for (int page_i = range.start; page_i < range.end; page_i++)
    {
        const Mat page = imread(root + page_files[page_i], mode);
        middle_pages_from_imread.push_back(page);
    }
    ASSERT_EQ(
        static_cast<size_t>(range.size()),
        static_cast<size_t>(middle_pages_from_imread.size())
    );

    vector<Mat> middle_pages_from_imdecodemulti;
    const bool res = imdecodemulti(buf, mode, middle_pages_from_imdecodemulti, range);
    ASSERT_TRUE(res == true);
    EXPECT_EQ(middle_pages_from_imread.size(), middle_pages_from_imdecodemulti.size());

    for (int i = 0, e = range.size(); i < e; i++)
    {
        EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0),
            middle_pages_from_imread[i],
            middle_pages_from_imdecodemulti[i]);
    }
}

const int all_modes[] =
{
    IMREAD_UNCHANGED,
    IMREAD_GRAYSCALE,
    IMREAD_COLOR,
    IMREAD_COLOR_RGB,
    IMREAD_ANYDEPTH,
    IMREAD_ANYCOLOR
};

INSTANTIATE_TEST_CASE_P(AllModes, Imgcodecs_Tiff_Modes, testing::ValuesIn(all_modes));

//==================================================================================================

TEST(Imgcodecs_Tiff_Modes, write_multipage)
{
    const string root = cvtest::TS::ptr()->get_data_path();
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
        const Mat page = imread(root + page_files[i], IMREAD_REDUCED_GRAYSCALE_8 + (int)i);
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
    EXPECT_EQ(0, remove(tmp_filename.c_str()));
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


TEST(Imgcodecs_Tiff, decode_black_and_write_image_pr12989_grayscale)
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

TEST(Imgcodecs_Tiff, decode_black_and_write_image_pr17275_grayscale)
{
    const string filename = cvtest::findDataFile("readwrite/bitsperpixel1_min.tiff");
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

TEST(Imgcodecs_Tiff, decode_black_and_write_image_pr17275_default)
{
    const string filename = cvtest::findDataFile("readwrite/bitsperpixel1_min.tiff");
    cv::Mat img;
    ASSERT_NO_THROW(img = cv::imread(filename));  // by default image type is CV_8UC3
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(64, img.cols);
    EXPECT_EQ(64, img.rows);
    EXPECT_EQ(CV_8UC3, img.type()) << cv::typeToString(img.type());
}

TEST(Imgcodecs_Tiff, count_multipage)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    {
        const string filename = root + "readwrite/multipage.tif";
        ASSERT_EQ((size_t)6, imcount(filename));
    }
    {
        const string filename = root + "readwrite/test32FC3_raw.tiff";
        ASSERT_EQ((size_t)1, imcount(filename));
    }
}

TEST(Imgcodecs_Tiff, read_multipage_indexed)
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
    const int page_count = sizeof(page_files) / sizeof(page_files[0]);
    vector<Mat> single_pages;
    for (int i = 0; i < page_count; i++)
    {
        // imread and imreadmulti have different default values for the flag
        const Mat page = imread(root + page_files[i], IMREAD_ANYCOLOR);
        single_pages.push_back(page);
    }
    ASSERT_EQ((size_t)page_count, single_pages.size());

    {
        SCOPED_TRACE("Edge Cases");
        vector<Mat> multi_pages;
        bool res = imreadmulti(filename, multi_pages, 0, 0);
        // If we asked for 0 images and we successfully read 0 images should this be false ?
        ASSERT_TRUE(res == false);
        ASSERT_EQ((size_t)0, multi_pages.size());
        res = imreadmulti(filename, multi_pages, 0, 123123);
        ASSERT_TRUE(res == true);
        ASSERT_EQ((size_t)6, multi_pages.size());
    }

    {
        SCOPED_TRACE("Read all with indices");
        vector<Mat> multi_pages;
        bool res = imreadmulti(filename, multi_pages, 0, 6);
        ASSERT_TRUE(res == true);
        ASSERT_EQ((size_t)page_count, multi_pages.size());
        for (int i = 0; i < page_count; i++)
        {
            EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), multi_pages[i], single_pages[i]);
        }
    }

    {
        SCOPED_TRACE("Read one by one");
        vector<Mat> multi_pages;
        for (int i = 0; i < page_count; i++)
        {
            bool res = imreadmulti(filename, multi_pages, i, 1);
            ASSERT_TRUE(res == true);
            ASSERT_EQ((size_t)1, multi_pages.size());
            EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), multi_pages[0], single_pages[i]);
            multi_pages.clear();
        }
    }

    {
        SCOPED_TRACE("Read multiple at a time");
        vector<Mat> multi_pages;
        for (int i = 0; i < page_count/2; i++)
        {
            bool res = imreadmulti(filename, multi_pages, i*2, 2);
            ASSERT_TRUE(res == true);
            ASSERT_EQ((size_t)2, multi_pages.size());
            EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), multi_pages[0], single_pages[i * 2]) << i;
            EXPECT_PRED_FORMAT2(cvtest::MatComparator(0, 0), multi_pages[1], single_pages[i * 2 + 1]);
            multi_pages.clear();
        }
    }
}

TEST(Imgcodecs_Tiff, read_bigtiff_images)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenamesInput[] = {
        "readwrite/BigTIFF.tif",
        "readwrite/BigTIFFMotorola.tif",
        "readwrite/BigTIFFLong.tif",
        "readwrite/BigTIFFLong8.tif",
        "readwrite/BigTIFFMotorolaLongStrips.tif",
        "readwrite/BigTIFFLong8Tiles.tif",
        "readwrite/BigTIFFSubIFD4.tif",
        "readwrite/BigTIFFSubIFD8.tif"
    };

    for (int i = 0; i < 8; i++)
    {
        const Mat bigtiff_img = imread(root + filenamesInput[i], IMREAD_UNCHANGED);
        ASSERT_FALSE(bigtiff_img.empty());
        EXPECT_EQ(64, bigtiff_img.cols);
        EXPECT_EQ(64, bigtiff_img.rows);
        ASSERT_EQ(CV_8UC3, bigtiff_img.type());
    }
}

#endif

}} // namespace
