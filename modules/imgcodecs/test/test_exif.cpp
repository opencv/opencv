// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#include <string>
#include <vector>

#include "test_precomp.hpp"

namespace opencv_test { namespace {

/**
 * Test to check whether the EXIF orientation tag was processed successfully or not.
 * The test uses a set of 8 images named testExifOrientation_{1 to 8}.(extension).
 * Each test image is a 10x10 square, divided into four smaller sub-squares:
 * (R corresponds to Red, G to Green, B to Blue, W to White)
 * ---------             ---------
 * | R | G |             | G | R |
 * |-------| - (tag 1)   |-------| - (tag 2)
 * | B | W |             | W | B |
 * ---------             ---------
 *
 * ---------             ---------
 * | W | B |             | B | W |
 * |-------| - (tag 3)   |-------| - (tag 4)
 * | G | R |             | R | G |
 * ---------             ---------
 *
 * ---------             ---------
 * | R | B |             | G | W |
 * |-------| - (tag 5)   |-------| - (tag 6)
 * | G | W |             | R | B |
 * ---------             ---------
 *
 * ---------             ---------
 * | W | G |             | B | R |
 * |-------| - (tag 7)   |-------| - (tag 8)
 * | B | R |             | W | G |
 * ---------             ---------
 *
 *
 * Each image contains an EXIF field with an orientation tag (0x112).
 * After reading each image and applying the orientation tag,
 * the resulting image should be:
 * ---------
 * | R | G |
 * |-------|
 * | B | W |
 * ---------
 *
 * Note:
 * The flags parameter of the imread function is set as IMREAD_COLOR | IMREAD_ANYCOLOR | IMREAD_ANYDEPTH.
 * Using this combination is an undocumented trick to load images similarly to the IMREAD_UNCHANGED flag,
 * preserving the alpha channel (if present) while also applying the orientation.
 */

typedef testing::TestWithParam<string> Exif;

TEST_P(Exif, exif_orientation)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + GetParam();
    const int colorThresholdHigh = 250;
    const int colorThresholdLow = 5;

    // Refer to the note in the explanation above.
    Mat m_img = imread(filename, IMREAD_COLOR | IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
    ASSERT_FALSE(m_img.empty());

    if (m_img.channels() == 3)
    {
        Vec3b vec;

        //Checking the first quadrant (with supposed red)
        vec = m_img.at<Vec3b>(2, 2); //some point inside the square
        EXPECT_LE(vec.val[0], colorThresholdLow);
        EXPECT_LE(vec.val[1], colorThresholdLow);
        EXPECT_GE(vec.val[2], colorThresholdHigh);

        //Checking the second quadrant (with supposed green)
        vec = m_img.at<Vec3b>(2, 7);  //some point inside the square
        EXPECT_LE(vec.val[0], colorThresholdLow);
        EXPECT_GE(vec.val[1], colorThresholdHigh);
        EXPECT_LE(vec.val[2], colorThresholdLow);

        //Checking the third quadrant (with supposed blue)
        vec = m_img.at<Vec3b>(7, 2);  //some point inside the square
        EXPECT_GE(vec.val[0], colorThresholdHigh);
        EXPECT_LE(vec.val[1], colorThresholdLow);
        EXPECT_LE(vec.val[2], colorThresholdLow);
    }
    else
    {
        Vec4b vec;

        //Checking the first quadrant (with supposed red)
        vec = m_img.at<Vec4b>(2, 2); //some point inside the square
        EXPECT_LE(vec.val[0], colorThresholdLow);
        EXPECT_LE(vec.val[1], colorThresholdLow);
        EXPECT_GE(vec.val[2], colorThresholdHigh);

        //Checking the second quadrant (with supposed green)
        vec = m_img.at<Vec4b>(2, 7);  //some point inside the square
        EXPECT_LE(vec.val[0], colorThresholdLow);
        EXPECT_GE(vec.val[1], colorThresholdHigh);
        EXPECT_LE(vec.val[2], colorThresholdLow);

        //Checking the third quadrant (with supposed blue)
        vec = m_img.at<Vec4b>(7, 2);  //some point inside the square
        EXPECT_GE(vec.val[0], colorThresholdHigh);
        EXPECT_LE(vec.val[1], colorThresholdLow);
        EXPECT_LE(vec.val[2], colorThresholdLow);
    }
}

const std::vector<std::string> exif_files
{
#ifdef HAVE_JPEG
    "readwrite/testExifOrientation_1.jpg",
    "readwrite/testExifOrientation_2.jpg",
    "readwrite/testExifOrientation_3.jpg",
    "readwrite/testExifOrientation_4.jpg",
    "readwrite/testExifOrientation_5.jpg",
    "readwrite/testExifOrientation_6.jpg",
    "readwrite/testExifOrientation_7.jpg",
    "readwrite/testExifOrientation_8.jpg",
#endif
#ifdef OPENCV_IMGCODECS_PNG_WITH_EXIF
    "readwrite/testExifOrientation_1.png",
    "readwrite/testExifOrientation_2.png",
    "readwrite/testExifOrientation_3.png",
    "readwrite/testExifOrientation_4.png",
    "readwrite/testExifOrientation_5.png",
    "readwrite/testExifOrientation_6.png",
    "readwrite/testExifOrientation_7.png",
    "readwrite/testExifOrientation_8.png",
#endif
#ifdef HAVE_AVIF
    "readwrite/testExifOrientation_1.avif",
    "readwrite/testExifOrientation_2.avif",
    "readwrite/testExifOrientation_3.avif",
    "readwrite/testExifOrientation_4.avif",
    "readwrite/testExifOrientation_5.avif",
    "readwrite/testExifOrientation_6.avif",
    "readwrite/testExifOrientation_7.avif",
    "readwrite/testExifOrientation_8.avif",
#endif
};

INSTANTIATE_TEST_CASE_P(Imgcodecs, Exif,
                        testing::ValuesIn(exif_files));

static Mat makeCirclesImage(Size size, int type, int nbits)
{
    Mat img(size, type);
    img.setTo(Scalar::all(0));
    RNG& rng = theRNG();
    int maxval = (int)(1 << nbits);
    for (int i = 0; i < 100; i++) {
        int x = rng.uniform(0, img.cols);
        int y = rng.uniform(0, img.rows);
        int radius = rng.uniform(5, std::min(img.cols, img.rows)/5);
        int b = rng.uniform(0, maxval);
        int g = rng.uniform(0, maxval);
        int r = rng.uniform(0, maxval);
        circle(img, Point(x, y), radius, Scalar(b, g, r), -1, LINE_AA);
    }
    return img;
}

#ifdef HAVE_AVIF
TEST(Imgcodecs_Avif, ReadWriteWithExif)
{
    static const uchar exif_data[] = {
        'M', 'M', 0, '*', 0, 0, 0, 8, 0, 10, 1, 0, 0, 4, 0, 0, 0, 1, 0, 0, 5,
        0, 1, 1, 0, 4, 0, 0, 0, 1, 0, 0, 2, 208, 1, 2, 0, 3, 0, 0, 0, 1,
        0, 10, 0, 0, 1, 18, 0, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 14, 0, 2, 0, 0,
        0, '"', 0, 0, 0, 176, 1, '1', 0, 2, 0, 0, 0, 7, 0, 0, 0, 210, 1, 26,
        0, 5, 0, 0, 0, 1, 0, 0, 0, 218, 1, 27, 0, 5, 0, 0, 0, 1, 0, 0, 0,
        226, 1, '(', 0, 3, 0, 0, 0, 1, 0, 2, 0, 0, 135, 'i', 0, 4, 0, 0, 0,
        1, 0, 0, 0, 134, 0, 0, 0, 0, 0, 3, 144, 0, 0, 7, 0, 0, 0, 4, '0', '2',
        '2', '1', 160, 2, 0, 4, 0, 0, 0, 1, 0, 0, 5, 0, 160, 3, 0, 4, 0, 0,
        0, 1, 0, 0, 2, 208, 0, 0, 0, 0, 'S', 'a', 'm', 'p', 'l', 'e', ' ', '1', '0',
        '-', 'b', 'i', 't', ' ', 'i', 'm', 'a', 'g', 'e', ' ', 'w', 'i', 't', 'h', ' ',
        'm', 'e', 't', 'a', 'd', 'a', 't', 'a', 0, 'O', 'p', 'e', 'n', 'C', 'V', 0, 0,
        0, 0, 0, 'H', 0, 0, 0, 1, 0, 0, 0, 'H', 0, 0, 0, 1
    };

    int avif_nbits = 10;
    int avif_speed = 10;
    int avif_quality = 85;
    int imgdepth = avif_nbits > 8 ? CV_16U : CV_8U;
    int imgtype = CV_MAKETYPE(imgdepth, 3);
    const string outputname = cv::tempfile(".avif");
    Mat img = makeCirclesImage(Size(1280, 720), imgtype, avif_nbits);

    std::vector<int> metadata_types = {IMAGE_METADATA_EXIF};
    std::vector<std::vector<uchar> > metadata(1);
    metadata[0].assign(exif_data, exif_data + sizeof(exif_data));

    std::vector<int> write_params = {
        IMWRITE_AVIF_DEPTH, avif_nbits,
        IMWRITE_AVIF_SPEED, avif_speed,
        IMWRITE_AVIF_QUALITY, avif_quality
    };

    imwriteWithMetadata(outputname, img, metadata_types, metadata, write_params);
    std::vector<uchar> compressed;
    imencodeWithMetadata(outputname, img, metadata_types, metadata, compressed, write_params);

    std::vector<int> read_metadata_types, read_metadata_types2;
    std::vector<std::vector<uchar> > read_metadata, read_metadata2;
    Mat img2 = imreadWithMetadata(outputname, read_metadata_types, read_metadata, IMREAD_UNCHANGED);
    Mat img3 = imdecodeWithMetadata(compressed, read_metadata_types2, read_metadata2, IMREAD_UNCHANGED);
    EXPECT_EQ(img2.cols, img.cols);
    EXPECT_EQ(img2.rows, img.rows);
    EXPECT_EQ(img2.type(), imgtype);
    EXPECT_EQ(read_metadata_types, read_metadata_types2);
    EXPECT_GE(read_metadata_types.size(), 1u);
    EXPECT_EQ(read_metadata, read_metadata2);
    EXPECT_EQ(read_metadata_types[0], IMAGE_METADATA_EXIF);
    EXPECT_EQ(read_metadata_types.size(), read_metadata.size());
    EXPECT_EQ(read_metadata[0], metadata[0]);
    EXPECT_EQ(cv::norm(img2, img3, NORM_INF), 0.);
    double mse = cv::norm(img, img2, NORM_L2SQR)/(img.rows*img.cols);
    EXPECT_LT(mse, 1500);
    remove(outputname.c_str());
}
#endif // HAVE_AVIF

TEST(Imgcodecs_Jpeg, ReadWriteWithExif)
{
    static const uchar exif_data[] = {
        'M', 'M', 0, '*', 0, 0, 0, 8, 0, 10, 1, 0, 0, 4, 0, 0, 0, 1, 0, 0, 5,
        0, 1, 1, 0, 4, 0, 0, 0, 1, 0, 0, 2, 208, 1, 2, 0, 3, 0, 0, 0, 1,
        0, 8, 0, 0, 1, 18, 0, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 14, 0, 2, 0, 0,
        0, '!', 0, 0, 0, 176, 1, '1', 0, 2, 0, 0, 0, 7, 0, 0, 0, 210, 1, 26,
        0, 5, 0, 0, 0, 1, 0, 0, 0, 218, 1, 27, 0, 5, 0, 0, 0, 1, 0, 0, 0,
        226, 1, '(', 0, 3, 0, 0, 0, 1, 0, 2, 0, 0, 135, 'i', 0, 4, 0, 0, 0,
        1, 0, 0, 0, 134, 0, 0, 0, 0, 0, 3, 144, 0, 0, 7, 0, 0, 0, 4, '0', '2',
        '2', '1', 160, 2, 0, 4, 0, 0, 0, 1, 0, 0, 5, 0, 160, 3, 0, 4, 0, 0,
        0, 1, 0, 0, 2, 208, 0, 0, 0, 0, 'S', 'a', 'm', 'p', 'l', 'e', ' ', '8', '-',
        'b', 'i', 't', ' ', 'i', 'm', 'a', 'g', 'e', ' ', 'w', 'i', 't', 'h', ' ', 'm',
        'e', 't', 'a', 'd', 'a', 't', 'a', 0, 0, 'O', 'p', 'e', 'n', 'C', 'V', 0, 0,
        0, 0, 0, 'H', 0, 0, 0, 1, 0, 0, 0, 'H', 0, 0, 0, 1
    };

    int jpeg_quality = 95;
    int imgtype = CV_MAKETYPE(CV_8U, 3);
    const string outputname = cv::tempfile(".jpeg");
    Mat img = makeCirclesImage(Size(1280, 720), imgtype, 8);

    std::vector<int> metadata_types = {IMAGE_METADATA_EXIF};
    std::vector<std::vector<uchar> > metadata(1);
    metadata[0].assign(exif_data, exif_data + sizeof(exif_data));

    std::vector<int> write_params = {
        IMWRITE_JPEG_QUALITY, jpeg_quality
    };

    imwriteWithMetadata(outputname, img, metadata_types, metadata, write_params);
    std::vector<uchar> compressed;
    imencodeWithMetadata(outputname, img, metadata_types, metadata, compressed, write_params);

    std::vector<int> read_metadata_types, read_metadata_types2;
    std::vector<std::vector<uchar> > read_metadata, read_metadata2;
    Mat img2 = imreadWithMetadata(outputname, read_metadata_types, read_metadata, IMREAD_UNCHANGED);
    Mat img3 = imdecodeWithMetadata(compressed, read_metadata_types2, read_metadata2, IMREAD_UNCHANGED);
    EXPECT_EQ(img2.cols, img.cols);
    EXPECT_EQ(img2.rows, img.rows);
    EXPECT_EQ(img2.type(), imgtype);
    EXPECT_EQ(read_metadata_types, read_metadata_types2);
    EXPECT_GE(read_metadata_types.size(), 1u);
    EXPECT_EQ(read_metadata, read_metadata2);
    EXPECT_EQ(read_metadata_types[0], IMAGE_METADATA_EXIF);
    EXPECT_EQ(read_metadata_types.size(), read_metadata.size());
    EXPECT_EQ(read_metadata[0], metadata[0]);
    EXPECT_EQ(cv::norm(img2, img3, NORM_INF), 0.);
    double mse = cv::norm(img, img2, NORM_L2SQR)/(img.rows*img.cols);
    EXPECT_LT(mse, 80);
    remove(outputname.c_str());
}

TEST(Imgcodecs_Png, ReadWriteWithExif)
{
    static const uchar exif_data[] = {
        'M', 'M', 0, '*', 0, 0, 0, 8, 0, 10, 1, 0, 0, 4, 0, 0, 0, 1, 0, 0, 5,
        0, 1, 1, 0, 4, 0, 0, 0, 1, 0, 0, 2, 208, 1, 2, 0, 3, 0, 0, 0, 1,
        0, 8, 0, 0, 1, 18, 0, 3, 0, 0, 0, 1, 0, 1, 0, 0, 1, 14, 0, 2, 0, 0,
        0, '!', 0, 0, 0, 176, 1, '1', 0, 2, 0, 0, 0, 7, 0, 0, 0, 210, 1, 26,
        0, 5, 0, 0, 0, 1, 0, 0, 0, 218, 1, 27, 0, 5, 0, 0, 0, 1, 0, 0, 0,
        226, 1, '(', 0, 3, 0, 0, 0, 1, 0, 2, 0, 0, 135, 'i', 0, 4, 0, 0, 0,
        1, 0, 0, 0, 134, 0, 0, 0, 0, 0, 3, 144, 0, 0, 7, 0, 0, 0, 4, '0', '2',
        '2', '1', 160, 2, 0, 4, 0, 0, 0, 1, 0, 0, 5, 0, 160, 3, 0, 4, 0, 0,
        0, 1, 0, 0, 2, 208, 0, 0, 0, 0, 'S', 'a', 'm', 'p', 'l', 'e', ' ', '8', '-',
        'b', 'i', 't', ' ', 'i', 'm', 'a', 'g', 'e', ' ', 'w', 'i', 't', 'h', ' ', 'm',
        'e', 't', 'a', 'd', 'a', 't', 'a', 0, 0, 'O', 'p', 'e', 'n', 'C', 'V', 0, 0,
        0, 0, 0, 'H', 0, 0, 0, 1, 0, 0, 0, 'H', 0, 0, 0, 1
    };

    int png_compression = 3;
    int imgtype = CV_MAKETYPE(CV_8U, 3);
    const string outputname = cv::tempfile(".png");
    Mat img = makeCirclesImage(Size(1280, 720), imgtype, 8);

    std::vector<int> metadata_types = {IMAGE_METADATA_EXIF};
    std::vector<std::vector<uchar> > metadata(1);
    metadata[0].assign(exif_data, exif_data + sizeof(exif_data));

    std::vector<int> write_params = {
        IMWRITE_PNG_COMPRESSION, png_compression
    };

    imwriteWithMetadata(outputname, img, metadata_types, metadata, write_params);
    std::vector<uchar> compressed;
    imencodeWithMetadata(outputname, img, metadata_types, metadata, compressed, write_params);

    std::vector<int> read_metadata_types, read_metadata_types2;
    std::vector<std::vector<uchar> > read_metadata, read_metadata2;
    Mat img2 = imreadWithMetadata(outputname, read_metadata_types, read_metadata, IMREAD_UNCHANGED);
    Mat img3 = imdecodeWithMetadata(compressed, read_metadata_types2, read_metadata2, IMREAD_UNCHANGED);
    EXPECT_EQ(img2.cols, img.cols);
    EXPECT_EQ(img2.rows, img.rows);
    EXPECT_EQ(img2.type(), imgtype);
    EXPECT_EQ(read_metadata_types, read_metadata_types2);
    EXPECT_GE(read_metadata_types.size(), 1u);
    EXPECT_EQ(read_metadata, read_metadata2);
    EXPECT_EQ(read_metadata_types[0], IMAGE_METADATA_EXIF);
    EXPECT_EQ(read_metadata_types.size(), read_metadata.size());
    EXPECT_EQ(read_metadata[0], metadata[0]);
    EXPECT_EQ(cv::norm(img2, img3, NORM_INF), 0.);
    double mse = cv::norm(img, img2, NORM_L2SQR)/(img.rows*img.cols);
    EXPECT_EQ(mse, 0); // png is lossless
    remove(outputname.c_str());
}

static size_t locateString(const uchar* exif, size_t exif_size, const std::string& pattern)
{
    size_t plen = pattern.size();
    for (size_t i = 0; i + plen <= exif_size; i++) {
        if (exif[i] == pattern[0] && memcmp(&exif[i], pattern.c_str(), plen) == 0)
            return i;
    }
    return 0xFFFFFFFFu;
}

typedef std::tuple<std::string, size_t, std::string, size_t> ReadExif_Sanity_Params;
typedef testing::TestWithParam<ReadExif_Sanity_Params> ReadExif_Sanity;

TEST_P(ReadExif_Sanity, Check)
{
    std::string filename = get<0>(GetParam());
    size_t exif_size = get<1>(GetParam());
    std::string pattern = get<2>(GetParam());
    size_t ploc = get<3>(GetParam());

    const string root = cvtest::TS::ptr()->get_data_path();
    filename = root + filename;

    std::vector<int> metadata_types;
    std::vector<Mat> metadata;
    Mat img = imreadWithMetadata(filename, metadata_types, metadata, 1);

    EXPECT_EQ(img.type(), CV_8UC3);
    ASSERT_GE(metadata_types.size(), 1u);
    EXPECT_EQ(metadata_types.size(), metadata.size());
    const Mat& exif = metadata[IMAGE_METADATA_EXIF];
    EXPECT_EQ(exif.type(), CV_8U);
    EXPECT_EQ(exif.total(), exif_size);
    ASSERT_GE(exif_size, 26u); // minimal exif should take at least 26 bytes
                                 // (the header + IDF0 with at least 1 entry).
    EXPECT_TRUE(exif.data[0] == 'I' || exif.data[0] == 'M');
    EXPECT_EQ(exif.data[0], exif.data[1]);
    EXPECT_EQ(locateString(exif.data, exif_size, pattern), ploc);
}

static const std::vector<ReadExif_Sanity_Params> exif_sanity_params
{
#ifdef HAVE_JPEG
    {"readwrite/testExifOrientation_3.jpg", 916, "Photoshop", 120},
#endif
#ifdef OPENCV_IMGCODECS_PNG_WITH_EXIF
    {"readwrite/testExifOrientation_5.png", 112, "ExifTool", 102},
#endif
#ifdef HAVE_AVIF
    {"readwrite/testExifOrientation_7.avif", 913, "Photoshop", 120},
#endif
};

INSTANTIATE_TEST_CASE_P(Imgcodecs, ReadExif_Sanity,
                        testing::ValuesIn(exif_sanity_params));

}}
