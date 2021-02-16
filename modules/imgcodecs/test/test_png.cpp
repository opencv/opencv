// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_PNG

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

    // Fourth Test : Read PNG without alpha, imread flag 1
    img = imread(root + "readwrite/color_palette_no_alpha.png", IMREAD_COLOR);
    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(img.channels() == 3);

    // pixel is red in BGR
    EXPECT_EQ(img.at<Vec3b>(0, 0), Vec3b(0, 0, 255));
    EXPECT_EQ(img.at<Vec3b>(0, 1), Vec3b(0, 0, 255));
}

/**
 * Test for check whether reading exif orientation tag was processed successfully or not
 * The test info is the set of 8 images named testExifRotate_{1 to 8}.png
 * The test image is the square 10x10 points divided by four sub-squares:
 * (R corresponds to Red, G to Green, B to Blue, W to white)
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
 * Every image contains exif field with orientation tag (0x112)
 * After reading each image and applying the orientation tag,
 * the resulting image should be:
 * ---------
 * | R | G |
 * |-------|
 * | B | W |
 * ---------
 *
 */

typedef testing::TestWithParam<string> Imgcodecs_PNG_Exif;

// Solution to issue 16579: PNG read doesn't support Exif orientation data
#ifdef OPENCV_IMGCODECS_PNG_WITH_EXIF
TEST_P(Imgcodecs_PNG_Exif, exif_orientation)
#else
TEST_P(Imgcodecs_PNG_Exif, DISABLED_exif_orientation)
#endif
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + GetParam();
    const int colorThresholdHigh = 250;
    const int colorThresholdLow = 5;

    Mat m_img = imread(filename);
    ASSERT_FALSE(m_img.empty());
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

const string exif_files[] =
{
    "readwrite/testExifOrientation_1.png",
    "readwrite/testExifOrientation_2.png",
    "readwrite/testExifOrientation_3.png",
    "readwrite/testExifOrientation_4.png",
    "readwrite/testExifOrientation_5.png",
    "readwrite/testExifOrientation_6.png",
    "readwrite/testExifOrientation_7.png",
    "readwrite/testExifOrientation_8.png"
};

INSTANTIATE_TEST_CASE_P(ExifFiles, Imgcodecs_PNG_Exif,
    testing::ValuesIn(exif_files));

#endif // HAVE_PNG

}} // namespace
