#include "test_precomp.hpp"

using namespace cv;
using namespace std;
using namespace std::tr1;

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
    cvtColor(original_image, gray_by_cvt, CV_BGR2GRAY);

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

#endif // HAVE_PNG
