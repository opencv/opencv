// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"
#include "test_common.hpp"

namespace opencv_test {

static
Mat generateTestImageBGR_()
{
    Size sz(640, 480);
    Mat result(sz, CV_8UC3, Scalar::all(0));

    const string fname = cvtest::findDataFile("../cv/shared/baboon.png");
    Mat image = imread(fname, IMREAD_COLOR);
    CV_Assert(!image.empty());
    CV_CheckEQ(image.size(), Size(512, 512), "");
    Rect roi((640-512) / 2, 0, 512, 480);
    image(Rect(0, 0, 512, 480)).copyTo(result(roi));
    result(Rect(0,  0, 5, 5)).setTo(Scalar(0, 0, 255));  // R
    result(Rect(5,  0, 5, 5)).setTo(Scalar(0, 255, 0));  // G
    result(Rect(10, 0, 5, 5)).setTo(Scalar(255, 0, 0));  // B
    result(Rect(0,  5, 5, 5)).setTo(Scalar(128, 128, 128));  // gray
    //imshow("test_image", result); waitKey();
    return result;
}
Mat generateTestImageBGR()
{
    static Mat image = generateTestImageBGR_();  // initialize once
    CV_Assert(!image.empty());
    return image;
}

static
Mat generateTestImageGrayscale_()
{
    Mat imageBGR = generateTestImageBGR();
    CV_Assert(!imageBGR.empty());

    Mat result;
    cvtColor(imageBGR, result, COLOR_BGR2GRAY);
    return result;
}
Mat generateTestImageGrayscale()
{
    static Mat image = generateTestImageGrayscale_();  // initialize once
    return image;
}

void readFileBytes(const std::string& fname, std::vector<unsigned char>& buf)
{
    FILE * wfile = fopen(fname.c_str(), "rb");
    if (wfile != NULL)
    {
        fseek(wfile, 0, SEEK_END);
        size_t wfile_size = ftell(wfile);
        fseek(wfile, 0, SEEK_SET);

        buf.resize(wfile_size);
        size_t data_size = fread(&buf[0], 1, wfile_size, wfile);
        fclose(wfile);

        EXPECT_EQ(data_size, wfile_size);
    }
}

TEST(Imgcodecs_Image, imwrite_png_invalid_two_channel)
{
    const Mat src(2, 2, CV_8UC2, Scalar::all(0));
    const string filename = cv::tempfile(".png");
    EXPECT_THROW(imwrite(filename, src), cv::Exception);
    remove(filename.c_str());
}

#ifdef CV_Bool

static Mat makeBoolImage(int rows, int cols, int channels, bool value)
{
    Mat image(rows, cols, CV_MAKETYPE(CV_Bool, channels));
    for (int y = 0; y < image.rows; ++y)
    {
        bool* row = image.ptr<bool>(y);
        for (int x = 0; x < image.cols * image.channels(); ++x)
        {
            row[x] = value;
        }
    }
    return image;
}

static Mat imwriteReadUnchanged(const Mat& src, const string& ext)
{
    const string filename = cv::tempfile(ext.c_str());
    bool ret = false;
    EXPECT_NO_THROW(ret = imwrite(filename, src));
    EXPECT_TRUE(ret);

    Mat dst;
    if (ret)
    {
        EXPECT_NO_THROW(dst = imread(filename, IMREAD_UNCHANGED));
    }
    remove(filename.c_str());
    return dst;
}

static Mat imencodeDecodeUnchanged(const Mat& src, const string& ext)
{
    vector<uchar> buf;
    bool ret = false;
    EXPECT_NO_THROW(ret = imencode(ext, src, buf));
    EXPECT_TRUE(ret);

    Mat dst;
    if (ret)
    {
        EXPECT_NO_THROW(dst = imdecode(buf, IMREAD_UNCHANGED));
    }
    return dst;
}

TEST(Imgcodecs_Bool, imwrite_png_mixed_grayscale)
{
    Mat src = makeBoolImage(2, 3, 1, false);
    src.ptr<bool>(0)[1] = true;
    src.ptr<bool>(1)[0] = true;
    src.ptr<bool>(1)[2] = true;

    Mat dst = imwriteReadUnchanged(src, ".png");
    ASSERT_FALSE(dst.empty());
    ASSERT_EQ(CV_8UC1, dst.type());

    Mat expected = (Mat_<uchar>(2, 3) << 0, 255, 0, 255, 0, 255);
    EXPECT_EQ(0, cv::norm(dst, expected, NORM_INF));
}

TEST(Imgcodecs_Bool, imencode_png_mixed_grayscale)
{
    Mat src = makeBoolImage(2, 3, 1, false);
    src.ptr<bool>(0)[0] = true;
    src.ptr<bool>(0)[2] = true;
    src.ptr<bool>(1)[1] = true;

    Mat dst = imencodeDecodeUnchanged(src, ".png");
    ASSERT_FALSE(dst.empty());
    ASSERT_EQ(CV_8UC1, dst.type());

    Mat expected = (Mat_<uchar>(2, 3) << 255, 0, 255, 0, 255, 0);
    EXPECT_EQ(0, cv::norm(dst, expected, NORM_INF));
}

TEST(Imgcodecs_Bool, imwrite_png_three_channel)
{
    Mat src = makeBoolImage(2, 2, 3, false);
    bool* row0 = src.ptr<bool>(0);
    row0[0] = true;
    row0[2] = true;
    row0[4] = true;
    bool* row1 = src.ptr<bool>(1);
    row1[1] = true;
    row1[3] = true;
    row1[5] = true;

    Mat dst = imwriteReadUnchanged(src, ".png");
    ASSERT_FALSE(dst.empty());
    ASSERT_EQ(CV_8UC3, dst.type());

    EXPECT_EQ(Vec3b(255, 0, 255), dst.at<Vec3b>(0, 0));
    EXPECT_EQ(Vec3b(0, 255, 0), dst.at<Vec3b>(0, 1));
    EXPECT_EQ(Vec3b(0, 255, 0), dst.at<Vec3b>(1, 0));
    EXPECT_EQ(Vec3b(255, 0, 255), dst.at<Vec3b>(1, 1));
}

TEST(Imgcodecs_Bool, imwrite_png_all_false_all_true)
{
    Mat all_false = imwriteReadUnchanged(makeBoolImage(3, 4, 1, false), ".png");
    ASSERT_FALSE(all_false.empty());
    ASSERT_EQ(CV_8UC1, all_false.type());
    EXPECT_EQ(0, countNonZero(all_false));

    Mat all_true = imwriteReadUnchanged(makeBoolImage(3, 4, 1, true), ".png");
    ASSERT_FALSE(all_true.empty());
    ASSERT_EQ(CV_8UC1, all_true.type());
    EXPECT_EQ(0, cv::norm(all_true, Mat(all_true.size(), CV_8UC1, Scalar::all(255)), NORM_INF));
}

TEST(Imgcodecs_Bool, imwrite_png_invalid_two_channel_bool)
{
    const Mat src = makeBoolImage(2, 2, 2, false);
    const string filename = cv::tempfile(".png");
    EXPECT_THROW(imwrite(filename, src), cv::Exception);
    remove(filename.c_str());
}

#endif

}  // namespace
