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

TEST(Imgcodecs_Bool, imwrite_true_maps_to_255)
{
    Mat src(2, 3, CV_BoolC1, Scalar::all(0));
    src.at<bool>(0, 1) = true;
    src.at<bool>(1, 0) = true;
    src.at<bool>(1, 2) = true;

    const string filename = cv::tempfile(".png");
    ASSERT_TRUE(imwrite(filename, src));

    Mat dst = imread(filename, IMREAD_UNCHANGED);
    remove(filename.c_str());

    ASSERT_FALSE(dst.empty());
    ASSERT_EQ(CV_8UC1, dst.type());

    Mat dst_bool;
    dst.convertTo(dst_bool, CV_Bool);
    EXPECT_EQ(0, cv::norm(dst_bool, src, NORM_INF));
}

}  // namespace
