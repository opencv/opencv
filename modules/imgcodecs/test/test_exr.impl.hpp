// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

//#define GENERATE_DATA

namespace opencv_test { namespace {

TEST(Imgcodecs_EXR, readWrite_32FC1)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC1.exr";
    const string filenameOutput = cv::tempfile(".exr");
#ifndef GENERATE_DATA
    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
#else
    const Size sz(64, 32);
    Mat img(sz, CV_32FC1, Scalar(0.5, 0.1, 1));
    img(Rect(10, 5, sz.width - 30, sz.height - 20)).setTo(Scalar(1, 0, 0));
    ASSERT_TRUE(cv::imwrite(filenameInput, img));
#endif
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC1,img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_EXR, readWrite_32FC3)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC3.exr";
    const string filenameOutput = cv::tempfile(".exr");
#ifndef GENERATE_DATA
    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
#else
    const Size sz(64, 32);
    Mat img(sz, CV_32FC3, Scalar(0.5, 0.1, 1));
    img(Rect(10, 5, sz.width - 30, sz.height - 20)).setTo(Scalar(1, 0, 0));
    ASSERT_TRUE(cv::imwrite(filenameInput, img));
#endif
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC3, img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}


TEST(Imgcodecs_EXR, readWrite_32FC1_half)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC1_half.exr";
    const string filenameOutput = cv::tempfile(".exr");

    std::vector<int> params;
    params.push_back(IMWRITE_EXR_TYPE);
    params.push_back(IMWRITE_EXR_TYPE_HALF);

#ifndef GENERATE_DATA
    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
#else
    const Size sz(64, 32);
    Mat img(sz, CV_32FC1, Scalar(0.5, 0.1, 1));
    img(Rect(10, 5, sz.width - 30, sz.height - 20)).setTo(Scalar(1, 0, 0));
    ASSERT_TRUE(cv::imwrite(filenameInput, img, params));
#endif
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC1,img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img, params));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}

TEST(Imgcodecs_EXR, readWrite_32FC3_half)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filenameInput = root + "readwrite/test32FC3_half.exr";
    const string filenameOutput = cv::tempfile(".exr");

    std::vector<int> params;
    params.push_back(IMWRITE_EXR_TYPE);
    params.push_back(IMWRITE_EXR_TYPE_HALF);

#ifndef GENERATE_DATA
    const Mat img = cv::imread(filenameInput, IMREAD_UNCHANGED);
#else
    const Size sz(64, 32);
    Mat img(sz, CV_32FC3, Scalar(0.5, 0.1, 1));
    img(Rect(10, 5, sz.width - 30, sz.height - 20)).setTo(Scalar(1, 0, 0));
    ASSERT_TRUE(cv::imwrite(filenameInput, img, params));
#endif
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(CV_32FC3, img.type());

    ASSERT_TRUE(cv::imwrite(filenameOutput, img, params));
    const Mat img2 = cv::imread(filenameOutput, IMREAD_UNCHANGED);
    ASSERT_EQ(img2.type(), img.type());
    ASSERT_EQ(img2.size(), img.size());
    EXPECT_LE(cvtest::norm(img, img2, NORM_INF | NORM_RELATIVE), 1e-3);
    EXPECT_EQ(0, remove(filenameOutput.c_str()));
}


}} // namespace
