// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Imgproc_Resize, accuracy_INTER_NEAREST_PIL_QUADRATIC)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "gradient.png";

    Mat img;
    ASSERT_NO_THROW(img = imread(filename, IMREAD_GRAYSCALE));
    ASSERT_FALSE(img.empty());

    Mat target_img;
    resize(img, target_img, Size(3, 3), 0, 0, INTER_NEAREST_PIL);

    uchar data[] = {41, 125, 209, 43, 127, 211, 45, 129, 213};
    Mat expected_img = Mat(3, 3, CV_8UC1, data);
    ASSERT_EQ( cvtest::norm(target_img, expected_img, NORM_INF), 0.);
}

TEST(Imgproc_Resize, accuracy_INTER_NEAREST_PIL_NONQUADRATIC)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "gradient.png";

    Mat img;
    ASSERT_NO_THROW(img = imread(filename, IMREAD_GRAYSCALE));
    ASSERT_FALSE(img.empty());

    Mat target_img;
    resize(img, target_img, Size(7, 11), 0, 0, INTER_NEAREST_PIL);

    uchar data[] = {16,  52,  88, 125, 160, 196, 232,
                    17,  52,  89, 125, 160, 196, 233,
                    17,  53,  89, 126, 161, 197, 233,
                    17,  54,  89, 126, 162, 197, 234,
                    18,  54,  90, 126, 162, 198, 234,
                    18,  54,  91, 127, 162, 198, 235,
                    19,  55,  91, 127, 162, 199, 235,
                    19,  55,  91, 127, 163, 199, 235,
                    20,  56,  92, 128, 164, 200, 236,
                    20,  56,  92, 129, 164, 200, 236,
                    20,  56,  93, 129, 165, 201, 237};
    Mat expected_img = Mat(11, 7, CV_8UC1, data);
    ASSERT_EQ( cvtest::norm(target_img, expected_img, NORM_INF), 0.);
}

TEST(Imgproc_Resize, accuracy_INTER_NEAREST_PIL_DOWN_UP_7_11)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "gradient.png";

    Mat img;
    ASSERT_NO_THROW(img = imread(filename, IMREAD_GRAYSCALE));
    ASSERT_FALSE(img.empty());

    Mat target_img;
    resize(img, target_img, Size(3, 3), 0, 0, INTER_NEAREST_PIL);
    resize(target_img, target_img, Size(7, 11), 0, 0, INTER_NEAREST_PIL);

    uchar data[] = {41,  41, 125, 125, 125, 209, 209,
                  41,  41, 125, 125, 125, 209, 209,
                  41,  41, 125, 125, 125, 209, 209,
                  41,  41, 125, 125, 125, 209, 209,
                  43,  43, 127, 127, 127, 211, 211,
                  43,  43, 127, 127, 127, 211, 211,
                  43,  43, 127, 127, 127, 211, 211,
                  45,  45, 129, 129, 129, 213, 213,
                  45,  45, 129, 129, 129, 213, 213,
                  45,  45, 129, 129, 129, 213, 213,
                  45,  45, 129, 129, 129, 213, 213};
    Mat expected_img = Mat(11, 7, CV_8UC1, data);
    ASSERT_EQ( cvtest::norm(target_img, expected_img, NORM_INF), 0.);
}

TEST(Imgproc_Resize, accuracy_INTER_NEAREST_PIL_DOWN_UP_11_7)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + "gradient.png";

    Mat img;
    ASSERT_NO_THROW(img = imread(filename, IMREAD_GRAYSCALE));
    ASSERT_FALSE(img.empty());

    Mat target_img;
    resize(img, target_img, Size(3, 3), 0, 0, INTER_NEAREST_PIL);
    resize(target_img, target_img, Size(11, 7), 0, 0, INTER_NEAREST_PIL);

    uchar data[] = {41,  41,  41,  41, 125, 125, 125, 209, 209, 209, 209,
                  41,  41,  41,  41, 125, 125, 125, 209, 209, 209, 209,
                  43,  43,  43,  43, 127, 127, 127, 211, 211, 211, 211,
                  43,  43,  43,  43, 127, 127, 127, 211, 211, 211, 211,
                  43,  43,  43,  43, 127, 127, 127, 211, 211, 211, 211,
                  45,  45,  45,  45, 129, 129, 129, 213, 213, 213, 213,
                  45,  45,  45,  45, 129, 129, 129, 213, 213, 213, 213};
    Mat expected_img = Mat(7, 11, CV_8UC1, data);
    ASSERT_EQ( cvtest::norm(target_img, expected_img, NORM_INF), 0.);
}

}} // namespace
