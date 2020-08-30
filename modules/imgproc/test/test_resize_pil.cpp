// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//

#include "test_precomp.hpp"

namespace opencv_test { namespace {

uchar synthethic_gradient[] = {10, 20,  30,  40,  50,  60,  70,  80,  90,
                               20, 30,  40,  50,  60,  70,  80,  90,  100,
                               30, 40,  50,  60,  70,  80,  90,  100, 110,
                               40, 50,  60,  70,  80,  90,  100, 110, 120,
                               50, 60,  70,  80,  90,  100, 110, 120, 130,
                               60, 70,  80,  90,  100, 110, 120, 130, 140,
                               70, 80,  90,  100, 110, 120, 130, 140, 150,
                               80, 90,  100, 110, 120, 130, 140, 150, 160,
                               90, 100, 110, 120, 130, 140, 150, 160, 170};

TEST(Imgproc_Resize, accuracy_INTER_NEAREST_PIL_QUADRATIC)
{
    Mat img = Mat(9, 9, CV_8UC1, synthethic_gradient);

    Mat target_img;
    resize(img, target_img, Size(3, 3), 0, 0, INTER_NEAREST_PIL);

    uchar data[] = {30, 60, 90, 60, 90, 120, 90, 120, 150};
    Mat expected_img = Mat(3, 3, CV_8UC1, data);
    ASSERT_EQ( cvtest::norm(target_img, expected_img, NORM_INF), 0.);
}

TEST(Imgproc_Resize, accuracy_INTER_NEAREST_PIL_NONQUADRATIC)
{
    Mat img = Mat(9, 9, CV_8UC1, synthethic_gradient);

    Mat target_img;
    resize(img, target_img, Size(5, 7), 0, 0, INTER_NEAREST_PIL);

    uchar data[] = {10, 30, 50, 70, 90,
                    20, 40, 60, 80, 100,
                    40, 60, 80, 100, 120,
                    50, 70, 90, 110, 130,
                    60, 80, 100,120, 140,
                    80, 100,120,140, 160,
                    90, 110,130,150, 170};
    Mat expected_img = Mat(7, 5, CV_8UC1, data);
    ASSERT_EQ( cvtest::norm(target_img, expected_img, NORM_INF), 0.);
}

TEST(Imgproc_Resize, accuracy_INTER_NEAREST_PIL_DOWN_UP_7_5)
{
    Mat img = Mat(9, 9, CV_8UC1, synthethic_gradient);

    Mat target_img;
    resize(img, target_img, Size(3, 3), 0, 0, INTER_NEAREST_PIL);
    resize(target_img, target_img, Size(7, 5), 0, 0, INTER_NEAREST_PIL);

    uchar data[] = {30, 30, 60, 60, 60, 90, 90,
                    30, 30, 60, 60, 60, 90, 90,
                    60, 60, 90, 90, 90, 120, 120,
                    90, 90, 120,120,120,150, 150,
                    90, 90, 120,120,120,150, 150};
    Mat expected_img = Mat(5, 7, CV_8UC1, data);
    ASSERT_EQ( cvtest::norm(target_img, expected_img, NORM_INF), 0.);
}

TEST(Imgproc_Resize, accuracy_INTER_NEAREST_PIL_DOWN_UP_4_2)
{
    Mat img = Mat(9, 9, CV_8UC1, synthethic_gradient);

    Mat target_img;
    resize(img, target_img, Size(3, 3), 0, 0, INTER_NEAREST_PIL);
    resize(target_img, target_img, Size(4, 2), 0, 0, INTER_NEAREST_PIL);

    uchar data[] = {30, 60, 60, 90,
                    90, 120, 120, 150};
    Mat expected_img = Mat(2, 4, CV_8UC1, data);
    ASSERT_EQ( cvtest::norm(target_img, expected_img, NORM_INF), 0.);
}

}} // namespace
