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
    ASSERT_NO_THROW(img = imread(filename));
    ASSERT_FALSE(img.empty());

    Mat target_img;
    resize(img, target_img, Size(3, 3), 0, 0, INTER_NEAREST_PIL);

    Mat expected_img = (Mat_<CV_8U>(3,3) << 40, 125, 209, 43, 127, 211, 45, 129, 213);
    EXPECT_EQ(target_img, expected_img);
}

}} // namespace
