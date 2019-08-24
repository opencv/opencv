// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include <opencv2/ts/ts.hpp>


namespace opencv_test { namespace {

#define SAVE(x) imwrite(folder + "output.png", x);

static const double numerical_precision = 0.05; // 95% of pixels should have exact values

TEST(Alphamat_infoFlow, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "alphamat/";
    string image_path = folder + "img/elephant.png";
    string trimap_path = folder + "trimap/elephant.png";
    string reference_path = folder + "reference/elephant.png";

    Mat image = imread(original_path, IMREAD_COLOR);
    Mat trimap = imread(original_path, IMREAD_COLOR);
    Mat reference = imread(expected_path, IMREAD_GRAYSCALE);

    ASSERT_FALSE(image.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(trimap.empty()) << "Could not load input trimap " << trimap_path;
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    ASSERT_EQ(image.rows, trimap.rows) << "Height of image and trimap dont match";
    ASSERT_EQ(image.cols, trimap.cols) << "Height of image and trimap dont match";

    Mat result;
    infoFlow(image, trimap, result, true, true);

    SAVE(result);

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

TEST(Alphamat_infoFlow, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "alphamat/";
    string image_path = folder + "img/elephant.png";
    string trimap_path = folder + "trimap/elephant.png";
    string reference_path = folder + "reference/elephant.png";

    Mat image = imread(original_path, IMREAD_COLOR);
    Mat trimap = imread(original_path, IMREAD_COLOR);
    Mat reference = imread(expected_path, IMREAD_GRAYSCALE);

    ASSERT_FALSE(image.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(trimap.empty()) << "Could not load input trimap " << trimap_path;
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    ASSERT_EQ(image.rows, trimap.rows) << "Height of image and trimap dont match";
    ASSERT_EQ(image.cols, trimap.cols) << "Height of image and trimap dont match";

    Mat result;
    infoFlow(original, result, true, false);

    SAVE(result);

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}


TEST(Alphamat_infoFlow, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "alphamat/";
    string image_path = folder + "img/elephant.png";
    string trimap_path = folder + "trimap/elephant.png";
    string reference_path = folder + "reference/elephant.png";

    Mat image = imread(original_path, IMREAD_COLOR);
    Mat trimap = imread(original_path, IMREAD_COLOR);
    Mat reference = imread(expected_path, IMREAD_GRAYSCALE);

    ASSERT_FALSE(image.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(trimap.empty()) << "Could not load input trimap " << trimap_path;
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    ASSERT_EQ(image.rows, trimap.rows) << "Height of image and trimap dont match";
    ASSERT_EQ(image.cols, trimap.cols) << "Height of image and trimap dont match";

    Mat result;
    infoFlow(original, result, false, true);

    SAVE(result);

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}


TEST(Alphamat_infoFlow, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "alphamat/";
    string image_path = folder + "img/elephant.png";
    string trimap_path = folder + "trimap/elephant.png";
    string reference_path = folder + "reference/elephant.png";

    Mat image = imread(original_path, IMREAD_COLOR);
    Mat trimap = imread(original_path, IMREAD_COLOR);
    Mat reference = imread(expected_path, IMREAD_GRAYSCALE);

    ASSERT_FALSE(image.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(trimap.empty()) << "Could not load input trimap " << trimap_path;
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    ASSERT_EQ(image.rows, trimap.rows) << "Height of image and trimap dont match";
    ASSERT_EQ(image.cols, trimap.cols) << "Height of image and trimap dont match";

    Mat result;
    infoFlow(original, result, false, false);

    SAVE(result);

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
	

}} //namespace