// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

const unsigned long EXPECTED_COEFFS_SIZE = 78;

class ChromaticAberrationTest : public testing::Test
{
protected:
    std::string test_yaml_file;
    cv::Mat test_image;
    cv::Mat coeffMat;
    cv::Mat corrected;
    int degree = -1;
    Size calib_size = {-1, -1};

    void SetUp() override
    {
        string data_path = cvtest::TS::ptr()->get_data_path();
        ASSERT_TRUE(!data_path.empty()) << "OPENCV_TEST_DATA_PATH not set";
        test_yaml_file = std::string(data_path) + "cameracalibration/chromatic_aberration/ca_photo_calib.yaml";
        test_image = cv::imread(std::string(data_path) + "cameracalibration/chromatic_aberration/ca_photo.png");
        ASSERT_FALSE(test_image.empty()) << "Failed to load test image";
    }
};

TEST_F(ChromaticAberrationTest, LoadCalibAndCorrectImage)
{
    FileStorage fs(test_yaml_file, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    ASSERT_NO_THROW(cv::loadChromaticAberrationParams(fs.root(), coeffMat, calib_size, degree));

    ASSERT_FALSE(coeffMat.empty());
    ASSERT_EQ(coeffMat.type(), CV_32F);
    ASSERT_EQ(coeffMat.rows, 4);
    ASSERT_GT(coeffMat.cols, 0);
    ASSERT_EQ((degree + 1) * (degree + 2) / 2, coeffMat.cols);
    ASSERT_GT(calib_size.width, 0);
    ASSERT_GT(calib_size.height, 0);

    ASSERT_EQ(test_image.cols, calib_size.width);
    ASSERT_EQ(test_image.rows, calib_size.height);

    ASSERT_NO_THROW(cv::correctChromaticAberration(test_image, coeffMat, corrected, calib_size, degree));

    EXPECT_EQ(corrected.size(), test_image.size());
    EXPECT_EQ(corrected.channels(), test_image.channels());
    EXPECT_EQ(corrected.type(), test_image.type());

    cv::Mat diff; cv::absdiff(test_image, corrected, diff);
    cv::Scalar s = cv::sum(diff);
    EXPECT_GT(s[0] + s[1] + s[2], 0.0);
}

TEST_F(ChromaticAberrationTest, YAMLContentsAsExpected)
{
    cv::FileStorage fs(test_yaml_file, cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    cv::FileNode red_node = fs["red_channel"];
    cv::FileNode blue_node = fs["blue_channel"];
    EXPECT_TRUE(red_node.isMap());
    EXPECT_TRUE(blue_node.isMap());

    std::vector<double> coeffs_x;
    red_node["coeffs_x"] >> coeffs_x;
    EXPECT_EQ(coeffs_x.size(), EXPECTED_COEFFS_SIZE);
    blue_node["coeffs_x"] >> coeffs_x;
    EXPECT_EQ(coeffs_x.size(), EXPECTED_COEFFS_SIZE);

    std::vector<double> coeffs_y;
    red_node["coeffs_y"] >> coeffs_y;
    EXPECT_EQ(coeffs_y.size(), EXPECTED_COEFFS_SIZE);
    blue_node["coeffs_y"] >> coeffs_y;
    EXPECT_EQ(coeffs_y.size(), EXPECTED_COEFFS_SIZE);

    fs.release();
}

TEST_F(ChromaticAberrationTest, InvalidSingleChannel)
{
    FileStorage fs(test_yaml_file, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    ASSERT_NO_THROW(cv::loadChromaticAberrationParams(fs.root(), coeffMat, calib_size, degree));

    cv::Mat gray;
    cv::cvtColor(test_image, gray, cv::COLOR_BGR2GRAY);

    EXPECT_THROW(cv::correctChromaticAberration(gray, coeffMat, corrected, calib_size, degree),
                 cv::Exception);
}

TEST_F(ChromaticAberrationTest, EmptyCoeffMat)
{
    FileStorage fs(test_yaml_file, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    ASSERT_NO_THROW(cv::loadChromaticAberrationParams(fs.root(), coeffMat, calib_size, degree));

    cv::Mat emptyCoeff;
    EXPECT_THROW(cv::correctChromaticAberration(test_image, emptyCoeff, corrected, calib_size, degree),
                 cv::Exception);
}

TEST_F(ChromaticAberrationTest, MismatchedImageSize)
{
    FileStorage fs(test_yaml_file, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    ASSERT_NO_THROW(cv::loadChromaticAberrationParams(fs.root(), coeffMat, calib_size, degree));

    cv::Mat resized;
    cv::resize(test_image, resized, cv::Size(test_image.cols/2, test_image.rows/2));
    EXPECT_THROW(cv::correctChromaticAberration(resized, coeffMat, corrected, calib_size, degree),
                 cv::Exception);
}

TEST_F(ChromaticAberrationTest, WrongCoeffType)
{
    FileStorage fs(test_yaml_file, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    ASSERT_NO_THROW(cv::loadChromaticAberrationParams(fs.root(), coeffMat, calib_size, degree));

    cv::Mat wrongType;
    coeffMat.convertTo(wrongType, CV_64F);
    EXPECT_THROW(cv::correctChromaticAberration(test_image, wrongType, corrected, calib_size, degree),
                 cv::Exception);
}

TEST_F(ChromaticAberrationTest, DegreeDoesNotMatchCoeffCols)
{
    FileStorage fs(test_yaml_file, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    ASSERT_NO_THROW(cv::loadChromaticAberrationParams(fs.root(), coeffMat, calib_size, degree));

    int wrongDegree = std::max(1, degree - 1);
    ASSERT_NE((wrongDegree + 1) * (wrongDegree + 2) / 2, coeffMat.cols);
    EXPECT_THROW(cv::correctChromaticAberration(test_image, coeffMat, corrected, calib_size, wrongDegree),
                 cv::Exception);
}

}}
