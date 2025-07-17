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

    void SetUp() override
    {
        string data_path = cvtest::TS::ptr()->get_data_path();
        ASSERT_TRUE(!data_path.empty()) << "OPENCV_TEST_DATA_PATH not set";
        test_yaml_file = std::string(data_path) + "cameracalibration/chromatic_aberration/calib_result_tablet.yaml";
        test_image = cv::imread(std::string(data_path) + "cameracalibration/chromatic_aberration/tablet_circles_.png");
        ASSERT_FALSE(test_image.empty()) << "Failed to load test image";
    }
};

TEST_F(ChromaticAberrationTest, ChromaticAberrationCorrectorLoadCalibration)
{
    EXPECT_NO_THROW({
        cv::ChromaticAberrationCorrector corrector(test_yaml_file);
        (void)corrector;
    });

    EXPECT_THROW({
        cv::ChromaticAberrationCorrector bad_corrector("non_existent_file.yaml");
        (void)bad_corrector;
    }, cv::Exception);

}

TEST_F(ChromaticAberrationTest, ChromaticAberrationCorrection)
{
    cv::ChromaticAberrationCorrector corrector(test_yaml_file);
    
    // Test image correction
    cv::Mat corrected = corrector.correctImage(test_image);
    
    EXPECT_EQ(corrected.size(), test_image.size());
    EXPECT_EQ(corrected.channels(), test_image.channels());
    EXPECT_EQ(corrected.type(), test_image.type());
    
    // Verify the image was actually processed (should be different)
    cv::Mat diff;
    cv::absdiff(test_image, corrected, diff);
    cv::Scalar sum_diff = cv::sum(diff);
    EXPECT_GT(sum_diff[0] + sum_diff[1] + sum_diff[2], 0.0);
}

TEST_F(ChromaticAberrationTest, ChromaticAberrationCorrectionTablet)
{
    string data_path = cvtest::TS::ptr()->get_data_path();
    ASSERT_TRUE(data_path != "") << "OPENCV_TEST_DATA_PATH not set";
    test_yaml_file = std::string(data_path) + "cameracalibration/chromatic_aberration/calib_result_tablet.yaml";
    test_image = cv::imread(std::string(data_path) + "cameracalibration/chromatic_aberration/tablet_circles_.png");
    ASSERT_FALSE(test_image.empty()) << "Failed to load test image";

    cv::ChromaticAberrationCorrector corrector(test_yaml_file);    
    cv::Mat corrected = corrector.correctImage(test_image);
    
    // Verify output properties
    EXPECT_EQ(corrected.size(), test_image.size());
    EXPECT_EQ(corrected.channels(), test_image.channels());
    EXPECT_EQ(corrected.type(), test_image.type());
    
    // Verify the image was actually processed (should be different)
    cv::Mat diff;
    cv::absdiff(test_image, corrected, diff);
    cv::Scalar sum_diff = cv::sum(diff);
    EXPECT_GT(sum_diff[0] + sum_diff[1] + sum_diff[2], 0.0);
}
TEST_F(ChromaticAberrationTest, ChromaticAberrationCorrectionSyntheticSimpleWarp)
{
    string data_path = cvtest::TS::ptr()->get_data_path();
    ASSERT_TRUE(data_path != "") << "OPENCV_TEST_DATA_PATH not set";
    test_yaml_file = std::string(data_path) + "cameracalibration/chromatic_aberration/simple_warp.yaml";
    test_image = cv::imread(std::string(data_path) + "cameracalibration/chromatic_aberration/synthetic_simple_warp.png");
    ASSERT_FALSE(test_image.empty()) << "Failed to load test image";

    cv::ChromaticAberrationCorrector corrector(test_yaml_file);    
    cv::Mat corrected = corrector.correctImage(test_image);
    
    // Verify output properties
    EXPECT_EQ(corrected.size(), test_image.size());
    EXPECT_EQ(corrected.channels(), test_image.channels());
    EXPECT_EQ(corrected.type(), test_image.type());
    
    // Verify the image was actually processed (should be different)
    cv::Mat diff;
    cv::absdiff(test_image, corrected, diff);
    cv::Scalar sum_diff = cv::sum(diff);
    EXPECT_GT(sum_diff[0] + sum_diff[1] + sum_diff[2], 0.0);
}
TEST_F(ChromaticAberrationTest, ChromaticAberrationCorrectionSyntheticRadial)
{
    string data_path = cvtest::TS::ptr()->get_data_path();
    ASSERT_TRUE(data_path != "") << "OPENCV_TEST_DATA_PATH not set";
    test_yaml_file = std::string(data_path) + "cameracalibration/chromatic_aberration/radial.yaml";
    test_image = cv::imread(std::string(data_path) + "cameracalibration/chromatic_aberration/synthetic_radial.png");
    ASSERT_FALSE(test_image.empty()) << "Failed to load test image";

    cv::ChromaticAberrationCorrector corrector(test_yaml_file);    
    cv::Mat corrected = corrector.correctImage(test_image);
    
    // Verify output properties
    EXPECT_EQ(corrected.size(), test_image.size());
    EXPECT_EQ(corrected.channels(), test_image.channels());
    EXPECT_EQ(corrected.type(), test_image.type());
    
    // Verify the image was actually processed (should be different)
    cv::Mat diff;
    cv::absdiff(test_image, corrected, diff);
    cv::Scalar sum_diff = cv::sum(diff);
    EXPECT_GT(sum_diff[0] + sum_diff[1] + sum_diff[2], 0.0);
}

TEST_F(ChromaticAberrationTest, ChromaticAberrationCorrectionInvalidInput)
{
    cv::ChromaticAberrationCorrector corrector(test_yaml_file);
    
    // Test with single channel image (should fail)
    cv::Mat gray_image;
    cv::cvtColor(test_image, gray_image, cv::COLOR_BGR2GRAY);
    
    EXPECT_THROW(corrector.correctImage(gray_image), cv::Exception);
}

TEST_F(ChromaticAberrationTest, CorrectChromaticAberrationFunction)
{
    // Test the standalone function
    cv::Mat corrected = cv::correctChromaticAberration(test_image, test_yaml_file);
    
    // Verify output properties
    EXPECT_EQ(corrected.rows, test_image.rows);
    EXPECT_EQ(corrected.cols, test_image.cols);
    EXPECT_EQ(corrected.channels(), test_image.channels());
    EXPECT_EQ(corrected.type(), test_image.type());
}

TEST_F(ChromaticAberrationTest, YAMLReadingIntegration)
{
    // Test that FileStorage correctly reads YAML format
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

TEST_F(ChromaticAberrationTest, FunctionClassEquivalence)
{
    cv::ChromaticAberrationCorrector corrector(test_yaml_file);
    cv::Mat ref  = corrector.correctImage(test_image);
    cv::Mat out  = cv::correctChromaticAberration(test_image, test_yaml_file);  // wrapper

    ASSERT_EQ(ref.size(),  test_image.size());
    ASSERT_EQ(ref.type(),  test_image.type());

    cv::Mat diff;  cv::absdiff(ref, out, diff);

    EXPECT_LE(cv::norm(diff, cv::NORM_INF), 1);

    int nz = cv::countNonZero(diff.reshape(1));
    EXPECT_EQ(nz, 0) << nz << " pixels differ between implementations";

}

}}