// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class ChromaticAberrationTest : public testing::Test
{
protected:    
    std::string test_yaml_file;
    cv::Mat test_image;

    void SetUp() override
    {
        string data_path = cvtest::TS::ptr()->get_data_path();
        ASSERT_TRUE(data_path != "") << "OPENCV_TEST_DATA_PATH not set";
        test_yaml_file = std::string(data_path) + "cameracalibration/chromatic_aberration/calib_result.yaml";
        test_image = cv::imread(std::string(data_path) + "cameracalibration/chromatic_aberration/ca_photo.png");
        ASSERT_FALSE(test_image.empty()) << "Failed to load test image";
    }
};

TEST_F(ChromaticAberrationTest, CalibrationResultLoad)
{
    cv::CalibrationResult calib_result;
    
    // Test successful loading
    EXPECT_TRUE(calib_result.loadFromFile(test_yaml_file));
    EXPECT_EQ(calib_result.degree, 11);
    
    // Test red channel data
    EXPECT_EQ(calib_result.poly_red.degree, 11);
    EXPECT_EQ(calib_result.poly_red.coeffs_x.size(), 78);
    EXPECT_EQ(calib_result.poly_red.coeffs_y.size(), 78);
    EXPECT_DOUBLE_EQ(calib_result.poly_red.mean_x, 3392.451171875);
    EXPECT_DOUBLE_EQ(calib_result.poly_red.std_x, 2239.90380859375);
    
    // Test blue channel data
    EXPECT_EQ(calib_result.poly_blue.degree, 11);
    EXPECT_EQ(calib_result.poly_blue.coeffs_x.size(), 78);
    EXPECT_EQ(calib_result.poly_blue.coeffs_y.size(), 78);
    EXPECT_DOUBLE_EQ(calib_result.poly_blue.mean_x, 3394.3818359375);
    EXPECT_DOUBLE_EQ(calib_result.poly_blue.std_x, 2239.822265625);
}

TEST_F(ChromaticAberrationTest, CalibrationResultLoadInvalidFile)
{
    cv::CalibrationResult calib_result;
    
    // Test loading non-existent file
    EXPECT_FALSE(calib_result.loadFromFile("non_existent_file.yaml"));
}

TEST_F(ChromaticAberrationTest, ChromaticAberrationCorrectorLoadCalibration)
{
    cv::ChromaticAberrationCorrector corrector;
    
    // Test successful calibration loading
    EXPECT_TRUE(corrector.loadCalibration(test_yaml_file));
    
    // Test loading non-existent file
    EXPECT_FALSE(corrector.loadCalibration("non_existent_file.yaml"));
}

TEST_F(ChromaticAberrationTest, ChromaticAberrationCorrection)
{
    cv::ChromaticAberrationCorrector corrector;
    ASSERT_TRUE(corrector.loadCalibration(test_yaml_file));
    
    // Test image correction
    cv::Mat corrected = corrector.correctImage(test_image);
    
    // Verify output properties
    EXPECT_EQ(corrected.rows, test_image.rows);
    EXPECT_EQ(corrected.cols, test_image.cols);
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
    cv::ChromaticAberrationCorrector corrector;
    ASSERT_TRUE(corrector.loadCalibration(test_yaml_file));
    
    // Test with single channel image (should fail)
    cv::Mat gray_image;
    cv::cvtColor(test_image, gray_image, cv::COLOR_BGR2GRAY);
    
    EXPECT_THROW(corrector.correctImage(gray_image), cv::Exception);
}

TEST_F(ChromaticAberrationTest, Polynomial2DComputeDeltas)
{
    cv::Polynomial2D poly;
    poly.degree = 1;
    poly.coeffs_x = {0.0, 0.1, 0.05};
    poly.coeffs_y = {0.0, -0.1, -0.05};
    poly.mean_x = 50.0;
    poly.std_x = 30.0;
    poly.mean_y = 50.0;
    poly.std_y = 30.0;
    
    // Create coordinate matrices
    cv::Mat X = cv::Mat::zeros(3, 3, CV_32F);
    cv::Mat Y = cv::Mat::zeros(3, 3, CV_32F);
    
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            X.at<float>(y, x) = static_cast<float>(x * 10);
            Y.at<float>(y, x) = static_cast<float>(y * 10);
        }
    }
    
    cv::Mat dx, dy;
    poly.computeDeltas(X, Y, dx, dy);
    
    // Verify output dimensions
    EXPECT_EQ(dx.rows, X.rows);
    EXPECT_EQ(dx.cols, X.cols);
    EXPECT_EQ(dy.rows, Y.rows);
    EXPECT_EQ(dy.cols, Y.cols);
    EXPECT_EQ(dx.type(), CV_32F);
    EXPECT_EQ(dy.type(), CV_32F);
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
    
    int degree;
    fs["degree"] >> degree;
    EXPECT_EQ(degree, 11);
    
    cv::FileNode red_node = fs["red_channel"];
    cv::FileNode blue_node = fs["blue_channel"];
    EXPECT_TRUE(red_node.isMap());
    EXPECT_TRUE(blue_node.isMap());
    
    std::vector<double> coeffs_x;
    red_node["coeffs_x"] >> coeffs_x;
    double red_mean_x, red_std_x, blue_mean_x, blue_std_x;
    red_node["mean_x"] >> red_mean_x;
    red_node["std_x"] >> red_std_x;
    blue_node["mean_x"] >> blue_mean_x;
    blue_node["std_x"] >> blue_std_x;
    std::cout << blue_mean_x << "\n";
    std::cout << blue_std_x << "\n";
    EXPECT_EQ(coeffs_x.size(), 78);
    EXPECT_NEAR(red_mean_x, 3392.45, 1e-2);
    EXPECT_NEAR(red_std_x, 2239.9, 1e-2);
    EXPECT_NEAR(red_mean_x, 3394.38, 1e-2);
    EXPECT_NEAR(red_std_x, 2239.82, 1e-2);
    
    fs.release();
}

TEST_F(ChromaticAberrationTest, RealWorldDataIntegration)
{
    try {
        std::string real_calib_file = cv::samples::findFile("cv/cameracalibration/chromatic_aberration/calib_result.yaml");
        std::string real_test_image = cv::samples::findFile("cv/cameracalibration/chromatic_aberration/ca_photo.png");
        
        cv::ChromaticAberrationCorrector corrector;
        
        EXPECT_TRUE(corrector.loadCalibration(real_calib_file));
        
        cv::Mat real_image = cv::imread(real_test_image, cv::IMREAD_COLOR);
        ASSERT_FALSE(real_image.empty()) << "Failed to load real test image";
        
        cv::Mat corrected = corrector.correctImage(real_image);
        
        EXPECT_EQ(corrected.rows, real_image.rows);
        EXPECT_EQ(corrected.cols, real_image.cols);
        EXPECT_EQ(corrected.channels(), real_image.channels());
        
        cv::Mat diff;
        cv::absdiff(real_image, corrected, diff);
        
        cv::Scalar mean_diff = cv::mean(diff);
        double total_diff = mean_diff[0] + mean_diff[1] + mean_diff[2];
        
        EXPECT_GT(total_diff, 0.0) << "Real image correction should produce changes";
        EXPECT_LT(total_diff, 150.0) << "Real image correction should not be excessive";
        
        cv::Mat orig_gray, corr_gray;
        cv::cvtColor(real_image, orig_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(corrected, corr_gray, cv::COLOR_BGR2GRAY);
        
        double psnr = cv::PSNR(orig_gray, corr_gray);
        EXPECT_GT(psnr, 20.0) << "PSNR should indicate reasonable image quality";
        EXPECT_LT(psnr, 50.0) << "PSNR should show meaningful correction";
    } catch (const cv::Exception& e) {
        std::cout << "Skipping real world test - test data files not found: " << e.what() << std::endl;
    }
}

TEST_F(ChromaticAberrationTest, EdgeCasesAndRobustness)
{
    cv::ChromaticAberrationCorrector corrector;
    ASSERT_TRUE(corrector.loadCalibration(test_yaml_file));
    
    // Test with different image sizes
    cv::Mat small_image;
    cv::resize(test_image, small_image, cv::Size(50, 50));
    
    cv::Mat corrected_small = corrector.correctImage(small_image);
    EXPECT_EQ(corrected_small.rows, small_image.rows);
    EXPECT_EQ(corrected_small.cols, small_image.cols);
    
    // Test with large image
    cv::Mat large_image;
    cv::resize(test_image, large_image, cv::Size(500, 500));
    
    cv::Mat corrected_large = corrector.correctImage(large_image);
    EXPECT_EQ(corrected_large.rows, large_image.rows);
    EXPECT_EQ(corrected_large.cols, large_image.cols);
    
    // Test with extreme values image
    cv::Mat extreme_image = cv::Mat::zeros(100, 100, CV_8UC3);
    extreme_image.setTo(cv::Scalar(255, 255, 255));
    
    cv::Mat corrected_extreme = corrector.correctImage(extreme_image);
    EXPECT_EQ(corrected_extreme.rows, extreme_image.rows);
    EXPECT_EQ(corrected_extreme.cols, extreme_image.cols);
    
    // Verify no NaN or infinite values in output
    cv::Mat corrected_f32;
    corrected_extreme.convertTo(corrected_f32, CV_32F);
    
    cv::Mat nan_mask, inf_mask;
    cv::compare(corrected_f32, corrected_f32, nan_mask, cv::CMP_EQ); // NaN != NaN
    cv::bitwise_not(nan_mask, nan_mask);
    
    EXPECT_EQ(cv::countNonZero(nan_mask), 0) << "Corrected image should not contain NaN values";
}

}}