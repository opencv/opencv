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
        ASSERT_TRUE(data_path != "") << "OPENCV_TEST_DATA_PATH not set";
        test_yaml_file = std::string(data_path) + "cameracalibration/chromatic_aberration/calib_result_tablet.yaml";
        test_image = cv::imread(std::string(data_path) + "cameracalibration/chromatic_aberration/tablet_circles_.png");
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
    EXPECT_EQ(calib_result.poly_red.coeffs_x.size(), EXPECTED_COEFFS_SIZE);
    EXPECT_EQ(calib_result.poly_red.coeffs_y.size(), EXPECTED_COEFFS_SIZE);
    
    // Test blue channel data
    EXPECT_EQ(calib_result.poly_blue.degree, 11);
    EXPECT_EQ(calib_result.poly_blue.coeffs_x.size(), EXPECTED_COEFFS_SIZE);
    EXPECT_EQ(calib_result.poly_blue.coeffs_y.size(), EXPECTED_COEFFS_SIZE);
}

TEST_F(ChromaticAberrationTest, CalibrationResultLoadInvalidFile)
{
    cv::CalibrationResult calib_result;
    
    // Test loading non-existent file
    EXPECT_THROW(
        calib_result.loadFromFile("non_existent_file.yaml"),
        cv::Exception
    );
}

TEST_F(ChromaticAberrationTest, ChromaticAberrationCorrectorLoadCalibration)
{
    cv::ChromaticAberrationCorrector corrector;
    
    // Test successful calibration loading
    EXPECT_TRUE(corrector.loadCalibration(test_yaml_file));

    EXPECT_THROW(
        corrector.loadCalibration("non_existent_file.yaml"),
        cv::Exception
    );

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
    
    
    cv::FileNode red_node = fs["red_channel"];
    cv::FileNode blue_node = fs["blue_channel"];
    EXPECT_TRUE(red_node.isMap());
    EXPECT_TRUE(blue_node.isMap());
    
    std::vector<double> coeffs_x;
    red_node["coeffs_x"] >> coeffs_x;
    EXPECT_EQ(coeffs_x.size(), EXPECTED_COEFFS_SIZE);
    
    fs.release();
}

TEST_F(ChromaticAberrationTest, FunctionClassEquivalence)
{
    cv::ChromaticAberrationCorrector corrector;
    ASSERT_TRUE(corrector.loadCalibration(test_yaml_file));
    cv::Mat ref  = corrector.correctImage(test_image);
    cv::Mat out  = cv::correctChromaticAberration(test_image, test_yaml_file);  // wrapper

    ASSERT_EQ(ref.size(),  test_image.size());
    ASSERT_EQ(ref.type(),  test_image.type());

    cv::Mat diff;  cv::absdiff(ref, out, diff);

    EXPECT_LE(cv::norm(diff, cv::NORM_INF), 1);

    int nz = cv::countNonZero(diff.reshape(1));
    EXPECT_EQ(nz, 0) << nz << " pixels differ between implementations";

}

TEST_F(ChromaticAberrationTest, Robustness)
{
    cv::ChromaticAberrationCorrector corrector;
    ASSERT_TRUE(corrector.loadCalibration(test_yaml_file));

    auto check_image = [&](const cv::Mat& src, const char* tag)
    {
        cv::Mat dst = corrector.correctImage(src);

        EXPECT_EQ(dst.size(),     src.size())   << tag;
        EXPECT_EQ(dst.type(),     src.type())   << tag;

        EXPECT_TRUE(cv::checkRange(dst, /*quiet=*/true)) << tag;

        return dst;
    };

    cv::Mat small;  cv::resize(test_image, small, cv::Size(50, 50));
    cv::Mat small_corr = check_image(small, "small");

    cv::Mat small_corr2 = corrector.correctImage(small);
    cv::Mat diff;

    cv::absdiff(small_corr, small_corr2, diff);
    
    EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0);

    cv::Mat white(128, 128, CV_8UC3, cv::Scalar::all(255));
    check_image(white, "white");

    cv::Mat black(128, 128, CV_8UC3, cv::Scalar::all(0));
    check_image(black, "black");
}

}}