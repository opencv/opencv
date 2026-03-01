#include "test_precomp.hpp"
#include "opencv2/features.hpp"
#include <fstream>
#include <opencv2/core/utils/filesystem.hpp>

namespace opencv_test { namespace {

TEST(Features2d_DISK, Regression)
{
    // 1. Locate the model
    // Using cvtest::findDataFile as requested by the maintainer.
    std::string modelPath;
    try {
        modelPath = cvtest::findDataFile("dnn/disk_standalone.onnx", false);
    } catch (...) {
        std::cout << "[ SKIPPED ] DISK test: model not found (check opencv_extra)." << std::endl;
        return;
    }

    // 2. Create the detector
    Ptr<Feature2D> detector;
    try {
        detector = DISK::create(modelPath);
    } catch (const cv::Exception& e) {
        FAIL() << "Failed to create DISK detector: " << e.what();
    }
    ASSERT_TRUE(detector);

    // 3. Load standard test image
    // cvtest::findDataFile throws if not found, which causes the test to fail (correct behavior for CI)
    std::string imgPath = cvtest::findDataFile("shared/lena.png");
    Mat img = imread(imgPath);
    ASSERT_FALSE(img.empty()) << "Could not load test image: " << imgPath;

    // 4. Detect and Compute
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);

    // 5. Verification
    EXPECT_GT(keypoints.size(), 2000u);
    EXPECT_EQ(descriptors.rows, (int)keypoints.size());
    EXPECT_EQ(descriptors.cols, 128);
    EXPECT_EQ(descriptors.type(), CV_32F);
}

}} // namespace
