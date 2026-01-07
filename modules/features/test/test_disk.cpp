#include "test_precomp.hpp"
#include "opencv2/disk.hpp"

namespace opencv_test { namespace {

TEST(Features2d_DISK, Regression)
{

    std::string modelPath = "disk_standalone.onnx";

    // Check if model exists
    std::ifstream f(modelPath.c_str());
    if (!f.good()) {
        std::cout << "[ SKIPPED ] DISK test: model file '" << modelPath << "' not found in current directory." << std::endl;
        return;
    }
    // 1. Create the detector
    Ptr<Feature2D> detector;
    try {
        detector = DISK::create(modelPath);
    } catch (const cv::Exception& e) {
        FAIL() << "Failed to create DISK detector: " << e.what();
    }
    ASSERT_TRUE(detector);

    // 2. Load standard test image (Lena)
    std::string imgPath = cvtest::TS::ptr()->get_data_path() + "cv/shared/lena.png";
    Mat img = imread(imgPath);

    if (img.empty()) {
        // Fallback for local testing if TS path isn't set perfectly
        imgPath = "shared/lena.png";
        img = imread(imgPath);
    }

    ASSERT_FALSE(img.empty()) << "Could not load test image: " << imgPath;

    // 3. Detect and Compute
    std::vector<KeyPoint> keypoints;
    Mat descriptors;

    detector->detectAndCompute(img, noArray(), keypoints, descriptors);

    // 4. Verification
    // We expect around 2048 keypoints (as configured in the model)
    // We allow a small margin of error or filtering
    EXPECT_GT(keypoints.size(), 2000u);

    // Descriptors should match keypoint count
    EXPECT_EQ(descriptors.rows, (int)keypoints.size());

    // DISK descriptors are 128-dim
    EXPECT_EQ(descriptors.cols, 128);

    // Data type should be 32-bit float
    EXPECT_EQ(descriptors.type(), CV_32F);
}

}} // namespace