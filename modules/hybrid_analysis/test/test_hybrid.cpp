#include "opencv2/hybrid_analysis.hpp"
#include <opencv2/ts.hpp>

TEST(HybridAnalysis, BasicFunctionality) {
    cv::hybrid::VisionTextFusion analyzer("test_models/vit_test.onnx", "test_models/face_test.onnx");
    cv::Mat test_image = cv::Mat::zeros(300, 300, CV_8UC3);
    float score = analyzer.analyze(test_image, "test");
    ASSERT_NEAR(score, 0.0f, 1e-3);
}
