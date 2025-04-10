#ifndef OPENCV_HYBRID_ANALYSIS_HPP
#define OPENCV_HYBRID_ANALYSIS_HPP

#include <opencv2/dnn.hpp>
#include <string>

namespace cv {
namespace hybrid {

class CV_EXPORTS_W VisionTextFusion {
public:
    CV_WRAP VisionTextFusion(const std::string& vit_model_path, const std::string& face_detector_path);
    CV_WRAP float analyze(cv::InputArray image, const std::string& text);

private:
    cv::dnn::Net face_detector_;
    cv::dnn::Net vit_model_;
    std::vector<float> text_to_embedding(const std::string& text);
    float cosine_similarity(const cv::Mat& vec1, const cv::Mat& vec2);
};

}} // namespace cv::hybrid
#endif
