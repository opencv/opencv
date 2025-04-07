#include <opencv2/hybrid_analysis.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

using namespace cv;
using namespace cv::hybrid;

VisionTextFusion::VisionTextFusion(const std::string& vit_model_path, const std::string& face_detector_path) {
    face_detector_ = dnn::readNetFromONNX(face_detector_path);
    vit_model_ = dnn::readNetFromONNX(vit_model_path);
}

float VisionTextFusion::analyze(cv::InputArray image, const std::string& text) {
    // Face detection
    cv::Mat frame = image.getMat();
    cv::Mat blob = dnn::blobFromImage(frame, 1.0, cv::Size(320, 240));
    face_detector_.setInput(blob);
    cv::Mat detections = face_detector_.forward();

    // ViT feature extraction
    cv::Mat vit_blob = dnn::blobFromImage(frame, 1.0 / 255, cv::Size(224, 224));
    vit_model_.setInput(vit_blob);
    cv::Mat vit_features = vit_model_.forward();

    // Text embedding
    cv::Mat text_embedding(text_to_embedding(text), true);

    // Similarity calculation
    return cosine_similarity(vit_features, text_embedding);
}

std::vector<float> VisionTextFusion::text_to_embedding(const std::string& text) {
    std::vector<float> embedding(512, 0.0f);
    for (size_t i = 0; i < std::min(text.size(), (size_t)512); ++i) {
        embedding[i] = static_cast<float>(text[i]) / 255.0f;
    }
    return embedding;
}

float VisionTextFusion::cosine_similarity(const cv::Mat& vec1, const cv::Mat& vec2) {
    return vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2));
}
