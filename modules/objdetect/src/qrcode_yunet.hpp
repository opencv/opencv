#pragma once


#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp> 

class YunetWrapper {
public:
    YunetWrapper();
    YunetWrapper(const std::string& model_path);
    ~YunetWrapper() = default;

    bool detect(const cv::Mat& img, cv::Rect& out_box);
    // Multi-box detection: return all QR candidate boxes after NMS.
    bool detectMulti(const cv::Mat& img, std::vector<cv::Rect>& out_boxes);


private:
    std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                         const std::vector<float>& scores,
                         float thresh);

private:
    cv::dnn::Net net_;
    std::vector<std::string> out_names_;
    
    const int input_w_ = 640;
    const int input_h_ = 640;
};