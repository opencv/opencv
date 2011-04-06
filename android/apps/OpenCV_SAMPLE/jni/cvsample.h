#pragma once

#include <opencv2/core/core.hpp>

class CVSample
{
public:
  void canny(const cv::Mat& input, cv::Mat& output, int edgeThresh);
  void invert(cv::Mat& inout);
  void blur(cv::Mat& inout, int half_kernel_size);
};
