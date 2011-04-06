#include "cvsample.h"
#include <opencv2/imgproc/imgproc.hpp>

void CVSample::canny(const cv::Mat& input, cv::Mat& output, int edgeThresh)
{
  if (input.empty())
    return;
  cv::Mat gray;
  if (input.channels() == 3)
  {
    cv::cvtColor(input, gray, CV_RGB2GRAY);
  }
  else
    gray = input;
  cv::Canny(gray, output, edgeThresh, edgeThresh * 3, 3);
}

void CVSample::invert(cv::Mat& inout)
{
  cv::bitwise_not(inout, inout);
}
void CVSample::blur(cv::Mat& inout, int half_kernel_size)
{
  int ksz = half_kernel_size*2 + 1;
  cv::Size kernel(ksz,ksz);
  cv::blur(inout,inout,kernel);
}
