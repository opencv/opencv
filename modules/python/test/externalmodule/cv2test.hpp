#ifndef __CV2TEST_HPP__
#define __CV2TEST_HPP__

#include "opencv2/core.hpp"

namespace cv2test
{

  CV_EXPORTS_W int image_height(const cv::Mat& image);

  class CV_EXPORTS_W TestKlass {
    const cv::Mat& _image;
  public:
    CV_WRAP TestKlass(const cv::Mat& image);
    CV_WRAP int image_height();
  };
}

#endif
