#include "cv2test.hpp"

namespace cv2test {
  int image_height(const cv::Mat& image) {
    return image.rows;
  }

  TestKlass::TestKlass(const cv::Mat& image) : _image(image) {

  }

  int TestKlass::image_height() {
    return cv2test::image_height(_image);
  }
}
