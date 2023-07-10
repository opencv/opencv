#include "fuzz_precomp.hpp"

#include <string>

using namespace cv;
using namespace cv::dnn;

////////////////////////////////////////////////////////////////////////////////

void FuzzReadNetFromTensorflow(const std::string& model) {
  InitializeOpenCV();
  try {
    // Load image and prepare canvas.
    readNetFromTensorflow(model);
  } catch (const cv::Exception& e) {
  }
}

FUZZ_TEST(OpenCVDnn, FuzzReadNetFromTensorflow);
