// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

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
