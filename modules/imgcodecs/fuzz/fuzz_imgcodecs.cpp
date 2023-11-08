// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "fuzz_precomp.hpp"

using namespace cv;

static void FuzzImdecode(const Mat& image, int mode) {
  InitializeOpenCV();

  try {
    // Create random image matrix
    Mat decoded_matrix = imdecode(image, mode);
  } catch (const Exception& e) {
  }
}

FUZZ_TEST(Imgcodecs, FuzzImdecode)
    .WithDomains(/*image:*/Arbitrary2DMat(),
                 /*mode:*/fuzztest::Arbitrary<int>());
