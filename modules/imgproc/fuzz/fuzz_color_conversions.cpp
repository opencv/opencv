// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "fuzz_precomp.hpp"

using namespace cv;

////////////////////////////////////////////////////////////////////////////////

static void FuzzCvtColor(const Mat& image, int code) {
  InitializeOpenCV();
  try {
    Mat output;

    cvtColor(image, output, code);
  } catch (const Exception& e) {
  }
}

FUZZ_TEST(Imgproc, FuzzCvtColor)
    .WithDomains(/*image=*/Arbitrary2DMat(),
                 /*code=*/fuzztest::InRange(0, (int)COLOR_COLORCVT_MAX));
