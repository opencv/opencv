// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "fuzz_precomp.hpp"

using namespace cv;

////////////////////////////////////////////////////////////////////////////////

static void FuzzFitEllipse(const Mat& points) {
  InitializeOpenCV();
  try {
    const RotatedRect rect = fitEllipse(points);
    (void)rect;
  } catch (const Exception& e) {
  }
}

FUZZ_TEST(Imgproc, FuzzFitEllipse)
    .WithDomains(/*points=*/Arbitrary2DMat());
