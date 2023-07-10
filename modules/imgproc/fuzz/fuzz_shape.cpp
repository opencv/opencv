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
