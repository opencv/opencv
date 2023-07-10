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
