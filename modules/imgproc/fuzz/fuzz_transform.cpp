// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cstdint>

#include "fuzz_precomp.hpp"

using namespace cv;

////////////////////////////////////////////////////////////////////////////////

static void FuzzResize(const Mat& image, int32_t new_height,
                       int32_t new_width) {
  InitializeOpenCV();
  try {
    Mat resized;
    resize(image, resized, Size(new_width, new_height));
  } catch (const Exception& e) {
  }
}

FUZZ_TEST(Imgproc, FuzzResize)
    .WithDomains(/*image=*/Arbitrary2DMat(),
                 /*new_height=*/fuzztest::InRange<int32_t>(0, 1e4),
                 /*new_width=*/fuzztest::InRange<int32_t>(0, 1e4));

////////////////////////////////////////////////////////////////////////////////

static void FuzzWarpAffine(const Mat& image, const Matx23f& M,
                           int32_t new_height, int32_t new_width,
                           int32_t inter_flag, bool use_warp_inverse_map,
                           int32_t border_mode) {
  InitializeOpenCV();
  try {
    Mat warped;
    const int inverse_map_flag =
        use_warp_inverse_map ? WARP_INVERSE_MAP : 0;
    warpAffine(image, warped, M, Size(new_width, new_height),
               inter_flag | inverse_map_flag, border_mode);
  } catch (const Exception& e) {
  }
}

FUZZ_TEST(Imgproc, FuzzWarpAffine)
    .WithDomains(/*image=*/Arbitrary2DMat(),
                 /*M=*/ArbitraryMatx<float, 2, 3>(),
                 /*new_height=*/fuzztest::InRange<int32_t>(0, 1e4),
                 /*new_width=*/fuzztest::InRange<int32_t>(0, 1e4),
                 /*inter_flag=*/fuzztest::InRange(0, (int)INTER_MAX),
                 /*use_warp_inverse_map=*/fuzztest::Arbitrary<bool>(),
                 /*border_mode=*/
                 fuzztest::InRange(0, (int)BORDER_ISOLATED + 1));
