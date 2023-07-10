#ifndef OPENCV_TS_FUZZER_UTILS_HPP_
#define OPENCV_TS_FUZZER_UTILS_HPP_

#include <cstdint>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "opencv2/core.hpp"

namespace opencv_fuzztest {

// Creates a 2D matrix from a stream of random data.
// The size of the matrix is determined from the length of the data available.
// Domain of arbitrary OpenCV matrices.
inline auto Arbitrary2DMat() {
  auto depth = fuzztest::InRange(0, CV_DEPTH_MAX);
  auto channels = fuzztest::InRange(1, CV_CN_MAX);
  auto ratio = fuzztest::InRange(0.f, 1.f);
  auto bytes = fuzztest::Arbitrary<std::vector<uint8_t>>();

  return fuzztest::Map(
      [](int depth, int channels, float ratio,
         const std::vector<uint8_t>& bytes) {
        unsigned int type = CV_MAKETYPE(depth, channels);

        // Calculate bytes per pixel for the generated image type
        size_t bytes_per_pixel = CV_ELEM_SIZE(type);

        // Compute the maximum number of pixels left.
        const int pixels = bytes.size() / bytes_per_pixel;
        int width = std::round(pixels * ratio);
        int height = (width == 0) ? 0 : pixels / width;

        cv::Mat mat(height, width, type, const_cast<uint8_t*>(bytes.data()));

        // In case of an empty matrix, clone does not preserve the type.
        if (mat.empty()) return cv::Mat(0, 0, type);

        // Clone data to make sure we do not refer to temporary data.
        mat = mat.clone();

        if (depth == CV_32F) {
          // OpenCV does not handle NaN or out-of-bounds input very well.
          // See b/264556446 for example.
          cv::patchNaNs(mat);
        }
        return mat;
      },
      depth, channels, ratio, bytes);
}

// Creates an arbitrary Matx according to the input types.
template <typename T, int rows, int cols>
inline auto ArbitraryMatx() {
  auto bytes = fuzztest::Arbitrary<std::vector<T>>().WithSize(rows * cols);

  return fuzztest::Map(
      [](const std::vector<T>& bytes) {
        cv::Matx<T, rows, cols> mat(const_cast<T*>(bytes.data()));
        // A clone is automatically done (no internal pointer in Matx).
        return mat;
      },
      bytes);
}

// Initializes OpenCV to throw on errors and not redirect errors.
void InitializeOpenCV();

}  // namespace opencv_fuzztest

#endif  // OPENCV_TS_FUZZER_UTILS_HPP_
