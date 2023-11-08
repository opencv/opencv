// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifdef OPENCV_BUILD_FUZZ_TESTS
#include "opencv2/ts/ts_fuzztest.hpp"

#include "fuzztest/fuzztest.h"

namespace opencv_fuzztest {

namespace {
int nullErrorCallback(int status, const char* func_name, const char* err_msg,
                      const char* file_name, int line, void* userdata) {
  (void)status;
  (void)func_name;
  (void)err_msg;
  (void)file_name;
  (void)line;
  (void)userdata;
  return 0;
}

}  // namespace

void InitializeOpenCV() {
  // Disable aborting on error to allow multiple fuzzing runs. OpenCV now
  // throws.
  cv::setBreakOnError(false);
  // Disable error reporting.
  cv::redirectError(nullErrorCallback, 0, 0);
}

}  // namespace opencv_fuzztest
#endif  // OPENCV_BUILD_FUZZ_TESTS
