// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/objdetect/mcc_checker_detector.hpp"

namespace opencv_test
{
namespace
{

using namespace std;
using namespace cv::mcc;

PERF_TEST(CV_mcc_perf, detect) {
    string path = cvtest::findDataFile("cv/mcc/mcc_ccm_test.jpg");
    Mat img = imread(path, IMREAD_COLOR);
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();

    // detect MCC24 board
    TEST_CYCLE() {
        ASSERT_TRUE(detector->process(img, MCC24, 1, false));
    }
    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
