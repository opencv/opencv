// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/objdetect/fractal_detector.hpp"

namespace opencv_test { namespace {

#define FRACTAL_TEST_CASE(imgname) \
TEST(CV_FractalAruco, can_detect_##imgname) \
{ \
    string imgPath = cvtest::findDataFile("fractal_aruco/" #imgname ".jpg"); \
    Mat image = imread(imgPath); \
    ASSERT_FALSE(image.empty()) << "Can't read image: " << imgPath; \
    aruco::FractalMarkerDetector detector; \
    detector.setParams("FRACTAL_4L_6"); \
    vector<Point3f> points3D; \
    vector<Point2f> points2D; \
    vector<aruco::FractalMarker> markers = detector.detect(image, points3D, points2D); \
    ASSERT_GT(markers.size(), 0u) << "No fractal markers detected in " << #imgname; \
    ASSERT_GT(points2D.size(), 0u) << "No 2D points detected in " << #imgname; \
    ASSERT_EQ(points3D.size(), points2D.size()) << "3D/2D points count mismatch in " << #imgname; \
    \
}

FRACTAL_TEST_CASE(hand)
FRACTAL_TEST_CASE(close)
FRACTAL_TEST_CASE(confusion)
FRACTAL_TEST_CASE(distortion)
FRACTAL_TEST_CASE(far)
}
} // namespace