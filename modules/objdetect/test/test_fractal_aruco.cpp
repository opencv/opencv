// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/objdetect/fractal_detector.hpp"

#include <string>

std::string fractal_aruco_images[] = {
    "hand.jpg",
    "close.jpg",
    "confusion.jpg",
    "distortion.jpg",
    "far.jpg"
};

namespace opencv_test { namespace {

typedef testing::TestWithParam<std::string> CV_FractalAruco;

TEST_P(CV_FractalAruco, can_detect)
{
    const std::string& imgname = GetParam();
    std::string imgPath = cvtest::findDataFile("fractal_aruco/" + imgname);
    Mat image = imread(imgPath);
    ASSERT_FALSE(image.empty()) << "Can't read image: " << imgPath;
    aruco::FractalDetector detector;
    detector.setParams("FRACTAL_4L_6");
    std::vector<Point3f> points3D;
    std::vector<Point2f> points2D;
    std::vector<aruco::FractalArucoMarker> markers;
    bool detected = detector.detect(image, markers, points3D, points2D);
    ASSERT_TRUE(detected) << "Fractal markers detection failed in " << imgname;
    ASSERT_GT(markers.size(), 0u) << "No fractal markers detected in " << imgname;
    ASSERT_GT(points2D.size(), 0u) << "No 2D points detected in " << imgname;
    ASSERT_EQ(points3D.size(), points2D.size()) << "3D/2D points count mismatch in " << imgname;
}

INSTANTIATE_TEST_CASE_P(
    /*empty*/, CV_FractalAruco,
    testing::ValuesIn(fractal_aruco_images)
);
}
} // namespace