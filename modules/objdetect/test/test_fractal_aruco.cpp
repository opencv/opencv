// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/objdetect/fractal_detector.hpp"

#include <string>

//#define RECORD_POINTS

std::string fractal_aruco_cases[] = { "hand", "close", "confusion", "distortion", "far" };

namespace opencv_test { namespace {

typedef testing::TestWithParam<std::string> FractalArucoTest;

TEST_P(FractalArucoTest, detect)
{
    const std::string& imgname = GetParam();
    std::string imgPath = cvtest::findDataFile("fractal_aruco/" + imgname + ".jpg");
    std::string gtPath = cvtest::findDataFile("fractal_aruco/" + imgname + ".yml");

    Mat image = imread(imgPath);
    ASSERT_FALSE(image.empty()) << "Can't read image: " << imgPath;
    aruco::FractalMarkerDictionary dictionary = aruco::getPredefinedFractalArucoDictionary(aruco::FRACTAL_4L_6);
    aruco::FractalDetector detector(dictionary);
    std::vector<Point3f> points3D;
    std::vector<Point2f> points2D;
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_points;

    bool detected = detector.detect(image, marker_points, marker_ids, points3D, points2D);

    std::vector<cv::Point3f> points3D_gt;
    std::vector<cv::Point2f> points2D_gt;

#ifdef RECORD_POINTS
    cv::FileStorage fs(gtPath, cv::FileStorage::WRITE);
    fs << "points_2d" << points2D;
    fs << "points_3d" << points3D;
    fs.release();

    points3D_gt = points3D;
    points2D_gt = points2D;
#else
    cv::FileStorage fs(gtPath, cv::FileStorage::READ);
    fs["points_2d"] >> points2D_gt;
    fs["points_3d"] >> points3D_gt;
    fs.release();
#endif

    ASSERT_TRUE(detected) << "Fractal markers detection failed in " << imgname;

    ASSERT_EQ(points2D.size(), points2D_gt.size());
    for (size_t i = 0; i < points2D.size(); i++)
    {
        // TODO: optimize epsilon
        EXPECT_NEAR(points2D[i].x, points2D_gt[i].x, 0.5);
        EXPECT_NEAR(points2D[i].y, points2D_gt[i].y, 0.5);
    }

    ASSERT_EQ(points3D.size(), points3D_gt.size());
    for (size_t i = 0; i < points3D.size(); i++)
    {
        EXPECT_NEAR(points3D[i].x, points3D_gt[i].x, 1e-3);
        EXPECT_NEAR(points3D[i].y, points3D_gt[i].y, 1e-3);
        EXPECT_NEAR(points3D[i].z, points3D_gt[i].z, 1e-3);
    }
}

INSTANTIATE_TEST_CASE_P(
    /*empty*/, FractalArucoTest,
    testing::ValuesIn(fractal_aruco_cases)
);
}
} // namespace
