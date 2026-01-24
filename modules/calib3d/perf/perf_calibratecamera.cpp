// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

#include "opencv2/core/utils/filesystem.hpp"

namespace opencv_test {

static std::vector<cv::String> loadBulkImages(size_t max_images)
{
    const std::string data_dir = findDataDirectory("perf/calib3d/bulk_n500", false);
    std::vector<cv::String> image_paths;
    cv::utils::fs::glob(data_dir, "*.png", image_paths, false, false);
    if (image_paths.empty())
        cv::utils::fs::glob(data_dir, "*.jpg", image_paths, false, false);
    if (image_paths.empty())
        throw SkipTestException("No images found in perf/calib3d/bulk_n500");

    std::sort(image_paths.begin(), image_paths.end());
    if (image_paths.size() > max_images)
        image_paths.resize(max_images);
    return image_paths;
}

static std::vector<Point3f> buildObjectPoints(const Size& pattern_size, float square_size)
{
    std::vector<Point3f> object_points;
    object_points.reserve(pattern_size.area());
    for (int y = 0; y < pattern_size.height; ++y)
        for (int x = 0; x < pattern_size.width; ++x)
            object_points.push_back(Point3f(x * square_size, y * square_size, 0.f));
    return object_points;
}

PERF_TEST(CalibrateCamera, DISABLED_BulkImages_N500)
{
    applyTestTag(CV_TEST_TAG_LONG, CV_TEST_TAG_SIZE_HD);

    const Size pattern_size(6, 8);
    const size_t max_images = 500;

    std::vector<cv::String> image_paths = loadBulkImages(max_images);
    std::vector<Point3f> object_pattern = buildObjectPoints(pattern_size, 1.0f);

    std::vector<std::vector<Point2f> > image_points;
    std::vector<std::vector<Point3f> > object_points;
    image_points.reserve(image_paths.size());
    object_points.reserve(image_paths.size());

    Size image_size;
    for (const auto& path : image_paths)
    {
        Mat gray = imread(path, IMREAD_GRAYSCALE);
        ASSERT_FALSE(gray.empty()) << "Can't read image: " << path;
        if (image_size.empty())
            image_size = gray.size();
        else
            ASSERT_EQ(gray.size(), image_size) << "Mismatched image size: " << path;

        std::vector<Point2f> corners;
        bool found = findChessboardCorners(
            gray, pattern_size, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        ASSERT_TRUE(found) << "Chessboard not found: " << path;

        cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                     TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

        image_points.push_back(corners);
        object_points.push_back(object_pattern);
    }

    ASSERT_FALSE(image_points.empty());

    Mat camera_matrix = Mat::eye(3, 3, CV_64F);
    Mat dist_coeffs = Mat::zeros(8, 1, CV_64F);
    std::vector<Mat> rvecs;
    std::vector<Mat> tvecs;
    double rms = 0.0;

    declare.in(image_points, object_points);
    declare.out(camera_matrix, dist_coeffs);
    declare.iterations(1);

    TEST_CYCLE()
    {
        camera_matrix = Mat::eye(3, 3, CV_64F);
        dist_coeffs = Mat::zeros(8, 1, CV_64F);
        rvecs.clear();
        tvecs.clear();
        rms = calibrateCamera(object_points, image_points, image_size,
                              camera_matrix, dist_coeffs, rvecs, tvecs, 0);
    }

    SANITY_CHECK_NOTHING();
    EXPECT_GT(rms, 0.0);
}

} // namespace opencv_test
