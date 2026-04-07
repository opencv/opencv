// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

#include "opencv2/core/utils/filesystem.hpp"

//#define SAVE_IMAGE_POINTS

namespace opencv_test {

#ifdef SAVE_IMAGE_POINTS

static std::vector<std::string> loadBulkImages(size_t max_images)
{
    const std::string data_dir = findDataDirectory("perf/calib3d/bulk_n500", false);
    std::vector<std::string> image_paths;
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

static std::vector<std::vector<Point2f>> buildImagePoints(const std::vector<std::string>& image_paths, const cv::Size pattern_size)
{
    std::vector<std::vector<Point2f>> image_points;
    image_points.reserve(image_paths.size());

    for (const auto& path : image_paths)
    {
        Mat gray = imread(path, IMREAD_GRAYSCALE);
        if (gray.empty())
        {
            printf("Can't read image: %s\n", path.c_str());
            return std::vector<std::vector<Point2f>>();
        }

        std::vector<Point2f> corners;
        bool found = findChessboardCorners(
            gray, pattern_size, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (found)
        {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            image_points.push_back(corners);
        }
    }

    return image_points;
}

static void saveImagePoints(const std::vector<std::vector<Point2f>>& image_points)
{
    const std::string points_file = "bulk_n500.yaml";
    cv::FileStorage fs(points_file, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    if (!fs.isOpened())
    {
        printf("Cannot open yaml config \"%s\" for output\n", points_file.c_str());
    }
    fs << "count" << (int)image_points.size();
    for (int i = 0; i < (int)image_points.size(); i++)
    {
        fs << cv::format("frame_%d", i) << image_points[i];
    }

    fs.release();
}

#else

static std::vector<std::vector<Point2f>> loadImagePoints()
{
    const std::string points_file = findDataFile("perf/calib3d/bulk_n500.yaml");
    cv::FileStorage fs(points_file, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        printf("Cannot open yaml config \"%s\" for output\n", points_file.c_str());
        return std::vector<std::vector<Point2f>>();
    }
    int count = fs["count"];
    std::vector<std::vector<Point2f>> image_points(count);

    for (int i = 0; i < (int)image_points.size(); i++)
    {
        fs[cv::format("frame_%d", i)] >> image_points[i];
    }

    fs.release();

    return image_points;
}

#endif

static std::vector<std::vector<Point3f>> buildObjectPoints(const Size& pattern_size, float square_size, size_t count)
{
    std::vector<Point3f> board;
    board.reserve(pattern_size.area());

    for (int y = 0; y < pattern_size.height; ++y)
        for (int x = 0; x < pattern_size.width; ++x)
            board.push_back(Point3f(x * square_size, y * square_size, 0.f));

    std::vector<std::vector<Point3f> > object_points;
    object_points.reserve(count);
    for (size_t i = 0; i < count; i++)
        object_points.push_back(board);

    return object_points;
}

PERF_TEST(CalibrateCamera, BulkImages_N500)
{
    // NOTE: The images archive is published at https://dl.opencv.org/data/bulk_n500.zip
    applyTestTag(CV_TEST_TAG_LONG);

    const cv::Size pattern_size(6, 8);
    const cv::Size image_size(1280, 720);

#ifdef SAVE_IMAGE_POINTS
    std::vector<std::string> image_paths = loadBulkImages(500);
    std::vector<std::vector<Point2f>> image_points = buildImagePoints(image_paths, pattern_size);
    ASSERT_FALSE(image_points.empty());
    saveImagePoints(image_points);
#else
    std::vector<std::vector<Point2f>> image_points = loadImagePoints();
    ASSERT_FALSE(image_points.empty());
#endif

    std::vector<std::vector<Point3f> > object_points = buildObjectPoints(pattern_size, 1.0f, image_points.size());

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

    EXPECT_NEAR(rms, 1.768263, 1e-4);
    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
