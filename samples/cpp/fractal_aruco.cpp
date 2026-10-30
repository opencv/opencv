#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

int main()
{
    cv::String path = cv::samples::findFile("fractal_4l_a.jpg", false);
    if (path.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }
    cv::Mat image = cv::imread(path);
    cv::aruco::FractalDetector detector;

    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point2f> points2D;
    std::vector<std::vector<cv::Point2f>> marker_points;
    std::vector<int> marker_ids;

    detector.detect(image, marker_points, marker_ids, points3D, points2D);
    cv::aruco::drawDetectedFractalMarkers(image, marker_points, marker_ids);

    std::string outputPath = "result.png";
    cv::imwrite(outputPath, image);
    std::cout << "Saved: " << outputPath << std::endl;

    return 0;
}
