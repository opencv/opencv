#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        cv::String path = cv::samples::findFile("opencv/samples/data/fractal_4l_a.jpg", false);
        if (path.empty()) {
            std::cerr << "Image not found!" << std::endl;
            return -1;
        }
        cv::Mat image = cv::imread(path);
        cv::aruco::FractalMarkerDetector detector;

        detector.setParams("FRACTAL_4L_6", 1000);
        // detector.setParams("FRACTAL_4L_6");
        std::vector<cv::Point3f> points3D;
        std::vector<cv::Point2f> points2D;
        std::vector<std::vector<Point2f>> marker_points;
        std::vector<int> marker_ids;

        detector.detect(image, marker_ids, marker_points, points3D, points2D);

        for (size_t i = 0 i < marker_points.size(); i++) {
            marker.draw(image, marker_points);
        }
        for (const auto& pt : points2D) {
            cv::circle(image, pt, 5, cv::Scalar(0, 255, 0), cv::FILLED);
        }

        std::string outputPath = "result.png";
        cv::imwrite(outputPath, image);
        std::cout << "Saved: " << outputPath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
