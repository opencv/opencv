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
        std::vector<cv::aruco::FractalMarker> markers = detector.detect(image, points3D, points2D);

        for (const auto& marker : markers) {
            marker.draw(image);
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