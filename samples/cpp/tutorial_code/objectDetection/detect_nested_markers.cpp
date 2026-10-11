#include <opencv2/geometry.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include "aruco_samples_utility.hpp"

namespace {
const char* about = "Nested ArUco marker tutorial";
const char* keys =
        "{help h usage ? |        | print this message }"
        "{mode           | create | mode: create, detect, pose }"
        "{c              |        | camera intrinsic parameters file for pose mode }";

void createNestedMarker() {
    //! [nested_marker_create_cpp]
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_NESTED_10);
    cv::Mat marker;
    cv::aruco::generateImageMarkerNested(dictionary, 0, 1200, marker);  // outer id 0, inner id 1
    cv::imwrite("pair_0.png", marker);
    //! [nested_marker_create_cpp]
}

void detectNestedMarkers(cv::VideoCapture& cap) {
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_NESTED_10);
    //! [nested_marker_detect_cpp]
    cv::aruco::DetectorParameters params;
    params.detectNestedMarkers = true;
    cv::aruco::ArucoDetector detector(dictionary, params);

    cv::Mat frame;
    while(cap.read(frame)) {
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        detector.detectMarkers(frame, corners, ids);
        cv::aruco::drawDetectedMarkers(frame, corners, ids);
        cv::imshow("nested markers", frame);
        if(cv::waitKey(1) == 27) break;
    }
    //! [nested_marker_detect_cpp]
}

void estimateNestedMarkerPose(cv::VideoCapture& cap, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_NESTED_10);
    cv::aruco::DetectorParameters params;
    params.detectNestedMarkers = true;
    cv::aruco::ArucoDetector detector(dictionary, params);
    //! [nested_marker_pose_cpp]
    float sideLength = 0.20f;  // printed outer side in meters
    cv::Mat outerPts, innerPts;
    cv::aruco::getNestedMarkerObjectPoints(dictionary, 0, sideLength, outerPts, innerPts);
    cv::aruco::Board board(std::vector<cv::Mat>{outerPts, innerPts}, dictionary, std::vector<int>{0, 1});

    cv::Mat frame;
    while(cap.read(frame)) {
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        detector.detectMarkers(frame, corners, ids);
        cv::aruco::drawDetectedMarkers(frame, corners, ids);
        if(!ids.empty()) {
            cv::Mat objPoints, imgPoints, rvec, tvec;
            board.matchImagePoints(corners, ids, objPoints, imgPoints);
            if(!objPoints.empty() && cv::solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec))
                cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, sideLength * 0.5f);
        }
        cv::imshow("nested marker pose", frame);
        if(cv::waitKey(1) == 27) break;
    }
    //! [nested_marker_pose_cpp]
}
} // namespace

int main(int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if(parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    const std::string mode = parser.get<std::string>("mode");
    if(mode == "create") {
        createNestedMarker();
        return 0;
    }
    if(mode != "detect" && mode != "pose") {
        std::cerr << "unknown mode: " << mode << std::endl;
        return 1;
    }

    cv::Mat cameraMatrix, distCoeffs;
    if(mode == "pose" && (!parser.has("c") ||
                          !readCameraParameters(parser.get<std::string>("c"), cameraMatrix, distCoeffs) ||
                          cameraMatrix.empty())) {
        std::cerr << "pose mode requires -c=<camera parameters file>" << std::endl;
        return 1;
    }
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) {
        std::cerr << "could not open camera 0" << std::endl;
        return 1;
    }
    if(mode == "detect")
        detectNestedMarkers(cap);
    else
        estimateNestedMarkerPose(cap, cameraMatrix, distCoeffs);
    return 0;
}
