#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {
const char* about = "Nested ArUco marker tutorial";

const char* keys =
        "{help h usage ? |        | print this message }"
        "{mode           | create | mode: create, detect, pose, custom-detect }"
        "{c              |        | camera intrinsic parameters file for pose mode }";

bool readCameraParameters(const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(!fs.isOpened()) {
        return false;
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return !cameraMatrix.empty() && !distCoeffs.empty();
}

int createNestedMarker() {
    //! [nested_marker_create_cpp]
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_NESTED_10);
    cv::Mat marker;
    cv::aruco::generateImageMarkerNested(dictionary, 0, 1200, marker);  // outer id 0, inner id 1
    cv::imwrite("pair_0.png", marker);
    //! [nested_marker_create_cpp]
    return 0;
}

int detectNestedMarkers() {
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_NESTED_10);

    //! [nested_marker_detect_cpp]
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) {
        std::cerr << "could not open camera 0" << std::endl;
        return 1;
    }

    cv::aruco::DetectorParameters params;
    params.detectNestedMarkers = true;
    cv::aruco::ArucoDetector detector(dictionary, params);

    cv::Mat frame;
    while(cap.read(frame)) {
        std::vector<std::vector<cv::Point2f>> corners, rejected;
        std::vector<int> ids;
        detector.detectMarkers(frame, corners, ids, rejected);
        cv::aruco::drawDetectedMarkers(frame, corners, ids);
        cv::imshow("nested markers", frame);
        if(cv::waitKey(1) == 27) break;
    }
    //! [nested_marker_detect_cpp]
    return 0;
}

int estimateNestedMarkerPose(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_NESTED_10);
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) {
        std::cerr << "could not open camera 0" << std::endl;
        return 1;
    }

    cv::aruco::DetectorParameters params;
    params.detectNestedMarkers = true;
    cv::aruco::ArucoDetector detector(dictionary, params);

    cv::Mat frame;
    //! [nested_marker_pose_cpp]
    float sideLength = 0.20f;  // printed outer side in meters
    cv::Mat outerPts, innerPts;
    cv::aruco::getNestedMarkerObjectPoints(dictionary, 0, sideLength, outerPts, innerPts);
    cv::aruco::Board board(std::vector<cv::Mat>{outerPts, innerPts}, dictionary,
                           std::vector<int>{0, 1});

    while(cap.read(frame)) {
        std::vector<std::vector<cv::Point2f>> corners, rejected;
        std::vector<int> ids;
        detector.detectMarkers(frame, corners, ids, rejected);
        cv::aruco::drawDetectedMarkers(frame, corners, ids);

        if(!ids.empty()) {
            cv::Mat objPoints, imgPoints;
            board.matchImagePoints(corners, ids, objPoints, imgPoints);

            if(objPoints.total() >= 4) {
                cv::Mat rvec, tvec;
                bool ok = cv::solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);

                if(ok) {
                    cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, sideLength * 0.5f);
                }
            }
        }

        cv::imshow("nested marker pose", frame);
        if(cv::waitKey(1) == 27) break;
    }
    //! [nested_marker_pose_cpp]
    return 0;
}

int detectCustomNestedMarkers() {
    //! [nested_custom_detect_cpp]
    const std::string name = "billboard";  // or "constellation"
    const std::string dir = "custom_nested_output/";

    cv::FileStorage storage(dir + name + ".yml", cv::FileStorage::READ);
    cv::aruco::Dictionary dictionary;
    if(!storage.isOpened() || !dictionary.readDictionary(storage.root())) {
        std::cerr << "dictionary not found or invalid" << std::endl;
        return 1;
    }
    storage.release();

    cv::aruco::DetectorParameters params;
    params.detectNestedMarkers = true;
    params.detectInvertedMarker = true;
    params.errorCorrectionRate = 0.0;
    params.perspectiveRemovePixelPerCell = 20;

    cv::aruco::ArucoDetector detector(dictionary, params);
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) {
        std::cerr << "could not open camera 0" << std::endl;
        return 1;
    }

    cv::Mat frame;
    while(cap.read(frame)) {
        std::vector<std::vector<cv::Point2f>> corners, rejected;
        std::vector<int> ids;
        detector.detectMarkers(frame, corners, ids, rejected);
        cv::aruco::drawDetectedMarkers(frame, corners, ids);

        cv::imshow("custom nested markers", frame);
        if(cv::waitKey(1) == 27) {
            break;
        }
    }
    //! [nested_custom_detect_cpp]
    return 0;
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
    if(!parser.check()) {
        parser.printErrors();
        return 1;
    }

    if(mode == "create") {
        return createNestedMarker();
    }
    if(mode == "detect") {
        return detectNestedMarkers();
    }
    if(mode == "custom-detect") {
        return detectCustomNestedMarkers();
    }
    if(mode == "pose") {
        if(!parser.has("c")) {
            std::cerr << "pose mode requires -c=<camera_parameters.yml>" << std::endl;
            return 1;
        }

        cv::Mat cameraMatrix, distCoeffs;
        if(!readCameraParameters(parser.get<std::string>("c"), cameraMatrix, distCoeffs)) {
            std::cerr << "could not read camera parameters" << std::endl;
            return 1;
        }

        return estimateNestedMarkerPose(cameraMatrix, distCoeffs);
    }

    std::cerr << "unknown mode: " << mode << std::endl;
    parser.printMessage();
    return 1;
}
