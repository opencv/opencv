//! [aruco2hdr]
#include <opencv2/objdetect/aruco2.hpp>
//! [aruco2hdr]
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main()
{
    //! [generate_marker]
    Mat markerImage;
    aruco2::getFiducialMarker(markerImage, aruco2::DICT_ARUCO_MIP_36h12, 42);
    imwrite("marker42.png", markerImage);
    //! [generate_marker]

    std::cout << "Generated marker size: " << markerImage.cols << "x" << markerImage.rows << std::endl;

    // Place marker on a white scene for detection
    Mat scene(300, 300, CV_8UC1, Scalar(255));
    markerImage.copyTo(scene(Rect(50, 50, markerImage.cols, markerImage.rows)));

    //! [detect_single]
    Mat image = scene.clone();
    auto markers = aruco2::detectFiducialMarkers(image, aruco2::DICT_ARUCO_MIP_36h12);

    for (const auto &m : markers) {
        std::cout << "Detected marker ID: " << m.id << " at " << m.corners[0] << std::endl;
    }
    //! [detect_single]

    //! [multi_dict]
    using namespace aruco2;
    auto multiMarkers = detectFiducialMarkers(image, {DICT_ARUCO_MIP_36h12, DICT_APRILTAG_36h11});

    for (const auto &m : multiMarkers) {
        std::string dictName = (m.dict == DICT_ARUCO_MIP_36h12) ? "ArUco" : "AprilTag";
        std::cout << "Found " << dictName << " marker ID: " << m.id << std::endl;
    }
    //! [multi_dict]

    //! [params]
    DetectionParameters params;
    params.boxFilterSize = 15;
    params.thres = 3;
    params.errorCorrectionRate = 0.0;

    auto paramsMarkers = aruco2::detectFiducialMarkers(image, aruco2::DICT_ARUCO_MIP_36h12, params);
    //! [params]

    std::cout << "Custom parameters detection found " << paramsMarkers.size() << " marker(s)" << std::endl;

    //! [draw_markers]
    Mat colorImage;
    cvtColor(image, colorImage, COLOR_GRAY2BGR);
    aruco2::drawFiducialMarkers(colorImage, markers);
    imshow("Detected Markers", colorImage);
    waitKey(0);
    //! [draw_markers]

    return 0;
}
