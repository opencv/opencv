#include <opencv2/objdetect/aruco2.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: aruco2_pose <image> [calibration.yaml]" << std::endl;
        return 1;
    }

    Mat image = imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Could not open image: " << argv[1] << std::endl;
        return 1;
    }

    Mat cameraMatrix, distCoeffs;
    if (argc >= 3) {
        FileStorage fs(argv[2], FileStorage::READ);
        fs["camera_matrix"] >> cameraMatrix;
        fs["distortion_coeffs"] >> distCoeffs;
    } else {
        // Approximate intrinsics when no calibration is available
        double f = image.cols;
        cameraMatrix = (Mat_<double>(3, 3) << f, 0, image.cols / 2.0,
                                              0, f, image.rows / 2.0,
                                              0, 0, 1);
        distCoeffs = Mat::zeros(1, 5, CV_64F);
    }

    float markerSize = 0.05f; // physical side length in meters

    //! [pose_single_marker]
    for (const auto &m : aruco2::detectFiducialMarkers(image, aruco2::DICT_ARUCO_MIP_36h12)) {
        Mat objPoints, imgPoints, rvec, tvec;
        aruco2::getSolvePnpPoints(m, objPoints, imgPoints, markerSize);
        solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
        aruco2::drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, markerSize);
    }
    //! [pose_single_marker]

    //! [pose_board]
    aruco2::GridBoard board;
    if (aruco2::detectGridBoard(image, Size(9, 5), aruco2::DICT_ARUCO_MIP_36h12, board)) {
        Mat objPoints, imgPoints, rvec, tvec;
        aruco2::getSolvePnpPoints(board, objPoints, imgPoints, markerSize);
        solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
        aruco2::drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, markerSize);
    }
    //! [pose_board]

    //! [pose_fractal]
    for (const auto &f : aruco2::detectFractals(image, aruco2::FRACTAL_3L_6)) {
        Mat objPoints, imgPoints, rvec, tvec;
        aruco2::getSolvePnpPoints(f, objPoints, imgPoints, markerSize);
        solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
        aruco2::drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, markerSize);
    }
    //! [pose_fractal]

    imshow("Pose Estimation", image);
    waitKey(0);

    return 0;
}
