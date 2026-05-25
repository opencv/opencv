#include <opencv2/objdetect/aruco2.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main()
{
    //! [create_diamond]
    Mat image;
    aruco2::getDiamondImage(image, aruco2::DICT_ARUCO_MIP_36h12, Vec4i  (10, 11, 12, 13));
    imwrite("diamond.png", image);
    //! [create_diamond]
    std::cout << "Diamond image size: " << image.cols << "x" << image.rows << std::endl;

    //! [detect_diamonds]
    auto diamonds = aruco2::detectDiamonds(image, aruco2::DICT_ARUCO_MIP_36h12);

    for (const auto &d : diamonds) {
        std::cout << "Detected diamond with IDs: " << d.id << std::endl;
    }
    //! [detect_diamonds]

    //! [draw_diamonds]
    Mat colorImage;
    cvtColor(image, colorImage, COLOR_GRAY2BGR);
    aruco2::drawDiamonds(colorImage, diamonds);
    imshow("Detected Diamonds", colorImage);
    waitKey(0);
    //! [draw_diamonds]

    // Pose estimation (requires camera calibration data)
    //! [pose_diamond]
    Mat cameraMatrix, distCoeffs;
    // Load from calibration file in a real application, e.g.:
    //   FileStorage fs("calibration.yaml", FileStorage::READ);
    //   fs["camera_matrix"] >> cameraMatrix;
    //   fs["distortion_coeffs"] >> distCoeffs;
    float markerSize = 0.05f;
    if (!cameraMatrix.empty()) {
        for (const auto &d : diamonds) {
            Mat objPoints, imgPoints, rvec, tvec;
            aruco2::getSolvePnpPoints(d, objPoints, imgPoints, markerSize);
            solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
            aruco2::drawAxis(colorImage, cameraMatrix, distCoeffs, rvec, tvec, markerSize);
        }
    }
    //! [pose_diamond]

    return 0;
}
