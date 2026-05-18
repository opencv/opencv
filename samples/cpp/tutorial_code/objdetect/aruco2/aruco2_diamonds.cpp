#include <opencv2/objdetect/aruco2.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main()
{
    //! [create_diamond]
    Mat diamondImage;
    aruco2::DictionaryType dict = aruco2::DICT_ARUCO_MIP_36h12;
    Vec4i ids(10, 11, 12, 13);
    aruco2::getDiamondImage(diamondImage, dict, ids);
    imwrite("diamond.png", diamondImage);
    //! [create_diamond]

    std::cout << "Diamond image size: " << diamondImage.cols << "x" << diamondImage.rows << std::endl;

    // Place diamond on a white scene for detection
    Mat scene(diamondImage.rows + 100, diamondImage.cols + 100, CV_8UC1, Scalar(255));
    diamondImage.copyTo(scene(Rect(50, 50, diamondImage.cols, diamondImage.rows)));

    //! [detect_diamonds]
    Mat image = scene.clone();
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
    Mat cameraMatrix, distCoeffs; // Load from calibration file in a real application
    float markerSize = 0.05f;
    for (const auto &d : diamonds) {
        Mat objPoints, imgPoints, rvec, tvec;
        aruco2::getSolvePnpPoints(d, objPoints, imgPoints, markerSize);
        solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);
        aruco2::drawAxis(colorImage, cameraMatrix, distCoeffs, rvec, tvec, markerSize);
    }
    //! [pose_diamond]

    return 0;
}
