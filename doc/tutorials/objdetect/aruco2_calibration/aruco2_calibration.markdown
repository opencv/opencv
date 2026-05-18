Calibration with ArUco2 Boards {#tutorial_aruco2_calibration}
==============================

@prev_tutorial{tutorial_aruco2_boards}
@next_tutorial{tutorial_aruco2_pose}

|    |    |
| -: | :- |
| Original author | Rafael Muñoz-Salinas |
| Compatibility    | OpenCV >= 5.0.0 |

Camera calibration is the process of estimating the intrinsic parameters of a camera (focal length, optical center) and its lens distortion coefficients. This is essential for any application that requires mapping between 2D image coordinates and 3D world coordinates, such as pose estimation, 3D reconstruction, or removing lens distortion.

This tutorial shows how to perform camera calibration using **ArUco2 boards**.

Why Use ArUco2 Boards for Calibration?
--------------------------------------

Traditional calibration often uses a chessboard pattern. While effective, chessboards require the entire pattern to be visible to detect the grid. ArUco2 boards offer several advantages:

-   **Partial Visibility**: Calibration points can be extracted even if only a fraction of the board is visible. This allows you to capture data at the very edges of the image sensor, where distortion is most extreme.
-   **No Ambiguity**: Each square on an ArUco2 board is a unique marker. This eliminates "phase" errors where a symmetric chessboard might be misoriented or its corners misidentified.
-   **Robustness**: The binary codification of markers provides error correction, making detection reliable even under uneven lighting or partial occlusion.

Calibration Process
-------------------

The calibration process involves three main steps:
1.  **Print a Board**: Generate and print an ArUco2 board.
2.  **Capture Images**: Take multiple photos of the board from different angles and distances, ensuring the board covers different parts of the image sensor.
3.  **Run Calibration**: Use the detected board corners to estimate the camera parameters.

### Detection Example

When an ArUco2 board is detected, the `detectGridBoard()` function identifies all visible markers and their corners. The following image shows a board detected in a cluttered scene:

![Detected ArUco2 Board](calib_detection.jpg)

Note how the board is correctly identified even when tilted and partially outside the frame.

Calibration Code
----------------

The following example demonstrates a complete calibration script. It iterates through a folder of images, detects the board in each, and performs the final calibration.

@code{.cpp}
#include <opencv2/objdetect/aruco2.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main() {
    float markerSize = 0.05f; // 5 cm
    std::string folder = "path/to/calibration_images";

    std::vector<std::vector<cv::Point3f>> allObjPts;
    std::vector<std::vector<cv::Point2f>> allImgPts;
    cv::Size imageSize;

    for (auto& entry : fs::directory_iterator(folder)) {
        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) continue;
        if (imageSize.width == 0) imageSize = image.size();

        cv::aruco2::GridBoard board;
        if (cv::aruco2::detectGridBoard(image, cv::Size(9, 5), cv::aruco2::DICT_ARUCO_MIP_36h12, board)) {
            cv::Mat imgPtsMat, objPtsMat;
            // Get 3D-2D correspondences
            cv::aruco2::getSolvePnpPoints(board, objPtsMat, imgPtsMat, markerSize);

            // Convert to vector for calibrateCamera
            std::vector<cv::Point2f> imgPts(imgPtsMat.begin<cv::Point2f>(), imgPtsMat.end<cv::Point2f>());
            std::vector<cv::Point3f> objPts(objPtsMat.begin<cv::Point3f>(), objPtsMat.end<cv::Point3f>());

            allImgPts.push_back(imgPts);
            allObjPts.push_back(objPts);
        }
    }

    // Perform calibration
    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rpe = cv::calibrateCamera(allObjPts, allImgPts, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

    std::cout << "Reprojection error: " << rpe << " px" << std::endl;
    std::cout << "Camera Matrix:\n" << cameraMatrix << std::endl;

    // Save results
    cv::FileStorage fs_out("calibration.yaml", cv::FileStorage::WRITE);
    fs_out << "camera_matrix" << cameraMatrix;
    fs_out << "distortion_coeffs" << distCoeffs;
    fs_out.release();

    return 0;
}
@endcode

### Understanding the Output

The `calibration.yaml` file contains the estimated parameters. A typical output looks like this:

@code{.yaml}
image_size: [ 2000, 1126 ]
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1364.09, 0., 928.58, 0., 1365.40, 581.61, 0., 0., 1. ]
distortion_coeffs: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ -0.024, 0.187, -0.003, -0.001, -0.239 ]
reprojection_error: 3.485
@endcode

-   **camera_matrix**: Contains the focal lengths ($f_x, f_y$) and the principal point ($c_x, c_y$).
-   **distortion_coeffs**: Parameters that describe how the lens bends light (radial and tangential distortion).
-   **reprojection_error**: A measure of calibration quality (lower is better). Values below 1.0 are usually considered excellent, though this depends on image resolution and sensor quality.

Tips for Better Calibration
---------------------------

-   **Board Coverage**: Ensure the board is seen in different parts of the image, especially the corners.
-   **Vary Orientations**: Capture the board with different tilts and rotations.
-   **Focus**: Make sure the board is in sharp focus in all images.
-   **Number of Images**: Typically, 10 to 20 good images are sufficient for a reliable calibration.
