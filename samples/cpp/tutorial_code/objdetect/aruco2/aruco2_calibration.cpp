#include <opencv2/objdetect/aruco2.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

using namespace cv;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: aruco2_calibration <images_folder>" << std::endl;
        return 1;
    }

    //! [calibration]
    float markerSize = 0.05f; // 5 cm
    std::string folder = argv[1];

    std::vector<std::vector<Point3f>> allObjPts;
    std::vector<std::vector<Point2f>> allImgPts;
    Size imageSize;

    for (auto& entry : fs::directory_iterator(folder)) {
        Mat image = imread(entry.path().string(), IMREAD_GRAYSCALE);
        if (image.empty()) continue;
        if (imageSize.width == 0) imageSize = image.size();

        aruco2::GridBoard board;
        if (aruco2::detectGridBoard(image, Size(9, 5), aruco2::DICT_ARUCO_MIP_36h12, board)) {
            std::vector<Point2f> imgPts; std::vector<Point3f> objPts;
            aruco2::getSolvePnpPoints(board, objPts, imgPts, markerSize);
            allImgPts.push_back(imgPts);
            allObjPts.push_back(objPts);
        }
    }

    Mat cameraMatrix, distCoeffs;
    std::vector<Mat> rvecs, tvecs;
    double rpe = calibrateCamera(allObjPts, allImgPts, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

    std::cout << "Reprojection error: " << rpe << " px" << std::endl;
    std::cout << "Camera Matrix:\n" << cameraMatrix << std::endl;

    FileStorage fsOut("calibration.yaml", FileStorage::WRITE);
    fsOut << "camera_matrix" << cameraMatrix;
    fsOut << "distortion_coeffs" << distCoeffs;
    fsOut.release();
    //! [calibration]

    return 0;
}
