#include <opencv2/objdetect/aruco2.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main()
{
    //! [create_raruco]
    Mat image;
    aruco2::DictionaryType dict = aruco2::DICT_APRILTAG_16h5;
    aruco2::getRArucoMarkerImage(image, dict, 0, 2, 5, 2, true);
    imwrite("raruco_marker.png", image);
    //! [create_raruco]

    std::cout << "RArUco marker image size: " << image.cols << "x" << image.rows << std::endl;

    //! [detect_raruco]
    auto markers = aruco2::detectRArucoMarkers(image, dict);

    for (const auto &m : markers) {
        std::cout << "Detected RArUco marker ID: " << m.id << " with " << m.corners.size() << " corners." << std::endl;
    }
    //! [detect_raruco]

    //! [draw_raruco]
    Mat colorImage;
    cvtColor(image, colorImage, COLOR_GRAY2BGR);
    aruco2::drawFiducialMarkers(colorImage, markers);
    imshow("Detected RArUco Markers", colorImage);
    waitKey(0);
    //! [draw_raruco]

    return 0;
}
