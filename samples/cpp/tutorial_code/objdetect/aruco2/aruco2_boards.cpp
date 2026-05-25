#include <opencv2/objdetect/aruco2.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main()
{
    //! [create_board]
    Mat boardImage;
    aruco2::getGridBoardImage(boardImage, Size(9, 5), aruco2::DICT_ARUCO_MIP_36h12);
    imwrite("board.png", boardImage);
    //! [create_board]

    std::cout << "Board image size: " << boardImage.cols << "x" << boardImage.rows << std::endl;

    // Place board on a white scene for detection
    Mat scene(boardImage.rows + 100, boardImage.cols + 100, CV_8UC1, Scalar(255));
    boardImage.copyTo(scene(Rect(50, 50, boardImage.cols, boardImage.rows)));

    //! [detect_board]
    Mat image = scene.clone();
    aruco2::GridBoard board;
    bool found = aruco2::detectGridBoard(image, Size(9, 5), aruco2::DICT_ARUCO_MIP_36h12, board);

    if (found) {
        std::cout << "Detected " << board.markers.size() << " markers on the board." << std::endl;
    }
    //! [detect_board]

    //! [draw_board]
    Mat colorImage;
    cvtColor(image, colorImage, COLOR_GRAY2BGR);
    aruco2::drawGridBoard(colorImage, board);
    imshow("Detected Board", colorImage);
    waitKey(0);
    //! [draw_board]

    return 0;
}
