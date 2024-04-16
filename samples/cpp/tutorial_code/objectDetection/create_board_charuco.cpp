#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <iostream>
#include "aruco_samples_utility.hpp"

using namespace cv;

namespace {
const char* about = "Create a ChArUco board image";
//! [charuco_detect_board_keys]
const char* keys  =
        "{@outfile |res.png| Output image }"
        "{w        |  5    | Number of squares in X direction }"
        "{h        |  7    | Number of squares in Y direction }"
        "{sl       |  100  | Square side length (in pixels) }"
        "{ml       |  60   | Marker side length (in pixels) }"
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{cd       |       | Input file with custom dictionary }"
        "{m        |       | Margins size (in pixels). Default is (squareLength-markerLength) }"
        "{bb       | 1     | Number of bits in marker borders }"
        "{si       | false | show generated image }";
}
//! [charuco_detect_board_keys]


int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (argc == 1) {
        parser.printMessage();
    }

    int squaresX = parser.get<int>("w");
    int squaresY = parser.get<int>("h");
    int squareLength = parser.get<int>("sl");
    int markerLength = parser.get<int>("ml");
    int margins = squareLength - markerLength;
    if(parser.has("m")) {
        margins = parser.get<int>("m");
    }

    int borderBits = parser.get<int>("bb");
    bool showImage = parser.get<bool>("si");

    std::string pathOutImg = parser.get<std::string>(0);

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    //! [create_charucoBoard]
    aruco::Dictionary dictionary = readDictionatyFromCommandLine(parser);
    cv::aruco::CharucoBoard board(Size(squaresX, squaresY), (float)squareLength, (float)markerLength, dictionary);
    //! [create_charucoBoard]

    // show created board
    //! [generate_charucoBoard]
    Mat boardImage;
    Size imageSize;
    imageSize.width = squaresX * squareLength + 2 * margins;
    imageSize.height = squaresY * squareLength + 2 * margins;
    board.generateImage(imageSize, boardImage, margins, borderBits);
    //! [generate_charucoBoard]

    if(showImage) {
        imshow("board", boardImage);
        waitKey(0);
    }

    if (pathOutImg != "")
        imwrite(pathOutImg, boardImage);
    return 0;
}
