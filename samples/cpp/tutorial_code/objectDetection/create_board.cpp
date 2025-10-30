#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <iostream>
#include "aruco_samples_utility.hpp"

using namespace cv;

namespace {
const char* about = "Create an ArUco grid board image";
const char* keys  =
        "{@outfile |<none> | Output image }"
        "{w        |       | Number of markers in X direction }"
        "{h        |       | Number of markers in Y direction }"
        "{l        |       | Marker side length (in pixels) }"
        "{s        |       | Separation between two consecutive markers in the grid (in pixels)}"
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{cd       |       | Input file with custom dictionary }"
        "{m        |       | Margins size (in pixels). Default is marker separation (-s) }"
        "{bb       | 1     | Number of bits in marker borders }"
        "{si       | false | show generated image }";
}

int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 7) {
        parser.printMessage();
        return 0;
    }

    int markersX = parser.get<int>("w");
    int markersY = parser.get<int>("h");
    int markerLength = parser.get<int>("l");
    int markerSeparation = parser.get<int>("s");
    int margins = markerSeparation;
    if(parser.has("m")) {
        margins = parser.get<int>("m");
    }

    int borderBits = parser.get<int>("bb");
    bool showImage = parser.get<bool>("si");

    String out = parser.get<String>(0);

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    Size imageSize;
    imageSize.width = markersX * (markerLength + markerSeparation) - markerSeparation + 2 * margins;
    imageSize.height =
        markersY * (markerLength + markerSeparation) - markerSeparation + 2 * margins;

    aruco::Dictionary dictionary = readDictionatyFromCommandLine(parser);
    aruco::GridBoard board(Size(markersX, markersY), float(markerLength), float(markerSeparation), dictionary);

    // show created board
    //! [aruco_generate_board_image]
    Mat boardImage;
    board.generateImage(imageSize, boardImage, margins, borderBits);
    //! [aruco_generate_board_image]

    if(showImage) {
        imshow("board", boardImage);
        waitKey(0);
    }

    imwrite(out, boardImage);
    return 0;
}
