#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <vector>
#include <iostream>
#include "aruco_samples_utility.hpp"

using namespace std;
using namespace cv;

namespace {
const char* about = "Create a ChArUco marker image";
const char* keys  =
        "{@outfile |   res.png | Output image }"
        "{sl       |   100     | Square side length (in pixels) }"
        "{ml       |   60      | Marker side length (in pixels) }"
        "{cd       |           | Input file with custom dictionary }"
        "{d        |   10      | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{ids      |0, 1, 2, 3 | Four ids for the ChArUco marker: id1,id2,id3,id4 }"
        "{m        |   0       | Margins size (in pixels) }"
        "{bb       |   1       | Number of bits in marker borders }"
        "{si       |   false   | show generated image }";
}

int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    int squareLength = parser.get<int>("sl");
    int markerLength = parser.get<int>("ml");
    string idsString = parser.get<string>("ids");
    int margins = parser.get<int>("m");
    int borderBits = parser.get<int>("bb");
    bool showImage = parser.get<bool>("si");
    string out = parser.get<string>(0);
    aruco::Dictionary dictionary = readDictionatyFromCommandLine(parser);

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    istringstream ss(idsString);
    vector<string> splittedIds;
    string token;
    while(getline(ss, token, ','))
        splittedIds.push_back(token);
    if(splittedIds.size() < 4) {
        throw std::runtime_error("Incorrect ids format\n");
    }
    Vec4i ids;
    for(int i = 0; i < 4; i++)
        ids[i] = atoi(splittedIds[i].c_str());

    //! [generate_diamond]
    vector<int> diamondIds = {ids[0], ids[1], ids[2], ids[3]};
    aruco::CharucoBoard charucoBoard(Size(3, 3), (float)squareLength, (float)markerLength, dictionary, diamondIds);
    Mat markerImg;
    charucoBoard.generateImage(Size(3*squareLength + 2*margins, 3*squareLength + 2*margins), markerImg, margins, borderBits);
    //! [generate_diamond]

    if(showImage) {
        imshow("board", markerImg);
        waitKey(0);
    }

    if (out != "")
        imwrite(out, markerImg);
    return 0;
}
