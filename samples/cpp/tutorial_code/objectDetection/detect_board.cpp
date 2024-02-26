#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include "aruco_samples_utility.hpp"

using namespace std;
using namespace cv;

namespace {
const char* about = "Pose estimation using a ArUco Planar Grid board";

//! [aruco_detect_board_keys]
const char* keys  =
        "{w        |       | Number of squares in X direction }"
        "{h        |       | Number of squares in Y direction }"
        "{l        |       | Marker side length (in pixels) }"
        "{s        |       | Separation between two consecutive markers in the grid (in pixels)}"
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{cd       |       | Input file with custom dictionary }"
        "{c        |       | Output file with calibrated camera parameters }"
        "{v        |       | Input from video or image file, if omitted, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{dp       |       | File of marker detector parameters }"
        "{rs       |       | Apply refind strategy }"
        "{r        |       | show rejected candidates too }";
}
//! [aruco_detect_board_keys]

static void readDetectorParamsFromCommandLine(CommandLineParser &parser, aruco::DetectorParameters& detectorParams) {
    if(parser.has("dp")) {
        FileStorage fs(parser.get<string>("dp"), FileStorage::READ);
        bool readOk = detectorParams.readDetectorParameters(fs.root());
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            throw -1;
        }
    }
}

static void readCameraParamsFromCommandLine(CommandLineParser &parser, Mat& camMatrix, Mat& distCoeffs) {
    if(parser.has("c")) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            throw -1;
        }
    }
}

static void readDictionatyFromCommandLine(CommandLineParser &parser, aruco::Dictionary& dictionary) {
    if (parser.has("d")) {
        int dictionaryId = parser.get<int>("d");
        dictionary = aruco::getPredefinedDictionary(aruco::PredefinedDictionaryType(dictionaryId));
    }
    else if (parser.has("cd")) {
        FileStorage fs(parser.get<string>("cd"), FileStorage::READ);
        bool readOk = dictionary.readDictionary(fs.root());
        if(!readOk) {
            cerr << "Invalid dictionary file" << endl;
            throw -1;
        }
    }
    else {
        cerr << "Dictionary not specified" << endl;
        throw -1;
    }
}

int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 7) {
        parser.printMessage();
        return 0;
    }

    //! [aruco_detect_board_full_sample]
    int markersX = parser.get<int>("w");
    int markersY = parser.get<int>("h");
    float markerLength = parser.get<float>("l");
    float markerSeparation = parser.get<float>("s");
    bool showRejected = parser.has("r");
    bool refindStrategy = parser.has("rs");
    int camId = parser.get<int>("ci");


    Mat camMatrix, distCoeffs;
    readCameraParamsFromCommandLine(parser, camMatrix, distCoeffs);

    aruco::DetectorParameters detectorParams;
    detectorParams.cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX; // do corner refinement in markers
    readDetectorParamsFromCommandLine(parser, detectorParams);

    String video;
    if(parser.has("v")) {
        video = parser.get<String>("v");
    }

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    readDictionatyFromCommandLine(parser, dictionary);

    aruco::ArucoDetector detector(dictionary, detectorParams);
    VideoCapture inputVideo;
    int waitTime;
    if(!video.empty()) {
        inputVideo.open(video);
        waitTime = 0;
    } else {
        inputVideo.open(camId);
        waitTime = 10;
    }

    float axisLength = 0.5f * ((float)min(markersX, markersY) * (markerLength + markerSeparation) +
                               markerSeparation);

    // Create GridBoard object
    //! [aruco_create_board]
    aruco::GridBoard board(Size(markersX, markersY), markerLength, markerSeparation, dictionary);
    //! [aruco_create_board]

    // Also you could create Board object
    //vector<vector<Point3f> > objPoints; // array of object points of all the marker corners in the board
    //vector<int> ids; // vector of the identifiers of the markers in the board
    //aruco::Board board(objPoints, dictionary, ids);

    double totalTime = 0;
    int totalIterations = 0;

    while(inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        double tick = (double)getTickCount();

        vector<int> ids;
        vector<vector<Point2f>> corners, rejected;
        Vec3d rvec, tvec;

        //! [aruco_detect_and_refine]

        // Detect markers
        detector.detectMarkers(image, corners, ids, rejected);

        // Refind strategy to detect more markers
        if(refindStrategy)
            detector.refineDetectedMarkers(image, board, corners, ids, rejected, camMatrix,
                                           distCoeffs);

        //! [aruco_detect_and_refine]

        // Estimate board pose
        int markersOfBoardDetected = 0;
        if(!ids.empty()) {
            // Get object and image points for the solvePnP function
            cv::Mat objPoints, imgPoints;
            board.matchImagePoints(corners, ids, objPoints, imgPoints);

            // Find pose
            cv::solvePnP(objPoints, imgPoints, camMatrix, distCoeffs, rvec, tvec);

            markersOfBoardDetected = (int)objPoints.total() / 4;
        }

        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations % 30 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
        }

        // Draw results
        image.copyTo(imageCopy);
        if(!ids.empty()) {
            aruco::drawDetectedMarkers(imageCopy, corners, ids);
        }

        if(showRejected && !rejected.empty())
            aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));

        if(markersOfBoardDetected > 0)
            cv::drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvec, tvec, axisLength);

        imshow("out", imageCopy);
        char key = (char)waitKey(waitTime);
        if(key == 27) break;
    //! [aruco_detect_board_full_sample]
    }

    return 0;
}
