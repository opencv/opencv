// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "parametersController.hpp"
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include <iostream>

template <typename T>
static bool readFromNode(cv::FileNode node, T& value)
{
    if(!node.isNone()) {
        node >> value;
        return true;
    }
    else
        return false;
}

static bool checkAssertion(bool value, const std::string& msg)
{
    if(!value)
        std::cerr << "Error: " << msg << std::endl;

    return value;
}

bool calib::parametersController::loadFromFile(const std::string &inputFileName)
{
    cv::FileStorage reader;
    reader.open(inputFileName, cv::FileStorage::READ);

    if(!reader.isOpened()) {
        std::cerr << "Warning: Unable to open " << inputFileName <<
                     " Application started with default advanced parameters" << std::endl;
        return true;
    }

    if (readFromNode(reader["charuco_square_lenght"], mCapParams.charucoSquareLength)) {
        std::cout << "DEPRECATION: Parameter 'charuco_square_lenght' has been deprecated (typo). Use 'charuco_square_length' instead." << std::endl;
    }
    readFromNode(reader["charuco_square_length"], mCapParams.charucoSquareLength);
    readFromNode(reader["charuco_marker_size"], mCapParams.charucoMarkerSize);
    readFromNode(reader["camera_resolution"], mCapParams.cameraResolution);
    readFromNode(reader["calibration_step"], mCapParams.calibrationStep);
    readFromNode(reader["max_frames_num"], mCapParams.maxFramesNum);
    readFromNode(reader["min_frames_num"], mCapParams.minFramesNum);
    readFromNode(reader["solver_eps"], mInternalParameters.solverEps);
    readFromNode(reader["solver_max_iters"], mInternalParameters.solverMaxIters);
    readFromNode(reader["fast_solver"], mInternalParameters.fastSolving);
    readFromNode(reader["frame_filter_conv_param"], mInternalParameters.filterAlpha);

    bool retValue =
            checkAssertion(mCapParams.charucoMarkerSize > 0, "Marker size must be positive") &&
            checkAssertion(mCapParams.charucoSquareLength > 0, "Square size must be positive") &&
            checkAssertion(mCapParams.minFramesNum > 1, "Minimal number of frames for calibration < 1") &&
            checkAssertion(mCapParams.calibrationStep > 0, "Calibration step must be positive") &&
            checkAssertion(mCapParams.maxFramesNum > mCapParams.minFramesNum, "maxFramesNum < minFramesNum") &&
            checkAssertion(mInternalParameters.solverEps > 0, "Solver precision must be positive") &&
            checkAssertion(mInternalParameters.solverMaxIters > 0, "Max solver iterations number must be positive") &&
            checkAssertion(mInternalParameters.filterAlpha >=0 && mInternalParameters.filterAlpha <=1 ,
                           "Frame filter convolution parameter must be in [0,1] interval") &&
            checkAssertion(mCapParams.cameraResolution.width > 0 && mCapParams.cameraResolution.height > 0,
                           "Wrong camera resolution values");

    reader.release();
    return retValue;
}

calib::parametersController::parametersController()
{
}

calib::captureParameters calib::parametersController::getCaptureParameters() const
{
    return mCapParams;
}

calib::internalParameters calib::parametersController::getInternalParameters() const
{
    return mInternalParameters;
}

bool calib::parametersController::loadFromParser(cv::CommandLineParser &parser)
{
    mCapParams.flipVertical = parser.get<bool>("flip");
    mCapParams.captureDelay = parser.get<float>("d");
    mCapParams.squareSize = parser.get<float>("sz");
    mCapParams.templDst = parser.get<float>("dst");
    mCapParams.saveFrames = parser.get<bool>("save_frames");
    mCapParams.zoom = parser.get<float>("zoom");
    mCapParams.forceReopen = parser.get<bool>("force_reopen");

    if(!checkAssertion(mCapParams.squareSize > 0, "Distance between corners or circles must be positive"))
        return false;
    if(!checkAssertion(mCapParams.templDst > 0, "Distance between parts of dual template must be positive"))
        return false;

    if (parser.has("v")) {
        mCapParams.source = File;
        mCapParams.videoFileName = parser.get<std::string>("v");
    }
    else {
        mCapParams.source = Camera;
        mCapParams.camID = parser.get<int>("ci");
    }

    std::string templateType = parser.get<std::string>("t");

    if(templateType.find("symcircles", 0) == 0) {
        mCapParams.board = CirclesGrid;
        mCapParams.boardSizeUnits = cv::Size(4, 11);
    }
    else if(templateType.find("circles", 0) == 0) {
        mCapParams.board = AcirclesGrid;
        mCapParams.boardSizeUnits = cv::Size(4, 11);
    }
    else if(templateType.find("chessboard", 0) == 0) {
        mCapParams.board = Chessboard;
        mCapParams.boardSizeUnits = cv::Size(7, 7);
    }
    else if(templateType.find("dualcircles", 0) == 0) {
        mCapParams.board = DoubleAcirclesGrid;
        mCapParams.boardSizeUnits = cv::Size(4, 11);
    }
    else if(templateType.find("charuco", 0) == 0) {
        mCapParams.board = ChArUco;
        mCapParams.boardSizeUnits = cv::Size(5, 7);
        mCapParams.charucoDictFile = parser.get<std::string>("fad");
        std::string arucoDictName = parser.get<std::string>("ad");

        if (arucoDictName == "DICT_4X4_50") { mCapParams.charucoDictName = cv::aruco::DICT_4X4_50; }
        else if (arucoDictName == "DICT_4X4_100") { mCapParams.charucoDictName = cv::aruco::DICT_4X4_100; }
        else if (arucoDictName == "DICT_4X4_250") { mCapParams.charucoDictName = cv::aruco::DICT_4X4_250; }
        else if (arucoDictName == "DICT_4X4_1000") { mCapParams.charucoDictName = cv::aruco::DICT_4X4_1000; }
        else if (arucoDictName == "DICT_5X5_50") { mCapParams.charucoDictName = cv::aruco::DICT_5X5_50; }
        else if (arucoDictName == "DICT_5X5_100") { mCapParams.charucoDictName = cv::aruco::DICT_5X5_100; }
        else if (arucoDictName == "DICT_5X5_250") { mCapParams.charucoDictName = cv::aruco::DICT_5X5_250; }
        else if (arucoDictName == "DICT_5X5_1000") { mCapParams.charucoDictName = cv::aruco::DICT_5X5_1000; }
        else if (arucoDictName == "DICT_6X6_50") { mCapParams.charucoDictName = cv::aruco::DICT_6X6_50; }
        else if (arucoDictName == "DICT_6X6_100") { mCapParams.charucoDictName = cv::aruco::DICT_6X6_100; }
        else if (arucoDictName == "DICT_6X6_250") { mCapParams.charucoDictName = cv::aruco::DICT_6X6_250; }
        else if (arucoDictName == "DICT_6X6_1000") { mCapParams.charucoDictName = cv::aruco::DICT_6X6_1000; }
        else if (arucoDictName == "DICT_7X7_50") { mCapParams.charucoDictName = cv::aruco::DICT_7X7_50; }
        else if (arucoDictName == "DICT_7X7_100") { mCapParams.charucoDictName = cv::aruco::DICT_7X7_100; }
        else if (arucoDictName == "DICT_7X7_250") { mCapParams.charucoDictName = cv::aruco::DICT_7X7_250; }
        else if (arucoDictName == "DICT_7X7_1000") { mCapParams.charucoDictName = cv::aruco::DICT_7X7_1000; }
        else if (arucoDictName == "DICT_ARUCO_ORIGINAL") { mCapParams.charucoDictName = cv::aruco::DICT_ARUCO_ORIGINAL; }
        else if (arucoDictName == "DICT_APRILTAG_16h5") { mCapParams.charucoDictName = cv::aruco::DICT_APRILTAG_16h5; }
        else if (arucoDictName == "DICT_APRILTAG_25h9") { mCapParams.charucoDictName = cv::aruco::DICT_APRILTAG_25h9; }
        else if (arucoDictName == "DICT_APRILTAG_36h10") { mCapParams.charucoDictName = cv::aruco::DICT_APRILTAG_36h10; }
        else if (arucoDictName == "DICT_APRILTAG_36h11") { mCapParams.charucoDictName = cv::aruco::DICT_APRILTAG_36h11; }
        else {
            std::cout << "incorrect name of aruco dictionary \n";
            return false;
        }
        mCapParams.charucoSquareLength = 200;
        mCapParams.charucoMarkerSize = 100;
    }
    else {
        std::cerr << "Wrong template name\n";
        return false;
    }

    if(parser.has("w") && parser.has("h")) {
        mCapParams.inputBoardSize = cv::Size(parser.get<int>("w"), parser.get<int>("h"));
        //only for chessboard pattern board size given in inner corners
        if (templateType != "chessboard") {
            mCapParams.boardSizeUnits = mCapParams.inputBoardSize;
        }
        else {
            mCapParams.boardSizeInnerCorners = mCapParams.inputBoardSize;
        }
        if(!checkAssertion(mCapParams.inputBoardSize.width > 0 || mCapParams.inputBoardSize.height > 0,
                           "Board size must be positive"))
            return false;
    }

    if(!checkAssertion(parser.get<std::string>("of").find(".xml") > 0,
                       "Wrong output file name: correct format is [name].xml"))
        return false;

    loadFromFile(parser.get<std::string>("pf"));
    return true;
}
