// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "parametersController.hpp"

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
                     " Applicatioin stated with default advanced parameters" << std::endl;
        return true;
    }

    readFromNode(reader["charuco_dict"], mCapParams.charucoDictName);
    readFromNode(reader["charuco_square_lenght"], mCapParams.charucoSquareLenght);
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
            checkAssertion(mCapParams.charucoDictName >= 0, "Dict name must be >= 0") &&
            checkAssertion(mCapParams.charucoMarkerSize > 0, "Marker size must be positive") &&
            checkAssertion(mCapParams.charucoSquareLenght > 0, "Square size must be positive") &&
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

    if(templateType.find("circles", 0) == 0) {
        mCapParams.board = AcirclesGrid;
        mCapParams.boardSize = cv::Size(4, 11);
    }
    else if(templateType.find("chessboard", 0) == 0) {
        mCapParams.board = Chessboard;
        mCapParams.boardSize = cv::Size(7, 7);
    }
    else if(templateType.find("dualcircles", 0) == 0) {
        mCapParams.board = DoubleAcirclesGrid;
        mCapParams.boardSize = cv::Size(4, 11);
    }
    else if(templateType.find("charuco", 0) == 0) {
        mCapParams.board = chAruco;
        mCapParams.boardSize = cv::Size(6, 8);
        mCapParams.charucoDictName = 0;
        mCapParams.charucoSquareLenght = 200;
        mCapParams.charucoMarkerSize = 100;
    }
    else {
        std::cerr << "Wrong template name\n";
        return false;
    }

    if(parser.has("w") && parser.has("h")) {
        mCapParams.boardSize = cv::Size(parser.get<int>("w"), parser.get<int>("h"));
        if(!checkAssertion(mCapParams.boardSize.width > 0 || mCapParams.boardSize.height > 0,
                           "Board size must be positive"))
            return false;
    }

    if(!checkAssertion(parser.get<std::string>("of").find(".xml") > 0,
                       "Wrong output file name: correct format is [name].xml"))
        return false;

    loadFromFile(parser.get<std::string>("pf"));
    return true;
}
