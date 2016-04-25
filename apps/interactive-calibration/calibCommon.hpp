#ifndef CALIB_COMMON_HPP
#define CALIB_COMMON_HPP

#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace calib
{
    #define OVERLAY_DELAY 1000
    #define IMAGE_MAX_WIDTH 1280
    #define IMAGE_MAX_HEIGHT 960

    bool showOverlayMessage(const std::string& message);

    enum InputType { Video, Pictures };
    enum InputVideoSource { Camera, File };
    enum TemplateType { AcirclesGrid, Chessboard, chAruco, DoubleAcirclesGrid };

    static const std::string mainWindowName = "Calibration";
    static const std::string gridWindowName = "Board locations";
    static const std::string consoleHelp = "Hot keys:\nesc - exit application\n"
                              "s - save current data to .xml file\n"
                              "r - delete last frame\n"
                              "u - enable/disable applying undistortion"
                              "d - delete all frames\n"
                              "v - switch visualization";

    static const double sigmaMult = 1.96;

    struct calibrationData
    {
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
        cv::Mat stdDeviations;
        cv::Mat perViewErrors;
        std::vector<cv::Mat> rvecs;
        std::vector<cv::Mat> tvecs;
        double totalAvgErr;
        cv::Size imageSize;

        std::vector<std::vector<cv::Point2f> > imagePoints;
        std::vector< std::vector<cv::Point3f> > objectPoints;

        std::vector<cv::Mat> allCharucoCorners;
        std::vector<cv::Mat> allCharucoIds;

        cv::Mat undistMap1, undistMap2;

        calibrationData()
        {
            imageSize = cv::Size(IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT);
        }
    };

    struct cameraParameters
    {
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
        cv::Mat stdDeviations;
        double avgError;

        cameraParameters(){}
        cameraParameters(cv::Mat& _cameraMatrix, cv::Mat& _distCoeffs, cv::Mat& _stdDeviations, double _avgError = 0) :
            cameraMatrix(_cameraMatrix), distCoeffs(_distCoeffs), stdDeviations(_stdDeviations), avgError(_avgError)
        {}
    };

    struct captureParameters
    {
        InputType captureMethod;
        InputVideoSource source;
        TemplateType board;
        cv::Size boardSize;
        int charucoDictName;
        int calibrationStep;
        float charucoSquareLenght, charucoMarkerSize;
        float captureDelay;
        float squareSize;
        float templDst;
        std::string videoFileName;
        bool flipVertical;
        int camID;
        int fps;
        cv::Size cameraResolution;
        int maxFramesNum;
        int minFramesNum;

        captureParameters()
        {
            calibrationStep = 1;
            captureDelay = 500.f;
            maxFramesNum = 30;
            minFramesNum = 10;
            fps = 30;
            cameraResolution = cv::Size(IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT);
        }
    };

    struct internalParameters
    {
        double solverEps;
        int solverMaxIters;
        bool fastSolving;
        double filterAlpha;

        internalParameters()
        {
            solverEps = 1e-7;
            solverMaxIters = 30;
            fastSolving = false;
            filterAlpha = 0.1;
        }
    };
}

#endif
