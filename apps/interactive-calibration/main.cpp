// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio/registry.hpp>

#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>

#include "calibCommon.hpp"
#include "calibPipeline.hpp"
#include "frameProcessor.hpp"
#include "calibController.hpp"
#include "parametersController.hpp"
#include "rotationConverters.hpp"

using namespace calib;

static std::string getVideoIoBackendsString()
{
    std::string result;
    auto backs = cv::videoio_registry::getBackends();
    for (const auto& b: backs)
    {
        if (!result.empty())
            result += ", ";

        result += cv::videoio_registry::getBackendName(b);
    }

    return result;
}

const char* keys  =
        "{v        |         | Input from video file }"
        "{ci       | 0       | Camera id }"
        "{vb       |         | Video I/O back-end. One of: %s }"
        "{flip     | false   | Vertical flip of input frames }"
        "{t        | circles | Template for calibration (circles, chessboard, dualCircles, charuco, symcircles) }"
        "{sz       | 16.3    | Distance between two nearest centers of circles or squares on calibration board}"
        "{dst      | 295     | Distance between white and black parts of daulCircles template}"
        "{w        |         | Width of template (in corners or circles)}"
        "{h        |         | Height of template (in corners or circles)}"
        "{ad       | DICT_4X4_50 | Name of predefined ArUco dictionary. Available ArUco dictionaries: "
        "DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, "
        "DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, DICT_7X7_100, "
        "DICT_7X7_250, DICT_7X7_1000, DICT_ARUCO_ORIGINAL, DICT_APRILTAG_16h5, DICT_APRILTAG_25h9, "
        "DICT_APRILTAG_36h10, DICT_APRILTAG_36h11 }"
        "{fad      | None    | name of file with ArUco dictionary}"
        "{of       | cameraParameters.xml | Output file name}"
        "{ft       | true    | Auto tuning of calibration flags}"
        "{vis      | grid    | Captured boards visualisation (grid, window)}"
        "{d        | 0.8     | Min delay between captures}"
        "{pf       | defaultConfig.xml| Advanced application parameters}"
        "{save_frames | false   | Save frames that contribute to final calibration}"
        "{zoom     | 1       | Zoom factor applied to the preview image}"
        "{force_reopen | false   | Forcefully reopen camera in case of errors}"
        "{help     |         | Print help}";

bool calib::showOverlayMessage(const std::string& message)
{
#ifdef HAVE_QT
    cv::displayOverlay(mainWindowName, message, OVERLAY_DELAY);
    return true;
#else
    std::cout << message << std::endl;
    return false;
#endif
}

static void deleteButton(int, void* data)
{
    (static_cast<cv::Ptr<calibDataController>*>(data))->get()->deleteLastFrame();
    calib::showOverlayMessage("Last frame deleted");
}

static void deleteAllButton(int, void* data)
{
    (static_cast<cv::Ptr<calibDataController>*>(data))->get()->deleteAllData();
    calib::showOverlayMessage("All frames deleted");
}

static void saveCurrentParamsButton(int, void* data)
{
    if((static_cast<cv::Ptr<calibDataController>*>(data))->get()->saveCurrentCameraParameters())
        calib::showOverlayMessage("Calibration parameters saved");
}

#ifdef HAVE_QT
static void switchVisualizationModeButton(int, void* data)
{
    ShowProcessor* processor = static_cast<ShowProcessor*>(((cv::Ptr<FrameProcessor>*)data)->get());
    processor->switchVisualizationMode();
}

static void undistortButton(int state, void* data)
{
    ShowProcessor* processor = static_cast<ShowProcessor*>(((cv::Ptr<FrameProcessor>*)data)->get());
    processor->setUndistort(static_cast<bool>(state));
    calib::showOverlayMessage(std::string("Undistort is ") +
                       (static_cast<bool>(state) ? std::string("on") : std::string("off")));
}
#endif //HAVE_QT

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, cv::format(keys, getVideoIoBackendsString().c_str()));

    if(parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::cout << consoleHelp << std::endl;
    parametersController paramsController;

    if(!paramsController.loadFromParser(parser))
        return 0;

    captureParameters capParams = paramsController.getCaptureParameters();
    internalParameters intParams = paramsController.getInternalParameters();

    cv::TermCriteria solverTermCrit = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                                       intParams.solverMaxIters, intParams.solverEps);
    cv::Ptr<calibrationData> globalData(new calibrationData);
    if(!parser.has("v")) globalData->imageSize = capParams.cameraResolution;

    int calibrationFlags = 0;
    if(intParams.fastSolving) calibrationFlags |= cv::CALIB_USE_QR;
    if(intParams.rationalModel) calibrationFlags |= cv::CALIB_RATIONAL_MODEL;
    if(intParams.thinPrismModel) calibrationFlags |= cv::CALIB_THIN_PRISM_MODEL;
    if(intParams.tiltedModel) calibrationFlags |= cv::CALIB_TILTED_MODEL;

    cv::Ptr<calibController> controller(new calibController(globalData, calibrationFlags,
                                                         parser.get<bool>("ft"), capParams.minFramesNum));
    cv::Ptr<calibDataController> dataController(new calibDataController(globalData, capParams.maxFramesNum,
                                                                     intParams.filterAlpha));
    dataController->setParametersFileName(parser.get<std::string>("of"));

    cv::Ptr<FrameProcessor> capProcessor, showProcessor;

    capProcessor = cv::Ptr<FrameProcessor>(new CalibProcessor(globalData, capParams));
    showProcessor = cv::Ptr<FrameProcessor>(new ShowProcessor(globalData, controller, capParams.board));

    if(parser.get<std::string>("vis").find("window") == 0) {
        static_cast<ShowProcessor*>(showProcessor.get())->setVisualizationMode(Window);
        cv::namedWindow(gridWindowName);
        cv::moveWindow(gridWindowName, 1280, 500);
    }

    cv::Ptr<CalibPipeline> pipeline(new CalibPipeline(capParams));
    std::vector<cv::Ptr<FrameProcessor> > processors;
    processors.push_back(capProcessor);
    processors.push_back(showProcessor);

    cv::namedWindow(mainWindowName);
    cv::moveWindow(mainWindowName, 10, 10);
#ifdef HAVE_QT
    cv::createButton("Delete last frame", deleteButton, &dataController,
                     cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Delete all frames", deleteAllButton, &dataController,
                     cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Undistort", undistortButton, &showProcessor,
                     cv::QT_CHECKBOX | cv::QT_NEW_BUTTONBAR, false);
    cv::createButton("Save current parameters", saveCurrentParamsButton, &dataController,
                     cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Switch visualisation mode", switchVisualizationModeButton, &showProcessor,
                     cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
#endif //HAVE_QT
    try {
        bool pipelineFinished = false;
        while(!pipelineFinished)
        {
            PipelineExitStatus exitStatus = pipeline->start(processors);
            if (exitStatus == Finished) {
                if(controller->getCommonCalibrationState())
                    saveCurrentParamsButton(0, &dataController);
                pipelineFinished = true;
                continue;
            }
            else if (exitStatus == Calibrate) {

                dataController->rememberCurrentParameters();
                globalData->imageSize = pipeline->getImageSize();
                calibrationFlags = controller->getNewFlags();

                globalData->totalAvgErr =
                        cv::calibrateCamera(globalData->objectPoints, globalData->imagePoints,
                                            globalData->imageSize, globalData->cameraMatrix,
                                            globalData->distCoeffs, cv::noArray(), cv::noArray(),
                                            globalData->stdDeviations, cv::noArray(), globalData->perViewErrors,
                                            calibrationFlags, solverTermCrit);
                dataController->updateUndistortMap();
                dataController->printParametersToConsole(std::cout);
                controller->updateState();
                for(int j = 0; j < capParams.calibrationStep; j++)
                    dataController->filterFrames();
                static_cast<ShowProcessor*>(showProcessor.get())->updateBoardsView();
            }
            else if (exitStatus == DeleteLastFrame) {
                deleteButton(0, &dataController);
                static_cast<ShowProcessor*>(showProcessor.get())->updateBoardsView();
            }
            else if (exitStatus == DeleteAllFrames) {
                deleteAllButton(0, &dataController);
                static_cast<ShowProcessor*>(showProcessor.get())->updateBoardsView();
            }
            else if (exitStatus == SaveCurrentData) {
                saveCurrentParamsButton(0, &dataController);
            }
            else if (exitStatus == SwitchUndistort)
                static_cast<ShowProcessor*>(showProcessor.get())->switchUndistort();
            else if (exitStatus == SwitchVisualisation)
                static_cast<ShowProcessor*>(showProcessor.get())->switchVisualizationMode();

            for (std::vector<cv::Ptr<FrameProcessor> >::iterator it = processors.begin(); it != processors.end(); ++it)
                (*it)->resetState();
        }
    }
    catch (const std::runtime_error& exp) {
        std::cout << exp.what() << std::endl;
    }

    return 0;
}
