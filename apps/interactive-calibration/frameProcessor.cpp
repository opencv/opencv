// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "frameProcessor.hpp"
#include "rotationConverters.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <string>
#include <limits>

using namespace calib;

#define VIDEO_TEXT_SIZE 4
#define POINT_SIZE 5

static cv::SimpleBlobDetector::Params getDetectorParams()
{
    cv::SimpleBlobDetector::Params detectorParams;

    detectorParams.thresholdStep = 40;
    detectorParams.minThreshold = 20;
    detectorParams.maxThreshold = 500;
    detectorParams.minRepeatability = 2;
    detectorParams.minDistBetweenBlobs = 5;

    detectorParams.filterByColor = true;
    detectorParams.blobColor = 0;

    detectorParams.filterByArea = true;
    detectorParams.minArea = 5;
    detectorParams.maxArea = 5000;

    detectorParams.filterByCircularity = false;
    detectorParams.minCircularity = 0.8f;
    detectorParams.maxCircularity = std::numeric_limits<float>::max();

    detectorParams.filterByInertia = true;
    detectorParams.minInertiaRatio = 0.1f;
    detectorParams.maxInertiaRatio = std::numeric_limits<float>::max();

    detectorParams.filterByConvexity = true;
    detectorParams.minConvexity = 0.8f;
    detectorParams.maxConvexity = std::numeric_limits<float>::max();

    return detectorParams;
}

FrameProcessor::~FrameProcessor()
{

}

bool CalibProcessor::detectAndParseChessboard(const cv::Mat &frame)
{
    int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
    bool isTemplateFound = cv::findChessboardCorners(frame, mBoardSizeInnerCorners, mCurrentImagePoints, chessBoardFlags);

    if (isTemplateFound) {
        cv::Mat viewGray;
        cv::cvtColor(frame, viewGray, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(viewGray, mCurrentImagePoints, cv::Size(11,11),
            cv::Size(-1,-1), cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.1 ));
        cv::drawChessboardCorners(frame, mBoardSizeInnerCorners, cv::Mat(mCurrentImagePoints), isTemplateFound);
        mTemplateLocations.insert(mTemplateLocations.begin(), mCurrentImagePoints[0]);
    }
    return isTemplateFound;
}

bool CalibProcessor::detectAndParseChAruco(const cv::Mat &frame)
{
    cv::Ptr<cv::aruco::Board> board = mCharucoBoard.staticCast<cv::aruco::Board>();

    std::vector<std::vector<cv::Point2f> > corners;
    std::vector<int> ids;
    cv::Mat currentCharucoCorners, currentCharucoIds;
    detector->detectBoard(frame, currentCharucoCorners, currentCharucoIds, corners, ids);
    if(ids.size() > 0) cv::aruco::drawDetectedMarkers(frame, corners);

    if(currentCharucoCorners.total() > 3) {
        float centerX = 0, centerY = 0;
        for (int i = 0; i < currentCharucoCorners.size[0]; i++) {
            centerX += currentCharucoCorners.at<float>(i, 0);
            centerY += currentCharucoCorners.at<float>(i, 1);
        }
        centerX /= currentCharucoCorners.size[0];
        centerY /= currentCharucoCorners.size[0];

        mTemplateLocations.insert(mTemplateLocations.begin(), cv::Point2f(centerX, centerY));
        cv::aruco::drawDetectedCornersCharuco(frame, currentCharucoCorners, currentCharucoIds);
        mCurrentCharucoCorners = currentCharucoCorners;
        mCurrentCharucoIds = currentCharucoIds;
        return true;
    }
    return false;
}

bool CalibProcessor::detectAndParseCircles(const cv::Mat &frame)
{
    bool isTemplateFound = findCirclesGrid(frame, mBoardSizeUnits, mCurrentImagePoints, cv::CALIB_CB_SYMMETRIC_GRID, mBlobDetectorPtr);
    if(isTemplateFound) {
        mTemplateLocations.insert(mTemplateLocations.begin(), mCurrentImagePoints[0]);
        cv::drawChessboardCorners(frame, mBoardSizeUnits, cv::Mat(mCurrentImagePoints), isTemplateFound);
    }
    return isTemplateFound;
}

bool CalibProcessor::detectAndParseACircles(const cv::Mat &frame)
{
    bool isTemplateFound = findCirclesGrid(frame, mBoardSizeUnits, mCurrentImagePoints, cv::CALIB_CB_ASYMMETRIC_GRID, mBlobDetectorPtr);
    if(isTemplateFound) {
        mTemplateLocations.insert(mTemplateLocations.begin(), mCurrentImagePoints[0]);
        cv::drawChessboardCorners(frame, mBoardSizeUnits, cv::Mat(mCurrentImagePoints), isTemplateFound);
    }
    return isTemplateFound;
}

bool CalibProcessor::detectAndParseDualACircles(const cv::Mat &frame)
{
    std::vector<cv::Point2f> blackPointbuf;

    cv::Mat invertedView;
    cv::bitwise_not(frame, invertedView);
    bool isWhiteGridFound = cv::findCirclesGrid(frame, mBoardSizeUnits, mCurrentImagePoints, cv::CALIB_CB_ASYMMETRIC_GRID, mBlobDetectorPtr);
    if(!isWhiteGridFound)
        return false;
    bool isBlackGridFound = cv::findCirclesGrid(invertedView, mBoardSizeUnits, blackPointbuf, cv::CALIB_CB_ASYMMETRIC_GRID, mBlobDetectorPtr);

    if(!isBlackGridFound)
    {
        mCurrentImagePoints.clear();
        return false;
    }
    cv::drawChessboardCorners(frame, mBoardSizeUnits, cv::Mat(mCurrentImagePoints), isWhiteGridFound);
    cv::drawChessboardCorners(frame, mBoardSizeUnits, cv::Mat(blackPointbuf), isBlackGridFound);
    mCurrentImagePoints.insert(mCurrentImagePoints.end(), blackPointbuf.begin(), blackPointbuf.end());
    mTemplateLocations.insert(mTemplateLocations.begin(), mCurrentImagePoints[0]);

    return true;
}

void CalibProcessor::saveFrameData()
{
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    switch(mBoardType)
    {
    case Chessboard:
        objectPoints.reserve(mBoardSizeInnerCorners.height*mBoardSizeInnerCorners.width);
        for( int i = 0; i < mBoardSizeInnerCorners.height; ++i )
            for( int j = 0; j < mBoardSizeInnerCorners.width; ++j )
                objectPoints.push_back(cv::Point3f(j*mSquareSize, i*mSquareSize, 0));
        mCalibData->imagePoints.push_back(mCurrentImagePoints);
        mCalibData->objectPoints.push_back(objectPoints);
        break;
    case ChArUco:
        mCalibData->allCharucoCorners.push_back(mCurrentCharucoCorners);
        mCalibData->allCharucoIds.push_back(mCurrentCharucoIds);

        mCharucoBoard->matchImagePoints(mCurrentCharucoCorners, mCurrentCharucoIds, objectPoints, imagePoints);
        CV_Assert(mCurrentCharucoIds.total() == imagePoints.size());
        mCalibData->imagePoints.push_back(imagePoints);
        mCalibData->objectPoints.push_back(objectPoints);
        break;
    case CirclesGrid:
        objectPoints.reserve(mBoardSizeUnits.height*mBoardSizeUnits.width);
        for( int i = 0; i < mBoardSizeUnits.height; i++ )
            for( int j = 0; j < mBoardSizeUnits.width; j++ )
                objectPoints.push_back(cv::Point3f(j*mSquareSize, i*mSquareSize, 0));
        mCalibData->imagePoints.push_back(mCurrentImagePoints);
        mCalibData->objectPoints.push_back(objectPoints);
        break;
    case AcirclesGrid:
        objectPoints.reserve(mBoardSizeUnits.height*mBoardSizeUnits.width);
        for( int i = 0; i < mBoardSizeUnits.height; i++ )
            for( int j = 0; j < mBoardSizeUnits.width; j++ )
                objectPoints.push_back(cv::Point3f((2*j + i % 2)*mSquareSize, i*mSquareSize, 0));
        mCalibData->imagePoints.push_back(mCurrentImagePoints);
        mCalibData->objectPoints.push_back(objectPoints);
        break;
    case DoubleAcirclesGrid:
    {
        float gridCenterX = (2*((float)mBoardSizeUnits.width - 1) + 1)*mSquareSize + mTemplDist / 2;
        float gridCenterY = (mBoardSizeUnits.height - 1)*mSquareSize / 2;
        objectPoints.reserve(2*mBoardSizeUnits.height*mBoardSizeUnits.width);

        //white part
        for( int i = 0; i < mBoardSizeUnits.height; i++ )
            for( int j = 0; j < mBoardSizeUnits.width; j++ )
                objectPoints.push_back(
                            cv::Point3f(-float((2*j + i % 2)*mSquareSize + mTemplDist +
                                               (2*(mBoardSizeUnits.width - 1) + 1)*mSquareSize - gridCenterX),
                                        -float(i*mSquareSize) - gridCenterY,
                                        0));
        //black part
        for( int i = 0; i < mBoardSizeUnits.height; i++ )
            for( int j = 0; j < mBoardSizeUnits.width; j++ )
                objectPoints.push_back(cv::Point3f(-float((2*j + i % 2)*mSquareSize - gridCenterX),
                                          -float(i*mSquareSize) - gridCenterY, 0));

        mCalibData->imagePoints.push_back(mCurrentImagePoints);
        mCalibData->objectPoints.push_back(objectPoints);
    }
        break;
    }
}

void CalibProcessor::showCaptureMessage(const cv::Mat& frame, const std::string &message)
{
    cv::Point textOrigin(100, 100);
    double textSize = VIDEO_TEXT_SIZE * frame.cols / (double) IMAGE_MAX_WIDTH;
    cv::bitwise_not(frame, frame);
    cv::putText(frame, message, textOrigin, 1, textSize, cv::Scalar(0,0,255), 2, cv::LINE_AA);
    cv::Mat resized;
    if (std::fabs(mZoom - 1.) > 0.001f)
    {
        cv::resize(frame, resized, cv::Size(), mZoom, mZoom);
    }
    else
    {
        resized = frame;
    }
    cv::imshow(mainWindowName, resized);
    cv::waitKey(300);
}

bool CalibProcessor::checkLastFrame()
{
    bool isFrameBad = false;
    cv::Mat tmpCamMatrix;
    const double badAngleThresh = 40;

    if(!mCalibData->cameraMatrix.total()) {
        tmpCamMatrix = cv::Mat::eye(3, 3, CV_64F);
        tmpCamMatrix.at<double>(0,0) = 20000;
        tmpCamMatrix.at<double>(1,1) = 20000;
        tmpCamMatrix.at<double>(0,2) = mCalibData->imageSize.height/2;
        tmpCamMatrix.at<double>(1,2) = mCalibData->imageSize.width/2;
    }
    else
        mCalibData->cameraMatrix.copyTo(tmpCamMatrix);

    cv::Mat r, t, angles;
    cv::solvePnP(mCalibData->objectPoints.back(), mCalibData->imagePoints.back(), tmpCamMatrix, mCalibData->distCoeffs, r, t);
    RodriguesToEuler(r, angles, CALIB_DEGREES);
    if(fabs(angles.at<double>(0)) > badAngleThresh || fabs(angles.at<double>(1)) > badAngleThresh) {
        mCalibData->objectPoints.pop_back();
        mCalibData->imagePoints.pop_back();
        if (mCalibData->allCharucoCorners.size()) {
            mCalibData->allCharucoCorners.pop_back();
            mCalibData->allCharucoIds.pop_back();
        }
        isFrameBad = true;
    }
    return isFrameBad;
}

CalibProcessor::CalibProcessor(cv::Ptr<calibrationData> data, captureParameters &capParams) :
    mCalibData(data), mBoardType(capParams.board), mBoardSizeUnits(capParams.boardSizeUnits),
    mBoardSizeInnerCorners(capParams.boardSizeInnerCorners)
{
    mCapuredFrames = 0;
    mNeededFramesNum = capParams.calibrationStep;
    mDelayBetweenCaptures = static_cast<int>(capParams.captureDelay * capParams.fps);
    mMaxTemplateOffset = std::sqrt(static_cast<float>(mCalibData->imageSize.height * mCalibData->imageSize.height) +
                                   static_cast<float>(mCalibData->imageSize.width * mCalibData->imageSize.width)) / 20.0;
    mSquareSize = capParams.squareSize;
    mTemplDist = capParams.templDst;
    mSaveFrames = capParams.saveFrames;
    mZoom = capParams.zoom;
    cv::aruco::CharucoParameters charucoParameters;
    charucoParameters.tryRefineMarkers = true;

    switch(mBoardType)
    {
    case ChArUco:
        if (capParams.charucoDictFile != "None") {
            std::string filename = capParams.charucoDictFile;
            cv::FileStorage dict_file(filename, cv::FileStorage::Mode::READ);
            cv::FileNode fn(dict_file.root());
            mArucoDictionary.readDictionary(fn);
        }
        else {
            mArucoDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PredefinedDictionaryType(capParams.charucoDictName));
        }
        mCharucoBoard = cv::makePtr<cv::aruco::CharucoBoard>(cv::Size(mBoardSizeUnits.width, mBoardSizeUnits.height), capParams.charucoSquareLength,
                                capParams.charucoMarkerSize, mArucoDictionary);
        detector = cv::makePtr<cv::aruco::CharucoDetector>(cv::aruco::CharucoDetector(*mCharucoBoard, charucoParameters));
        break;
    case CirclesGrid:
    case AcirclesGrid:
        mBlobDetectorPtr = cv::SimpleBlobDetector::create();
        break;
    case DoubleAcirclesGrid:
        mBlobDetectorPtr = cv::SimpleBlobDetector::create(getDetectorParams());
        break;
    case Chessboard:
        break;
    }
}

cv::Mat CalibProcessor::processFrame(const cv::Mat &frame)
{
    cv::Mat frameCopy;
    cv::Mat frameCopyToSave;
    frame.copyTo(frameCopy);
    bool isTemplateFound = false;
    mCurrentImagePoints.clear();

    if(mSaveFrames)
        frame.copyTo(frameCopyToSave);

    switch(mBoardType)
    {
    case Chessboard:
        isTemplateFound = detectAndParseChessboard(frameCopy);
        break;
    case ChArUco:
        isTemplateFound = detectAndParseChAruco(frameCopy);
        break;
    case CirclesGrid:
        isTemplateFound = detectAndParseCircles(frameCopy);
        break;
    case AcirclesGrid:
        isTemplateFound = detectAndParseACircles(frameCopy);
        break;
    case DoubleAcirclesGrid:
        isTemplateFound = detectAndParseDualACircles(frameCopy);
        break;
    }

    if(mTemplateLocations.size() > mDelayBetweenCaptures)
        mTemplateLocations.pop_back();
    if(mTemplateLocations.size() == mDelayBetweenCaptures && isTemplateFound) {
        if(cv::norm(mTemplateLocations.front() - mTemplateLocations.back()) < mMaxTemplateOffset) {
            saveFrameData();
            bool isFrameBad = checkLastFrame();
            if (!isFrameBad) {
                std::string displayMessage = cv::format("Frame # %zu captured", std::max(mCalibData->imagePoints.size(),
                                                                                        mCalibData->allCharucoCorners.size()));
                if(!showOverlayMessage(displayMessage))
                    showCaptureMessage(frame, displayMessage);

                if(mSaveFrames)
                    mCalibData->allFrames.push_back(frameCopyToSave);

                mCapuredFrames++;
            }
            else {
                std::string displayMessage = "Frame rejected";
                if(!showOverlayMessage(displayMessage))
                    showCaptureMessage(frame, displayMessage);
            }
            mTemplateLocations.clear();
            mTemplateLocations.reserve(mDelayBetweenCaptures);
        }
    }

    return frameCopy;
}

bool CalibProcessor::isProcessed() const
{
    if(mCapuredFrames < mNeededFramesNum)
        return false;
    else
        return true;
}

void CalibProcessor::resetState()
{
    mCapuredFrames = 0;
    mTemplateLocations.clear();
}

CalibProcessor::~CalibProcessor()
{

}

////////////////////////////////////////////

void ShowProcessor::drawBoard(cv::Mat &img, cv::InputArray points)
{
    cv::Mat tmpView = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
    std::vector<cv::Point2f> templateHull;
    std::vector<cv::Point> poly;
    cv::convexHull(points, templateHull);
    poly.resize(templateHull.size());
    for(size_t i=0; i<templateHull.size();i++)
        poly[i] = cv::Point((int)(templateHull[i].x*mGridViewScale), (int)(templateHull[i].y*mGridViewScale));
    cv::fillConvexPoly(tmpView, poly, cv::Scalar(0, 255, 0), cv::LINE_AA);
    cv::addWeighted(tmpView, .2, img, 1, 0, img);
}

void ShowProcessor::drawGridPoints(const cv::Mat &frame)
{
    if(mBoardType != ChArUco)
        for(std::vector<std::vector<cv::Point2f> >::iterator it = mCalibdata->imagePoints.begin(); it != mCalibdata->imagePoints.end(); ++it)
            for(std::vector<cv::Point2f>::iterator pointIt = (*it).begin(); pointIt != (*it).end(); ++pointIt)
                cv::circle(frame, *pointIt, POINT_SIZE, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    else
        for(std::vector<cv::Mat>::iterator it = mCalibdata->allCharucoCorners.begin(); it != mCalibdata->allCharucoCorners.end(); ++it)
            for(int i = 0; i < (*it).size[0]; i++)
                cv::circle(frame, cv::Point((int)(*it).at<float>(i, 0), (int)(*it).at<float>(i, 1)),
                           POINT_SIZE, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
}

ShowProcessor::ShowProcessor(cv::Ptr<calibrationData> data, cv::Ptr<calibController> controller, TemplateType board) :
    mCalibdata(data), mController(controller), mBoardType(board)
{
    mNeedUndistort = true;
    mVisMode = Grid;
    mGridViewScale = 0.5;
    mTextSize = VIDEO_TEXT_SIZE;
}

cv::Mat ShowProcessor::processFrame(const cv::Mat &frame)
{
    if (!mCalibdata->cameraMatrix.empty() && !mCalibdata->distCoeffs.empty())
    {
        mTextSize = VIDEO_TEXT_SIZE * (double) frame.cols / IMAGE_MAX_WIDTH;
        cv::Scalar textColor = cv::Scalar(0,0,255);
        cv::Mat frameCopy;

        if (mNeedUndistort && mController->getFramesNumberState()) {
            if(mVisMode == Grid)
                drawGridPoints(frame);
            cv::remap(frame, frameCopy, mCalibdata->undistMap1, mCalibdata->undistMap2, cv::INTER_LINEAR);
            int baseLine = 100;
            cv::Size textSize = cv::getTextSize("Undistorted view", 1, mTextSize, 2, &baseLine);
            cv::Point textOrigin(baseLine, frame.rows - (int)(2.5*textSize.height));
            cv::putText(frameCopy, "Undistorted view", textOrigin, 1, mTextSize, textColor, 2, cv::LINE_AA);
        }
        else {
            frame.copyTo(frameCopy);
            if(mVisMode == Grid)
                drawGridPoints(frameCopy);
        }
        std::string displayMessage;
        if(mCalibdata->stdDeviations.at<double>(0) == 0)
            displayMessage = cv::format("F = %d RMS = %.3f", (int)mCalibdata->cameraMatrix.at<double>(0,0), mCalibdata->totalAvgErr);
        else
            displayMessage = cv::format("Fx = %d Fy = %d RMS = %.3f", (int)mCalibdata->cameraMatrix.at<double>(0,0),
                                            (int)mCalibdata->cameraMatrix.at<double>(1,1), mCalibdata->totalAvgErr);
        if(mController->getRMSState() && mController->getFramesNumberState())
            displayMessage.append(" OK");

        int baseLine = 100;
        cv::Size textSize = cv::getTextSize(displayMessage, 1, mTextSize - 1, 2, &baseLine);
        cv::Point textOrigin = cv::Point(baseLine, 2*textSize.height);
        cv::putText(frameCopy, displayMessage, textOrigin, 1, mTextSize - 1, textColor, 2, cv::LINE_AA);

        if(mCalibdata->stdDeviations.at<double>(0) == 0)
            displayMessage = cv::format("DF = %.2f", mCalibdata->stdDeviations.at<double>(1)*sigmaMult);
        else
            displayMessage = cv::format("DFx = %.2f DFy = %.2f", mCalibdata->stdDeviations.at<double>(0)*sigmaMult,
                                                    mCalibdata->stdDeviations.at<double>(1)*sigmaMult);
        if(mController->getConfidenceIntrervalsState() && mController->getFramesNumberState())
            displayMessage.append(" OK");
        cv::putText(frameCopy, displayMessage, cv::Point(baseLine, 4*textSize.height), 1, mTextSize - 1, textColor, 2, cv::LINE_AA);

        if(mController->getCommonCalibrationState()) {
            displayMessage = cv::format("Calibration is done");
            cv::putText(frameCopy, displayMessage, cv::Point(baseLine, 6*textSize.height), 1, mTextSize - 1, textColor, 2, cv::LINE_AA);
        }
        int calibFlags = mController->getNewFlags();
        displayMessage = "";
        if(!(calibFlags & cv::CALIB_FIX_ASPECT_RATIO))
            displayMessage.append(cv::format("AR=%.3f ", mCalibdata->cameraMatrix.at<double>(0,0)/mCalibdata->cameraMatrix.at<double>(1,1)));
        if(calibFlags & cv::CALIB_ZERO_TANGENT_DIST)
            displayMessage.append("TD=0 ");
        displayMessage.append(cv::format("K1=%.2f K2=%.2f K3=%.2f", mCalibdata->distCoeffs.at<double>(0), mCalibdata->distCoeffs.at<double>(1),
                                         mCalibdata->distCoeffs.at<double>(4)));
        cv::putText(frameCopy, displayMessage, cv::Point(baseLine, frameCopy.rows - (int)(1.5*textSize.height)),
                    1, mTextSize - 1, textColor, 2, cv::LINE_AA);
        return frameCopy;
    }

    return frame;
}

bool ShowProcessor::isProcessed() const
{
    return false;
}

void ShowProcessor::resetState()
{

}

void ShowProcessor::setVisualizationMode(visualisationMode mode)
{
    mVisMode = mode;
}

void ShowProcessor::switchVisualizationMode()
{
    if(mVisMode == Grid) {
        mVisMode = Window;
        updateBoardsView();
    }
    else {
        mVisMode = Grid;
        cv::destroyWindow(gridWindowName);
    }
}

void ShowProcessor::clearBoardsView()
{
    cv::imshow(gridWindowName, cv::Mat());
}

void ShowProcessor::updateBoardsView()
{
    if(mVisMode == Window) {
        cv::Size originSize = mCalibdata->imageSize;
        cv::Mat altGridView = cv::Mat::zeros((int)(originSize.height*mGridViewScale), (int)(originSize.width*mGridViewScale), CV_8UC3);
        if(mBoardType != ChArUco)
            for(std::vector<std::vector<cv::Point2f> >::iterator it = mCalibdata->imagePoints.begin(); it != mCalibdata->imagePoints.end(); ++it)
                if(mBoardType != DoubleAcirclesGrid)
                    drawBoard(altGridView, *it);
                else {
                    size_t pointsNum = (*it).size()/2;
                    std::vector<cv::Point2f> points(pointsNum);
                    std::copy((*it).begin(), (*it).begin() + pointsNum, points.begin());
                    drawBoard(altGridView, points);
                    std::copy((*it).begin() + pointsNum, (*it).begin() + 2*pointsNum, points.begin());
                    drawBoard(altGridView, points);
                }
        else
            for(std::vector<cv::Mat>::iterator it = mCalibdata->allCharucoCorners.begin(); it != mCalibdata->allCharucoCorners.end(); ++it)
                drawBoard(altGridView, *it);
        cv::imshow(gridWindowName, altGridView);
    }
}

void ShowProcessor::switchUndistort()
{
    mNeedUndistort = !mNeedUndistort;
}

void ShowProcessor::setUndistort(bool isEnabled)
{
    mNeedUndistort = isEnabled;
}

ShowProcessor::~ShowProcessor()
{

}
