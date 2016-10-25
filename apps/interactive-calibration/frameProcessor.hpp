// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef FRAME_PROCESSOR_HPP
#define FRAME_PROCESSOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>

#include "calibCommon.hpp"
#include "calibController.hpp"

namespace calib
{
class FrameProcessor
{
protected:

public:
    virtual ~FrameProcessor();
    virtual cv::Mat processFrame(const cv::Mat& frame) = 0;
    virtual bool isProcessed() const = 0;
    virtual void resetState() = 0;
};

class CalibProcessor : public FrameProcessor
{
protected:
    cv::Ptr<calibrationData> mCalibData;
    TemplateType mBoardType;
    cv::Size mBoardSize;
    std::vector<cv::Point2f> mTemplateLocations;
    std::vector<cv::Point2f> mCurrentImagePoints;
    cv::Mat mCurrentCharucoCorners;
    cv::Mat mCurrentCharucoIds;

    cv::Ptr<cv::SimpleBlobDetector> mBlobDetectorPtr;
    cv::Ptr<cv::aruco::Dictionary> mArucoDictionary;
    cv::Ptr<cv::aruco::CharucoBoard> mCharucoBoard;

    int mNeededFramesNum;
    unsigned mDelayBetweenCaptures;
    int mCapuredFrames;
    double mMaxTemplateOffset;
    float mSquareSize;
    float mTemplDist;

    bool detectAndParseChessboard(const cv::Mat& frame);
    bool detectAndParseChAruco(const cv::Mat& frame);
    bool detectAndParseACircles(const cv::Mat& frame);
    bool detectAndParseDualACircles(const cv::Mat& frame);
    void saveFrameData();
    void showCaptureMessage(const cv::Mat &frame, const std::string& message);
    bool checkLastFrame();

public:
    CalibProcessor(cv::Ptr<calibrationData> data, captureParameters& capParams);
    virtual cv::Mat processFrame(const cv::Mat& frame);
    virtual bool isProcessed() const;
    virtual void resetState();
    ~CalibProcessor();
};

enum visualisationMode {Grid, Window};

class ShowProcessor : public FrameProcessor
{
protected:
    cv::Ptr<calibrationData> mCalibdata;
    cv::Ptr<calibController> mController;
    TemplateType mBoardType;
    visualisationMode mVisMode;
    bool mNeedUndistort;
    double mGridViewScale;
    double mTextSize;

    void drawBoard(cv::Mat& img, cv::InputArray points);
    void drawGridPoints(const cv::Mat& frame);
public:
    ShowProcessor(cv::Ptr<calibrationData> data, cv::Ptr<calibController> controller, TemplateType board);
    virtual cv::Mat processFrame(const cv::Mat& frame);
    virtual bool isProcessed() const;
    virtual void resetState();

    void setVisualizationMode(visualisationMode mode);
    void switchVisualizationMode();
    void clearBoardsView();
    void updateBoardsView();

    void switchUndistort();
    void setUndistort(bool isEnabled);
    ~ShowProcessor();
};

}


#endif
