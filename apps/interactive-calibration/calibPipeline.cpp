#include "calibPipeline.hpp"
#include <opencv2/highgui.hpp>
#include <stdexcept>

using namespace calib;

#define CAP_DELAY 10

cv::Size CalibPipeline::getCameraResolution()
{
    mCapture.set(cv::CAP_PROP_FRAME_WIDTH, 10000);
    mCapture.set(cv::CAP_PROP_FRAME_HEIGHT, 10000);
    int w = (int)mCapture.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)mCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    return cv::Size(w,h);
}

CalibPipeline::CalibPipeline(captureParameters params) :
    mCaptureParams(params)
{

}

PipelineExitStatus CalibPipeline::start(std::vector<cv::Ptr<FrameProcessor> > processors)
{
    if(mCaptureParams.source == Camera && !mCapture.isOpened())
    {
        mCapture.open(mCaptureParams.camID);
        cv::Size maxRes = getCameraResolution();
        cv::Size neededRes = mCaptureParams.cameraResolution;

        if(maxRes.width < neededRes.width) {
            double aR = (double)maxRes.width / maxRes.height;
            mCapture.set(cv::CAP_PROP_FRAME_WIDTH, neededRes.width);
            mCapture.set(cv::CAP_PROP_FRAME_HEIGHT, neededRes.width/aR);
        }
        else if(maxRes.height < neededRes.height) {
            double aR = (double)maxRes.width / maxRes.height;
            mCapture.set(cv::CAP_PROP_FRAME_HEIGHT, neededRes.height);
            mCapture.set(cv::CAP_PROP_FRAME_WIDTH, neededRes.height*aR);
        }
        else {
            mCapture.set(cv::CAP_PROP_FRAME_HEIGHT, neededRes.height);
            mCapture.set(cv::CAP_PROP_FRAME_WIDTH, neededRes.width);
        }
        mCapture.set(cv::CAP_PROP_AUTOFOCUS, 0);
    }
    else if (mCaptureParams.source == File && !mCapture.isOpened())
        mCapture.open(mCaptureParams.videoFileName);
    mImageSize = cv::Size((int)mCapture.get(cv::CAP_PROP_FRAME_WIDTH), (int)mCapture.get(cv::CAP_PROP_FRAME_HEIGHT));

    if(!mCapture.isOpened())
        throw std::runtime_error("Unable to open video source");

    cv::Mat frame, processedFrame;
    while(mCapture.grab()) {
        mCapture.retrieve(frame);
        if(mCaptureParams.flipVertical)
            cv::flip(frame, frame, -1);

        frame.copyTo(processedFrame);
        for (std::vector<cv::Ptr<FrameProcessor> >::iterator it = processors.begin(); it != processors.end(); ++it)
            processedFrame = (*it)->processFrame(processedFrame);
        cv::imshow(mainWindowName, processedFrame);
        int key = cv::waitKey(CAP_DELAY);

        if(key == 27) // esc
            return Finished;
        else if (key == 114) // r
            return DeleteLastFrame;
        else if (key == 100) // d
            return DeleteAllFrames;
        else if (key == 115) // s
            return SaveCurrentData;
        else if (key == 117) // u
            return SwitchUndistort;
        else if (key == 118) // v
            return SwitchVisualisation;

        for (std::vector<cv::Ptr<FrameProcessor> >::iterator it = processors.begin(); it != processors.end(); ++it)
            if((*it)->isProcessed())
                return Calibrate;
    }

    return Finished;
}

cv::Size CalibPipeline::getImageSize() const
{
    return mImageSize;
}
