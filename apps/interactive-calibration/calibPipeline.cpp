// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "calibPipeline.hpp"

#include <opencv2/highgui.hpp>
//#include <opencv2/core.hpp> // Not needed
//#include <opencv2/opencv.hpp> // Not needed
#include <opencv2/imgproc.hpp>
#include <iostream>             // For output

#include <stdexcept>

using namespace calib;
using namespace cv;

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

PipelineExitStatus CalibPipeline::start(std::vector<cv::Ptr<FrameProcessor>> processors)
{
    std::cout << "\n\t\t Starting calib Pipeline... \n" << std::endl;

    /*
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
    */
    // NOTE: Instead of grabbing pictures from device, lets override the pipeline to some local directory
    //       Maybe it should be better to move this to a different sourceFile to actually be a merge candidate fork

    // FIXME : Use some filename prefix instead
    // static std::string name[ 26] = {"mSetA_00.jpg", "mSetA_01.jpg","mSetA_02.jpg", "mSetA_03.jpg","mSetA_04.jpg", "mSetA_05.jpg","mSetA_06.jpg", "mSetA_07.jpg",      "mSetA_08.jpg", "mSetA_09.jpg","mSetA_10.jpg", "mSetA_11.jpg","mSetA_12.jpg", "mSetA_13.jpg","mSetA_14.jpg", "mSetA_15.jpg", "mSetA_16.jpg", "mSetA_17.jpg","mSetA_18.jpg", "mSetA_19.jpg","mSetA_20.jpg", "mSetA_21.jpg","mSetA_22.jpg", "mSetA_23.jpg" , "mSetA_24.jpg", "mSetA_25.jpg" };
    static std::string name[26] = {"mSetB_00.jpg", "mSetB_01.jpg", "mSetB_02.jpg", "mSetB_03.jpg", "mSetB_04.jpg", "mSetB_05.jpg", "mSetB_06.jpg", "mSetB_07.jpg", "mSetB_08.jpg", "mSetB_09.jpg", "mSetB_10.jpg", "mSetB_11.jpg", "mSetB_12.jpg", "mSetB_13.jpg", "mSetB_14.jpg", "mSetB_15.jpg", "mSetB_16.jpg", "mSetB_17.jpg", "mSetB_18.jpg", "mSetB_19.jpg", "mSetB_20.jpg", "mSetB_21.jpg", "mSetB_22.jpg", "mSetB_23.jpg", "mSetB_24.jpg", "mSetB_25.jpg"};
    static cv::Mat frame, processedFrame;

    static int lastFrame = -1;
    static int frameCount = 0;

    while (frameCount < 42)
    {

        if (frameCount != lastFrame)
        {
            std::cout << "Trying to open: " << name[frameCount] << "  | ";
            //image = cv::imread( name[0] , CV_LOAD_IMAGE_GRAYSCALE );
            frame = cv::imread(name[frameCount], 1); // Read the file
            lastFrame = frameCount;

            mImageSize = cv::Size(frame.cols, frame.rows);

            if (!frame.data)
            { // Check for invalid input
                std::cout << "Could not open or find the image" << std::endl;
                return Finished;
            }
        }
        // namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
        // imshow( name[i] , frame );                        // Show our image inside it.
        // cv::waitKey(0);                                   // Wait for a keystroke in the window

        frame.copyTo(processedFrame);
        for (std::vector<cv::Ptr<FrameProcessor> >::iterator it = processors.begin(); it != processors.end(); ++it)
            processedFrame = (*it)->processFrame(processedFrame);

        // to half size or even smaller
        // resize(processedFrame, processedFrame, Size(processedFrame.cols/3.5, processedFrame.rows/3.5)) ;

        cv::imshow(mainWindowName, processedFrame);
        char key = (char)cv::waitKey(CAP_DELAY);

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

        // Use key for changing Frame
        else if (key == 120) // x
        {
            frameCount++;
            std::cout << "    NEXT FRAME .... \n ";
        }

        for (std::vector<cv::Ptr<FrameProcessor> >::iterator it = processors.begin(); it != processors.end(); ++it)
            if((*it)->isProcessed())
                return Calibrate;
    }
    //return SaveCurrentData; // What was this for?
    return Finished;
}

cv::Size CalibPipeline::getImageSize() const
{
    return mImageSize;
}
