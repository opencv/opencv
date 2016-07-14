#ifndef CALIB_PIPELINE_HPP
#define CALIB_PIPELINE_HPP

#include <vector>
#include <opencv2/highgui.hpp>

#include "calibCommon.hpp"
#include "frameProcessor.hpp"

namespace calib
{

enum PipelineExitStatus { Finished,
                                DeleteLastFrame,
                                Calibrate,
                                DeleteAllFrames,
                                SaveCurrentData,
                                SwitchUndistort,
                                SwitchVisualisation
                              };

class CalibPipeline
{
protected:
    captureParameters mCaptureParams;
    cv::Size mImageSize;
    cv::VideoCapture mCapture;

    cv::Size getCameraResolution();

public:
    CalibPipeline(captureParameters params);
    PipelineExitStatus start(std::vector<cv::Ptr<FrameProcessor> > processors);
    cv::Size getImageSize() const;
};

}

#endif
