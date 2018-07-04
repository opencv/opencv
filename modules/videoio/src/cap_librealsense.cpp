// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#ifdef HAVE_LIBREALSENSE
#include "cap_librealsense.hpp"

namespace cv
{

VideoCapture_LibRealsense::VideoCapture_LibRealsense(int) : mAlign(RS2_STREAM_COLOR)
{
    try
    {
        rs2::config config;
        // Configure all streams to run at VGA resolution at default fps
        config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16);
        config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8);
        config.enable_stream(RS2_STREAM_INFRARED, 640, 480, RS2_FORMAT_Y8);
        mPipe.start();
    }
    catch (const rs2::error&)
    {
    }
}
VideoCapture_LibRealsense::~VideoCapture_LibRealsense(){}

double VideoCapture_LibRealsense::getProperty(int prop) const
{
    double propValue = 0;

    if (prop == CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE)
        return mPipe.get_active_profile().get_device().first<rs2::depth_sensor>().get_depth_scale();

    return propValue;
}
bool VideoCapture_LibRealsense::setProperty(int, double)
{
    bool isSet = false;
    return isSet;
}

bool VideoCapture_LibRealsense::grabFrame()
{
    if (!isOpened())
        return false;

    try
    {
        mData = mAlign.process(mPipe.wait_for_frames());
    }
    catch (const rs2::error&)
    {
        return false;
    }

    return true;
}
bool VideoCapture_LibRealsense::retrieveFrame(int outputType, cv::OutputArray frame)
{
    rs2::video_frame _frame(nullptr);
    int type;
    switch (outputType)
    {
    case CAP_INTELPERC_DEPTH_MAP:
        _frame = mData.get_depth_frame().as<rs2::video_frame>();
        type = CV_16UC1;
        break;
    case CAP_INTELPERC_IR_MAP:
        _frame = mData.get_infrared_frame();
        type = CV_8UC1;
        break;
    case CAP_INTELPERC_IMAGE:
        _frame = mData.get_color_frame();
        type = CV_8UC3;
        break;
    default:
        return false;
    }

    try
    {
        // we copy the data straight away, so const_cast should be fine
        void* data = const_cast<void*>(_frame.get_data());
        Mat(_frame.get_height(), _frame.get_width(), type, data, _frame.get_stride_in_bytes()).copyTo(frame);

        if(_frame.get_profile().format() == RS2_FORMAT_RGB8)
            cvtColor(frame, frame, COLOR_RGB2BGR);
    }
    catch (const rs2::error&)
    {
        return false;
    }

    return true;
}
int VideoCapture_LibRealsense::getCaptureDomain()
{
    return CAP_INTELPERC;
}

bool VideoCapture_LibRealsense::isOpened() const
{
    return bool(std::shared_ptr<rs2_pipeline>(mPipe));
}

}

#endif
