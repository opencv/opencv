// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#ifdef HAVE_LIBREALSENSE
#include "cap_librealsense.hpp"

namespace cv
{

VideoCapture_LibRealsense::VideoCapture_LibRealsense(int index)
{
    try
    {
        mDev = mContext.get_device(index);
        // Configure all streams to run at VGA resolution at 60 frames per second
        mDev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 60);
        mDev->enable_stream(rs::stream::color, 640, 480, rs::format::bgr8, 60);
        mDev->enable_stream(rs::stream::infrared, 640, 480, rs::format::y8, 60);
        mDev->start();
    }
    catch (rs::error&)
    {
        mDev = nullptr;
    }
}
VideoCapture_LibRealsense::~VideoCapture_LibRealsense(){}

double VideoCapture_LibRealsense::getProperty(int prop) const
{
    double propValue = 0;

    if(prop == CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE)
        return mDev->get_depth_scale();

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
        mDev->wait_for_frames();
    }
    catch (rs::error&)
    {
        return false;
    }

    return true;
}
bool VideoCapture_LibRealsense::retrieveFrame(int outputType, cv::OutputArray frame)
{
    rs::stream stream;
    int type;
    switch (outputType)
    {
    case CAP_INTELPERC_DEPTH_MAP:
        stream = rs::stream::depth_aligned_to_color;
        type = CV_16UC1;
        break;
    case CAP_INTELPERC_IR_MAP:
        stream = rs::stream::infrared;
        type = CV_8UC1;
        break;
    case CAP_INTELPERC_IMAGE:
        stream = rs::stream::color;
        type = CV_8UC3;
        break;
    default:
        return false;
    }

    try
    {
        // we copy the data straight away, so const_cast should be fine
        void* data = const_cast<void*>(mDev->get_frame_data(stream));
        Mat(mDev->get_stream_height(stream), mDev->get_stream_width(stream), type, data).copyTo(frame);
    }
    catch (rs::error&)
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
    return mDev && mDev->is_streaming();
}

}

#endif
