/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#include "opencv2/videoio/registry.hpp"
#include "videoio_registry.hpp"

namespace cv {

void DefaultDeleter<CvCapture>::operator ()(CvCapture* obj) const { cvReleaseCapture(&obj); }
void DefaultDeleter<CvVideoWriter>::operator ()(CvVideoWriter* obj) const { cvReleaseVideoWriter(&obj); }


VideoCapture::VideoCapture()
{}

VideoCapture::VideoCapture(const String& filename, int apiPreference)
{
    CV_TRACE_FUNCTION();
    open(filename, apiPreference);
}

VideoCapture::VideoCapture(int index, int apiPreference)
{
    CV_TRACE_FUNCTION();
    open(index, apiPreference);
}

VideoCapture::~VideoCapture()
{
    CV_TRACE_FUNCTION();

    icap.release();
    cap.release();
}

bool VideoCapture::open(const String& filename, int apiPreference)
{
    CV_TRACE_FUNCTION();

    if (isOpened()) release();

    const std::vector<VideoBackendInfo> backends = cv::videoio_registry::getAvailableBackends_CaptureByFilename();
    for (size_t i = 0; i < backends.size(); i++)
    {
        const VideoBackendInfo& info = backends[i];
        if (apiPreference == CAP_ANY || apiPreference == info.id)
        {
            CvCapture* capture = NULL;
            VideoCapture_create(capture, icap, info.id, filename);
            if (!icap.empty())
            {
                if (icap->isOpened())
                    return true;
                icap.release();
            }
            if (capture)
            {
                cap.reset(capture);
                // assume it is opened
                return true;
            }
        }
    }
    return false;
}

bool  VideoCapture::open(int cameraNum, int apiPreference)
{
    CV_TRACE_FUNCTION();

    if (isOpened()) release();

    if (apiPreference == CAP_ANY)
    {
        // interpret preferred interface (0 = autodetect)
        int backendID = (cameraNum / 100) * 100;
        if (backendID)
        {
            cameraNum %= 100;
            apiPreference = backendID;
        }
    }

    const std::vector<VideoBackendInfo> backends = cv::videoio_registry::getAvailableBackends_CaptureByIndex();
    for (size_t i = 0; i < backends.size(); i++)
    {
        const VideoBackendInfo& info = backends[i];
        if (apiPreference == CAP_ANY || apiPreference == info.id)
        {
            CvCapture* capture = NULL;
            VideoCapture_create(capture, icap, info.id, cameraNum);
            if (!icap.empty())
            {
                if (icap->isOpened())
                    return true;
                icap.release();
            }
            if (capture)
            {
                cap.reset(capture);
                // assume it is opened
                return true;
            }
        }
    }
    return false;
}

bool VideoCapture::isOpened() const
{
    if (!icap.empty())
        return icap->isOpened();
    return !cap.empty();  // legacy interface doesn't support closed files
}

String VideoCapture::getBackendName() const
{
    int api = 0;
    if (icap)
        api = icap->isOpened() ? icap->getCaptureDomain() : 0;
    else if (cap)
        api = cap->getCaptureDomain();
    CV_Assert(api != 0);
    return cv::videoio_registry::getBackendName((VideoCaptureAPIs)api);
}

void VideoCapture::release()
{
    CV_TRACE_FUNCTION();
    icap.release();
    cap.release();
}

bool VideoCapture::grab()
{
    CV_INSTRUMENT_REGION();

    if (!icap.empty())
        return icap->grabFrame();
    return cvGrabFrame(cap) != 0;
}

bool VideoCapture::retrieve(OutputArray image, int channel)
{
    CV_INSTRUMENT_REGION();

    if (!icap.empty())
        return icap->retrieveFrame(channel, image);

    IplImage* _img = cvRetrieveFrame(cap, channel);
    if( !_img )
    {
        image.release();
        return false;
    }
    if(_img->origin == IPL_ORIGIN_TL)
        cv::cvarrToMat(_img).copyTo(image);
    else
    {
        Mat temp = cv::cvarrToMat(_img);
        flip(temp, image, 0);
    }
    return true;
}

bool VideoCapture::read(OutputArray image)
{
    CV_INSTRUMENT_REGION();

    if(grab())
        retrieve(image);
    else
        image.release();
    return !image.empty();
}

VideoCapture& VideoCapture::operator >> (Mat& image)
{
#ifdef WINRT_VIDEO
    // FIXIT grab/retrieve methods() should work too
    if (grab())
    {
        if (retrieve(image))
        {
            std::lock_guard<std::mutex> lock(VideoioBridge::getInstance().inputBufferMutex);
            VideoioBridge& bridge = VideoioBridge::getInstance();

            // double buffering
            bridge.swapInputBuffers();
            auto p = bridge.frontInputPtr;

            bridge.bIsFrameNew = false;

            // needed here because setting Mat 'image' is not allowed by OutputArray in read()
            Mat m(bridge.getHeight(), bridge.getWidth(), CV_8UC3, p);
            image = m;
        }
    }
#else
    read(image);
#endif
    return *this;
}

VideoCapture& VideoCapture::operator >> (UMat& image)
{
    CV_INSTRUMENT_REGION();

    read(image);
    return *this;
}

bool VideoCapture::set(int propId, double value)
{
    CV_CheckNE(propId, (int)CAP_PROP_BACKEND, "Can't set read-only property");

    if (!icap.empty())
        return icap->setProperty(propId, value);
    return cvSetCaptureProperty(cap, propId, value) != 0;
}

double VideoCapture::get(int propId) const
{
    if (propId == CAP_PROP_BACKEND)
    {
        int api = 0;
        if (icap)
            api = icap->isOpened() ? icap->getCaptureDomain() : 0;
        else if (cap)
            api = cap->getCaptureDomain();
        if (api <= 0)
            return -1.0;
        return (double)api;
    }
    if (!icap.empty())
        return icap->getProperty(propId);
    return cap ? cap->getProperty(propId) : 0;
}


//=================================================================================================



VideoWriter::VideoWriter()
{}

VideoWriter::VideoWriter(const String& filename, int _fourcc, double fps, Size frameSize, bool isColor)
{
    open(filename, _fourcc, fps, frameSize, isColor);
}


VideoWriter::VideoWriter(const String& filename, int apiPreference, int _fourcc, double fps, Size frameSize, bool isColor)
{
    open(filename, apiPreference, _fourcc, fps, frameSize, isColor);
}

void VideoWriter::release()
{
    iwriter.release();
    writer.release();
}

VideoWriter::~VideoWriter()
{
    release();
}

bool VideoWriter::open(const String& filename, int _fourcc, double fps, Size frameSize, bool isColor)
{
    return open(filename, CAP_ANY, _fourcc, fps, frameSize, isColor);
}

bool VideoWriter::open(const String& filename, int apiPreference, int _fourcc, double fps, Size frameSize, bool isColor)
{
    CV_INSTRUMENT_REGION();

    if (isOpened()) release();

    const std::vector<VideoBackendInfo> backends = cv::videoio_registry::getAvailableBackends_Writer();
    for (size_t i = 0; i < backends.size(); i++)
    {
        const VideoBackendInfo& info = backends[i];
        if (apiPreference == CAP_ANY || apiPreference == info.id)
        {
            CvVideoWriter* writer_ = NULL;
            VideoWriter_create(writer_, iwriter, info.id, filename, _fourcc, fps, frameSize, isColor);
            if (!iwriter.empty())
            {
                if (iwriter->isOpened())
                    return true;
                iwriter.release();
            }
            if (writer_)
            {
                // assume it is opened
                writer.reset(writer_);
                return true;
            }
        }
    }
    return false;
}

bool VideoWriter::isOpened() const
{
    return !iwriter.empty() || !writer.empty();
}


bool VideoWriter::set(int propId, double value)
{
    CV_CheckNE(propId, (int)CAP_PROP_BACKEND, "Can't set read-only property");

    if (!iwriter.empty())
        return iwriter->setProperty(propId, value);
    return false;
}

double VideoWriter::get(int propId) const
{
    if (propId == CAP_PROP_BACKEND)
    {
        int api = 0;
        if (iwriter)
            api = iwriter->getCaptureDomain();
        else if (writer)
            api = writer->getCaptureDomain();
        if (api <= 0)
            return -1.0;
        return (double)api;
    }
    if (!iwriter.empty())
        return iwriter->getProperty(propId);
    return 0.;
}

String VideoWriter::getBackendName() const
{
    int api = 0;
    if (iwriter)
        api = iwriter->getCaptureDomain();
    else if (writer)
        api = writer->getCaptureDomain();
    CV_Assert(api != 0);
    return cv::videoio_registry::getBackendName((VideoCaptureAPIs)api);
}

void VideoWriter::write(InputArray image)
{
    CV_INSTRUMENT_REGION();

    if( iwriter )
        iwriter->write(image);
    else
    {
        IplImage _img = cvIplImage(image.getMat());
        cvWriteFrame(writer, &_img);
    }
}

VideoWriter& VideoWriter::operator << (const Mat& image)
{
    CV_INSTRUMENT_REGION();

    write(image);
    return *this;
}

VideoWriter& VideoWriter::operator << (const UMat& image)
{
    CV_INSTRUMENT_REGION();
    write(image);
    return *this;
}

// FIXIT OpenCV 4.0: make inline
int VideoWriter::fourcc(char c1, char c2, char c3, char c4)
{
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24);
}

} // namespace
