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

#if !defined(HAVE_FFMPEG)
#error "Build configuration error"
#endif

#include <string>

#include "cap_ffmpeg_impl.hpp"

#define icvCreateFileCapture_FFMPEG_p cvCreateFileCapture_FFMPEG
#define icvReleaseCapture_FFMPEG_p cvReleaseCapture_FFMPEG
#define icvGrabFrame_FFMPEG_p cvGrabFrame_FFMPEG
#define icvRetrieveFrame_FFMPEG_p cvRetrieveFrame_FFMPEG
#define icvSetCaptureProperty_FFMPEG_p cvSetCaptureProperty_FFMPEG
#define icvGetCaptureProperty_FFMPEG_p cvGetCaptureProperty_FFMPEG
#define icvCreateVideoWriter_FFMPEG_p cvCreateVideoWriter_FFMPEG
#define icvReleaseVideoWriter_FFMPEG_p cvReleaseVideoWriter_FFMPEG
#define icvWriteFrame_FFMPEG_p cvWriteFrame_FFMPEG


namespace cv {
namespace {

class CvCapture_FFMPEG_proxy CV_FINAL : public cv::IVideoCapture
{
public:
    CvCapture_FFMPEG_proxy() { ffmpegCapture = 0; }
    CvCapture_FFMPEG_proxy(const cv::String& filename) { ffmpegCapture = 0; open(filename); }
    virtual ~CvCapture_FFMPEG_proxy() { close(); }

    virtual double getProperty(int propId) const CV_OVERRIDE
    {
        return ffmpegCapture ? icvGetCaptureProperty_FFMPEG_p(ffmpegCapture, propId) : 0;
    }
    virtual bool setProperty(int propId, double value) CV_OVERRIDE
    {
        return ffmpegCapture ? icvSetCaptureProperty_FFMPEG_p(ffmpegCapture, propId, value)!=0 : false;
    }
    virtual bool grabFrame() CV_OVERRIDE
    {
        return ffmpegCapture ? icvGrabFrame_FFMPEG_p(ffmpegCapture)!=0 : false;
    }
    virtual bool retrieveFrame(int, cv::OutputArray frame) CV_OVERRIDE
    {
        unsigned char* data = 0;
        int step=0, width=0, height=0, cn=0;

        if (!ffmpegCapture ||
           !icvRetrieveFrame_FFMPEG_p(ffmpegCapture, &data, &step, &width, &height, &cn))
            return false;
        cv::Mat(height, width, CV_MAKETYPE(CV_8U, cn), data, step).copyTo(frame);
        return true;
    }
    virtual bool open( const cv::String& filename )
    {
        close();

        ffmpegCapture = icvCreateFileCapture_FFMPEG_p( filename.c_str() );
        return ffmpegCapture != 0;
    }
    virtual void close()
    {
        if (ffmpegCapture)
            icvReleaseCapture_FFMPEG_p( &ffmpegCapture );
        CV_Assert(ffmpegCapture == 0);
        ffmpegCapture = 0;
    }

    virtual bool isOpened() const CV_OVERRIDE { return ffmpegCapture != 0; }
    virtual int getCaptureDomain() CV_OVERRIDE { return CV_CAP_FFMPEG; }

protected:
    CvCapture_FFMPEG* ffmpegCapture;
};

} // namespace

cv::Ptr<cv::IVideoCapture> cvCreateFileCapture_FFMPEG_proxy(const std::string &filename)
{
    cv::Ptr<CvCapture_FFMPEG_proxy> capture = cv::makePtr<CvCapture_FFMPEG_proxy>(filename);
    if (capture && capture->isOpened())
        return capture;
    return cv::Ptr<cv::IVideoCapture>();
}

namespace {

class CvVideoWriter_FFMPEG_proxy CV_FINAL :
    public cv::IVideoWriter
{
public:
    CvVideoWriter_FFMPEG_proxy() { ffmpegWriter = 0; }
    CvVideoWriter_FFMPEG_proxy(const cv::String& filename, int fourcc, double fps, cv::Size frameSize, bool isColor) { ffmpegWriter = 0; open(filename, fourcc, fps, frameSize, isColor); }
    virtual ~CvVideoWriter_FFMPEG_proxy() { close(); }

    int getCaptureDomain() const CV_OVERRIDE { return cv::CAP_FFMPEG; }

    virtual void write(cv::InputArray image ) CV_OVERRIDE
    {
        if(!ffmpegWriter)
            return;
        CV_Assert(image.depth() == CV_8U);

        icvWriteFrame_FFMPEG_p(ffmpegWriter, (const uchar*)image.getMat().ptr(), (int)image.step(), image.cols(), image.rows(), image.channels(), 0);
    }
    virtual bool open( const cv::String& filename, int fourcc, double fps, cv::Size frameSize, bool isColor )
    {
        close();
        ffmpegWriter = icvCreateVideoWriter_FFMPEG_p( filename.c_str(), fourcc, fps, frameSize.width, frameSize.height, isColor );
        return ffmpegWriter != 0;
    }

    virtual void close()
    {
        if (ffmpegWriter)
            icvReleaseVideoWriter_FFMPEG_p( &ffmpegWriter );
        CV_Assert(ffmpegWriter == 0);
        ffmpegWriter = 0;
    }

    virtual double getProperty(int) const CV_OVERRIDE { return 0; }
    virtual bool setProperty(int, double) CV_OVERRIDE { return false; }
    virtual bool isOpened() const CV_OVERRIDE { return ffmpegWriter != 0; }

protected:
    CvVideoWriter_FFMPEG* ffmpegWriter;
};

} // namespace

cv::Ptr<cv::IVideoWriter> cvCreateVideoWriter_FFMPEG_proxy(const std::string& filename, int fourcc,
                                                           double fps, const cv::Size& frameSize,
                                                           const VideoWriterParameters& params)
{
    const bool isColor = params.get(VIDEOWRITER_PROP_IS_COLOR, true);
    cv::Ptr<CvVideoWriter_FFMPEG_proxy> writer = cv::makePtr<CvVideoWriter_FFMPEG_proxy>(filename, fourcc, fps, frameSize, isColor);
    if (writer && writer->isOpened())
        return writer;
    return cv::Ptr<cv::IVideoWriter>();
}

} // namespace



//==================================================================================================

#if defined(BUILD_PLUGIN)

#include "plugin_api.hpp"

namespace cv {

static
CvResult CV_API_CALL cv_capture_open(const char* filename, int camera_index, CV_OUT CvPluginCapture* handle)
{
    if (!handle)
        return CV_ERROR_FAIL;
    *handle = NULL;
    if (!filename)
        return CV_ERROR_FAIL;
    CV_UNUSED(camera_index);
    CvCapture_FFMPEG_proxy *cap = 0;
    try
    {
        cap = new CvCapture_FFMPEG_proxy(filename);
        if (cap->isOpened())
        {
            *handle = (CvPluginCapture)cap;
            return CV_ERROR_OK;
        }
    }
    catch (...)
    {
    }
    if (cap)
        delete cap;
    return CV_ERROR_FAIL;
}

static
CvResult CV_API_CALL cv_capture_release(CvPluginCapture handle)
{
    if (!handle)
        return CV_ERROR_FAIL;
    CvCapture_FFMPEG_proxy* instance = (CvCapture_FFMPEG_proxy*)handle;
    delete instance;
    return CV_ERROR_OK;
}


static
CvResult CV_API_CALL cv_capture_get_prop(CvPluginCapture handle, int prop, CV_OUT double* val)
{
    if (!handle)
        return CV_ERROR_FAIL;
    if (!val)
        return CV_ERROR_FAIL;
    try
    {
        CvCapture_FFMPEG_proxy* instance = (CvCapture_FFMPEG_proxy*)handle;
        *val = instance->getProperty(prop);
        return CV_ERROR_OK;
    }
    catch (...)
    {
        return CV_ERROR_FAIL;
    }
}

static
CvResult CV_API_CALL cv_capture_set_prop(CvPluginCapture handle, int prop, double val)
{
    if (!handle)
        return CV_ERROR_FAIL;
    try
    {
        CvCapture_FFMPEG_proxy* instance = (CvCapture_FFMPEG_proxy*)handle;
        return instance->setProperty(prop, val) ? CV_ERROR_OK : CV_ERROR_FAIL;
    }
    catch(...)
    {
        return CV_ERROR_FAIL;
    }
}

static
CvResult CV_API_CALL cv_capture_grab(CvPluginCapture handle)
{
    if (!handle)
        return CV_ERROR_FAIL;
    try
    {
        CvCapture_FFMPEG_proxy* instance = (CvCapture_FFMPEG_proxy*)handle;
        return instance->grabFrame() ? CV_ERROR_OK : CV_ERROR_FAIL;
    }
    catch(...)
    {
        return CV_ERROR_FAIL;
    }
}

static
CvResult CV_API_CALL cv_capture_retrieve(CvPluginCapture handle, int stream_idx, cv_videoio_retrieve_cb_t callback, void* userdata)
{
    if (!handle)
        return CV_ERROR_FAIL;
    try
    {
        CvCapture_FFMPEG_proxy* instance = (CvCapture_FFMPEG_proxy*)handle;
        Mat img;
        // TODO: avoid unnecessary copying
        if (instance->retrieveFrame(stream_idx, img))
            return callback(stream_idx, img.data, img.step, img.cols, img.rows, img.channels(), userdata);
        return CV_ERROR_FAIL;
    }
    catch(...)
    {
        return CV_ERROR_FAIL;
    }
}

static
CvResult CV_API_CALL cv_writer_open(const char* filename, int fourcc, double fps, int width, int height, int isColor,
                                    CV_OUT CvPluginWriter* handle)
{
    Size sz(width, height);
    CvVideoWriter_FFMPEG_proxy* wrt = 0;
    try
    {
        wrt = new CvVideoWriter_FFMPEG_proxy(filename, fourcc, fps, sz, isColor != 0);
        if(wrt && wrt->isOpened())
        {
            *handle = (CvPluginWriter)wrt;
            return CV_ERROR_OK;
        }
    }
    catch(...)
    {
    }
    if (wrt)
        delete wrt;
    return CV_ERROR_FAIL;
}

static
CvResult CV_API_CALL cv_writer_release(CvPluginWriter handle)
{
    if (!handle)
        return CV_ERROR_FAIL;
    CvVideoWriter_FFMPEG_proxy* instance = (CvVideoWriter_FFMPEG_proxy*)handle;
    delete instance;
    return CV_ERROR_OK;
}

static
CvResult CV_API_CALL cv_writer_get_prop(CvPluginWriter /*handle*/, int /*prop*/, CV_OUT double* /*val*/)
{
    return CV_ERROR_FAIL;
}

static
CvResult CV_API_CALL cv_writer_set_prop(CvPluginWriter /*handle*/, int /*prop*/, double /*val*/)
{
    return CV_ERROR_FAIL;
}

static
CvResult CV_API_CALL cv_writer_write(CvPluginWriter handle, const unsigned char *data, int step, int width, int height, int cn)
{
    if (!handle)
        return CV_ERROR_FAIL;
    try
    {
        CvVideoWriter_FFMPEG_proxy* instance = (CvVideoWriter_FFMPEG_proxy*)handle;
        Mat img(Size(width, height), CV_MAKETYPE(CV_8U, cn), const_cast<uchar*>(data), step);
        instance->write(img);
        return CV_ERROR_OK;
    }
    catch(...)
    {
        return CV_ERROR_FAIL;
    }
}

static const OpenCV_VideoIO_Plugin_API_preview plugin_api_v0 =
{
    {
        sizeof(OpenCV_VideoIO_Plugin_API_preview), ABI_VERSION, API_VERSION,
        CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS,
        "FFmpeg OpenCV Video I/O plugin"
    },
    /*  1*/CAP_FFMPEG,
    /*  2*/cv_capture_open,
    /*  3*/cv_capture_release,
    /*  4*/cv_capture_get_prop,
    /*  5*/cv_capture_set_prop,
    /*  6*/cv_capture_grab,
    /*  7*/cv_capture_retrieve,
    /*  8*/cv_writer_open,
    /*  9*/cv_writer_release,
    /* 10*/cv_writer_get_prop,
    /* 11*/cv_writer_set_prop,
    /* 12*/cv_writer_write
};

} // namespace

const OpenCV_VideoIO_Plugin_API_preview* opencv_videoio_plugin_init_v0(int requested_abi_version, int requested_api_version, void* /*reserved=NULL*/) CV_NOEXCEPT
{
    if (requested_abi_version != 0)
        return NULL;
    if (requested_api_version != 0)
        return NULL;
    return &cv::plugin_api_v0;
}

#endif // BUILD_PLUGIN
