// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#if defined(BUILD_PLUGIN)

#include <string>
#include "cap_mfx_reader.hpp"
#include "cap_mfx_writer.hpp"
#include "plugin_api.hpp"

using namespace std;

namespace cv {

static
CvResult CV_API_CALL cv_capture_open(const char* filename, int, CV_OUT CvPluginCapture* handle)
{
    if (!handle)
        return CV_ERROR_FAIL;
    *handle = NULL;
    if (!filename)
        return CV_ERROR_FAIL;
    VideoCapture_IntelMFX *cap = 0;
    try
    {
        if (filename)
        {
            cap = new VideoCapture_IntelMFX(string(filename));
            if (cap->isOpened())
            {
                *handle = (CvPluginCapture)cap;
                return CV_ERROR_OK;
            }
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
    VideoCapture_IntelMFX* instance = (VideoCapture_IntelMFX*)handle;
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
        VideoCapture_IntelMFX* instance = (VideoCapture_IntelMFX*)handle;
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
        VideoCapture_IntelMFX* instance = (VideoCapture_IntelMFX*)handle;
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
        VideoCapture_IntelMFX* instance = (VideoCapture_IntelMFX*)handle;
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
        VideoCapture_IntelMFX* instance = (VideoCapture_IntelMFX*)handle;
        Mat img;
        if (instance->retrieveFrame(stream_idx, img))
            return callback(stream_idx, img.data, (int)img.step, img.cols, img.rows, img.channels(), userdata);
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
    VideoWriter_IntelMFX* wrt = 0;
    try
    {
        wrt = new VideoWriter_IntelMFX(filename, fourcc, fps, Size(width, height), isColor);
        if(wrt->isOpened())
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
    VideoWriter_IntelMFX* instance = (VideoWriter_IntelMFX*)handle;
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
        VideoWriter_IntelMFX* instance = (VideoWriter_IntelMFX*)handle;
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
        "MediaSDK OpenCV Video I/O plugin"
    },
    /*  1*/CAP_INTEL_MFX,
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
