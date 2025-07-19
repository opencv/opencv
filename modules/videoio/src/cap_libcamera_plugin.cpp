// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#define CAPTURE_ABI_VERSION 1
#define CAPTURE_API_VERSION 2

#include "plugin_capture_api.hpp"
#include "cap_libcamera.hpp"

#include <opencv2/core/cvdef.h>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;

namespace {

struct LibcameraPluginCapture
{
    Ptr<LibcameraCapture> capture;
    
    LibcameraPluginCapture() = default;
    ~LibcameraPluginCapture() = default;
};

}

static CvResult CV_API_CALL libcamera_capture_open(const char* filename, int camera_index, CV_OUT CvPluginCapture* handle)
{
    (void)filename;  // Suppress unused parameter warning
    
    if (!handle)
        return CV_ERROR_FAIL;
    
    *handle = nullptr;
    
    try
    {
        LibcameraPluginCapture* instance = new LibcameraPluginCapture();
        instance->capture = makePtr<LibcameraCapture>(camera_index);
        
        if (!instance->capture || !instance->capture->isOpened())
        {
            delete instance;
            return CV_ERROR_FAIL;
        }
        
        *handle = reinterpret_cast<CvPluginCapture>(instance);
        return CV_ERROR_OK;
    }
    catch (const std::exception& e)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to open capture: " << e.what());
        return CV_ERROR_FAIL;
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to open capture: unknown exception");
        return CV_ERROR_FAIL;
    }
}

static CvResult CV_API_CALL libcamera_capture_open_with_params(
    const char* filename, int camera_index,
    int* params, unsigned n_params,
    CV_OUT CvPluginCapture* handle)
{
    (void)filename;  // Suppress unused parameter warning
    
    if (!handle)
        return CV_ERROR_FAIL;
    
    *handle = nullptr;
    
    try
    {
        LibcameraPluginCapture* instance = new LibcameraPluginCapture();
        instance->capture = makePtr<LibcameraCapture>(camera_index);
        
        if (!instance->capture || !instance->capture->isOpened())
        {
            delete instance;
            return CV_ERROR_FAIL;
        }
        
        // Apply parameters
        for (unsigned i = 0; i < n_params; ++i)
        {
            int prop = params[2*i];
            double val = static_cast<double>(params[2*i + 1]);
            instance->capture->setProperty(prop, val);
        }
        
        *handle = reinterpret_cast<CvPluginCapture>(instance);
        return CV_ERROR_OK;
    }
    catch (const std::exception& e)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to open capture with params: " << e.what());
        return CV_ERROR_FAIL;
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to open capture with params: unknown exception");
        return CV_ERROR_FAIL;
    }
}

static CvResult CV_API_CALL libcamera_capture_release(CvPluginCapture handle)
{
    if (!handle)
        return CV_ERROR_FAIL;
    
    try
    {
        LibcameraPluginCapture* instance = reinterpret_cast<LibcameraPluginCapture*>(handle);
        delete instance;
        return CV_ERROR_OK;
    }
    catch (const std::exception& e)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to release capture: " << e.what());
        return CV_ERROR_FAIL;
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to release capture: unknown exception");
        return CV_ERROR_FAIL;
    }
}

static CvResult CV_API_CALL libcamera_capture_get_property(CvPluginCapture handle, int prop, CV_OUT double* val)
{
    if (!handle || !val)
        return CV_ERROR_FAIL;
    
    try
    {
        LibcameraPluginCapture* instance = reinterpret_cast<LibcameraPluginCapture*>(handle);
        if (!instance->capture)
            return CV_ERROR_FAIL;
            
        *val = instance->capture->getProperty(prop);
        return CV_ERROR_OK;
    }
    catch (const std::exception& e)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to get property: " << e.what());
        return CV_ERROR_FAIL;
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to get property: unknown exception");
        return CV_ERROR_FAIL;
    }
}

static CvResult CV_API_CALL libcamera_capture_set_property(CvPluginCapture handle, int prop, double val)
{
    if (!handle)
        return CV_ERROR_FAIL;
    
    try
    {
        LibcameraPluginCapture* instance = reinterpret_cast<LibcameraPluginCapture*>(handle);
        if (!instance->capture)
            return CV_ERROR_FAIL;
            
        bool result = instance->capture->setProperty(prop, val);
        return result ? CV_ERROR_OK : CV_ERROR_FAIL;
    }
    catch (const std::exception& e)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to set property: " << e.what());
        return CV_ERROR_FAIL;
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to set property: unknown exception");
        return CV_ERROR_FAIL;
    }
}

static CvResult CV_API_CALL libcamera_capture_grab(CvPluginCapture handle)
{
    if (!handle)
        return CV_ERROR_FAIL;
    
    try
    {
        LibcameraPluginCapture* instance = reinterpret_cast<LibcameraPluginCapture*>(handle);
        if (!instance->capture)
            return CV_ERROR_FAIL;
            
        bool result = instance->capture->grab();
        return result ? CV_ERROR_OK : CV_ERROR_FAIL;
    }
    catch (const std::exception& e)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to grab frame: " << e.what());
        return CV_ERROR_FAIL;
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to grab frame: unknown exception");
        return CV_ERROR_FAIL;
    }
}

static CvResult CV_API_CALL libcamera_capture_retrieve(CvPluginCapture handle, int stream_idx, cv_videoio_capture_retrieve_cb_t callback, void* userdata)
{
    if (!handle || !callback)
        return CV_ERROR_FAIL;
    
    try
    {
        LibcameraPluginCapture* instance = reinterpret_cast<LibcameraPluginCapture*>(handle);
        if (!instance->capture)
            return CV_ERROR_FAIL;
        
        cv::Mat frame;
        bool result = instance->capture->retrieve(frame, stream_idx);
        if (!result || frame.empty())
            return CV_ERROR_FAIL;
        
        // Call the callback with frame data
        int type = frame.type();
        CvResult res = callback(stream_idx, frame.ptr(), static_cast<int>(frame.step[0]), 
                               frame.cols, frame.rows, type, userdata);
        
        return res;
    }
    catch (const std::exception& e)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to retrieve frame: " << e.what());
        return CV_ERROR_FAIL;
    }
    catch (...)
    {
        CV_LOG_ERROR(NULL, "Libcamera plugin: Failed to retrieve frame: unknown exception");
        return CV_ERROR_FAIL;
    }
}

static CvResult CV_API_CALL libcamera_capture_open_stream(
    void* opaque,
    long long(*read)(void* opaque, char* buffer, long long size),
    long long(*seek)(void* opaque, long long offset, int way),
    int* params, unsigned n_params,
    CV_OUT CvPluginCapture* handle)
{
    (void)opaque;   // Suppress unused parameter warning
    (void)read;     // Suppress unused parameter warning
    (void)seek;     // Suppress unused parameter warning
    (void)params;   // Suppress unused parameter warning
    (void)n_params; // Suppress unused parameter warning
    (void)handle;   // Suppress unused parameter warning
    
    // Libcamera doesn't support stream-based input
    return CV_ERROR_FAIL;
}

static const OpenCV_VideoIO_Capture_Plugin_API plugin_api =
{
    {
        sizeof(OpenCV_VideoIO_Capture_Plugin_API), // size
        CAPTURE_ABI_VERSION, // api_abi_version
        CAPTURE_API_VERSION, // api_api_version
        CV_VERSION_MAJOR, // opencv_version_major
        CV_VERSION_MINOR, // opencv_version_minor
        CV_VERSION_REVISION, // opencv_version_patch
        CV_VERSION_STATUS, // opencv_version_status
        "libcamera OpenCV plugin v1.2" // api_description
    },
    {
        CAP_LIBCAMERA, // id
        libcamera_capture_open,
        libcamera_capture_release,
        libcamera_capture_get_property,
        libcamera_capture_set_property,
        libcamera_capture_grab,
        libcamera_capture_retrieve,
    },
    {
        libcamera_capture_open_with_params,
    },
    {
        libcamera_capture_open_stream,
    }
};

extern "C" {

CV_PLUGIN_EXPORTS
const OpenCV_VideoIO_Capture_Plugin_API* CV_API_CALL opencv_videoio_capture_plugin_init_v1(int requested_abi_version, int requested_api_version, void* /*reserved*/) CV_NOEXCEPT
{
    if (requested_abi_version == CAPTURE_ABI_VERSION && requested_api_version <= CAPTURE_API_VERSION)
        return reinterpret_cast<const OpenCV_VideoIO_Capture_Plugin_API*>(&plugin_api);
    return NULL;
}

} // extern "C"
