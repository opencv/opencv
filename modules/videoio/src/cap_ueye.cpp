// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
This file adds support for uEye cameras in OpenCV.

Cameras can be opened by ID. If 0 is passed as ID the first available camera
will be used. For any other number, the camera associated with that ID will be
opened (c.f. IDS documentation for is_InitCamera).

Images are double buffered in a ring buffer of size 2 (called 'image memory
sequence' in the uEye SDK c.f. is_AddToSequence). The memory is locked on a
'grab' call and copied and unlocked during 'retrieve'. The image queue provided
in the uEye SDK is not used since it automatically locks the buffers when a new
image arrives, which means the buffer can fill up when frames are retrieved too
slowly.
*/

#include "precomp.hpp"

#include <ueye.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>

namespace cv
{
namespace
{
struct image_buffer
{
    char* data;
    INT id;
};
}
#define ASSERT_UEYE(expr) { UINT expr_result = expr; if(IS_SUCCESS != expr_result) CV_Error_(Error::StsAssert, ("%s %s %d: failed with code %u", #expr, __FILE__, __LINE__, expr_result)); }
#define PRINT_ON_UEYE_ERROR( expr ) { UINT expr_result = expr; if(IS_SUCCESS != expr_result) CV_LOG_ERROR(NULL, "VIDEOIO(UEYE:" << cam_id << "): " << #expr << " " << __FILE__ << " " << __LINE__ << ": failed with code " << expr_result); }

struct VideoCapture_uEye CV_FINAL: public IVideoCapture
{
    int getCaptureDomain() CV_OVERRIDE
    {
        return cv::CAP_UEYE;
    }

    VideoCapture_uEye(int camera);

    bool isOpened() const CV_OVERRIDE
    {
        return 255 != cam_id;
    }

    ~VideoCapture_uEye() CV_OVERRIDE
    {
        close();
    }

    double getProperty(int property_id) const CV_OVERRIDE;
    bool setProperty(int property_id, double value) CV_OVERRIDE;
    bool grabFrame() CV_OVERRIDE;
    bool retrieveFrame(int outputType, OutputArray frame) CV_OVERRIDE;

    void close();
    void start_camera();
    void stop_camera();

    void unlock_image_buffer();

    HIDS cam_id = 255;
    SENSORINFO sensor_info;
    double fps;
    int width;
    int height;
    int pitch;
    std::array<image_buffer, 2> ring_buffer = {{{nullptr, 0}, {nullptr, 0}}};
    char* locked_image = nullptr;
};

Ptr<IVideoCapture> create_ueye_camera(int camera)
{
    return cv::makePtr<VideoCapture_uEye>(camera);
}

namespace
{
std::vector<IMAGE_FORMAT_INFO> get_freerun_formats(HIDS cam_id)
{
    UINT count;
    ASSERT_UEYE(is_ImageFormat(cam_id, IMGFRMT_CMD_GET_NUM_ENTRIES, &count, sizeof(count)));
    UINT sizeof_list = sizeof(IMAGE_FORMAT_LIST) + (count - 1) * sizeof(IMAGE_FORMAT_INFO);
    std::unique_ptr<IMAGE_FORMAT_LIST> list(new (std::malloc(sizeof_list)) IMAGE_FORMAT_LIST);

    list->nSizeOfListEntry = sizeof(IMAGE_FORMAT_INFO);
    list->nNumListElements = count;
    ASSERT_UEYE(is_ImageFormat(cam_id, IMGFRMT_CMD_GET_LIST, list.get(), sizeof_list));

    // copy to vector and filter out non-live modes
    std::vector<IMAGE_FORMAT_INFO> formats;
    formats.reserve(count + 1);
    std::copy_if(list->FormatInfo, list->FormatInfo+count, std::back_inserter(formats), [](const IMAGE_FORMAT_INFO& format)
    {
        return (format.nSupportedCaptureModes & CAPTMODE_FREERUN);
    });

    return formats;
}

void set_matching_format(HIDS cam_id, const SENSORINFO& sensor_info, int width, int height)
{
    // uEye camera formats sometimes do not include the native resolution (without binning, subsampling or AOI)
    if(width == int(sensor_info.nMaxWidth) && height == int(sensor_info.nMaxHeight))
    {
        ASSERT_UEYE(is_SetBinning(cam_id, IS_BINNING_DISABLE));
        ASSERT_UEYE(is_SetSubSampling(cam_id, IS_SUBSAMPLING_DISABLE));
        IS_RECT rectAOI = {0, 0, width, height};
        ASSERT_UEYE(is_AOI(cam_id, IS_AOI_IMAGE_SET_AOI, &rectAOI, sizeof(rectAOI)));
        return;
    }
    auto formats = get_freerun_formats(cam_id);
    CV_Assert(formats.size() > 0);
    auto calc_err = [=](const IMAGE_FORMAT_INFO& format)
    {
        return format.nWidth - width + format.nHeight - height + (sensor_info.nMaxWidth - width)/2 - format.nX0 + (sensor_info.nMaxHeight - height)/2 - format.nY0;
    };

    std::sort(formats.begin(), formats.end(), [=](const IMAGE_FORMAT_INFO& f0, const IMAGE_FORMAT_INFO& f1)
    {
        return calc_err(f0) < calc_err(f1);
    });

    ASSERT_UEYE(is_ImageFormat(cam_id, IMGFRMT_CMD_SET_FORMAT, &formats.front().nFormatID, sizeof(UINT)));
}
}


VideoCapture_uEye::VideoCapture_uEye(int camera)
{
    CV_Assert(camera >= 0);
    CV_Assert(camera < 255); // max camera id is 254
    cam_id = static_cast<HIDS>(camera);
    CV_LOG_DEBUG(NULL, "VIDEOIO(UEYE:" << cam_id << "): opening...");
    ASSERT_UEYE(is_InitCamera(&cam_id, nullptr));

    IS_INIT_EVENT init_event = {IS_SET_EVENT_FRAME, FALSE, FALSE};
    ASSERT_UEYE(is_Event(cam_id, IS_EVENT_CMD_INIT, &init_event, sizeof(init_event)));
    UINT frame_event = IS_SET_EVENT_FRAME;
    ASSERT_UEYE(is_Event(cam_id, IS_EVENT_CMD_ENABLE, &frame_event, sizeof(frame_event)));

    ASSERT_UEYE(is_ResetToDefault(cam_id));

    ASSERT_UEYE(is_SetFrameRate(cam_id, IS_GET_FRAMERATE, &fps));

    start_camera();
}

double VideoCapture_uEye::getProperty(int property_id) const
{
    auto value = 0.;
    switch (property_id)
    {
    case CAP_PROP_FRAME_WIDTH:
        value = width;
        break;
    case CAP_PROP_FRAME_HEIGHT:
        value = height;
        break;
    case CAP_PROP_FPS:
        value = fps;
        break;
    }
    return value;
}

bool VideoCapture_uEye::setProperty(int property_id, double value)
{
    if(!isOpened())
        return false;
    try
    {
        bool set_format = false;
        switch (property_id)
        {
        case CAP_PROP_FRAME_WIDTH:
            if(width == value)
                break;
            width = static_cast<int>(value);
            set_format = true;
            break;
        case CAP_PROP_FRAME_HEIGHT:
            if(height == value)
                break;
            height = static_cast<int>(value);
            set_format = true;
            break;
        case CAP_PROP_FPS:
            if(fps == value)
                break;
            ASSERT_UEYE(is_SetFrameRate(cam_id, value, &fps));
            break;
        }
        if(set_format)
        {
            set_matching_format(cam_id, sensor_info, width, height);
            start_camera();
        }
    }
    catch(const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "VIDEOIO(UEYE:" << cam_id << "): " <<  e.what());
        return false;
    }

    return true;
}

bool VideoCapture_uEye::grabFrame()
{
    if (!isOpened())
        return false;

    try
    {
        IS_WAIT_EVENT wait_event{IS_SET_EVENT_FRAME, static_cast<UINT>(3*1000/fps), 0, 0}; // wait for the time it should take to get 3 frames
        ASSERT_UEYE(is_Event(cam_id, IS_EVENT_CMD_WAIT, &wait_event, sizeof(wait_event)));
        INT current_buffer_id;
        char* current_buffer;
        char* last;
        ASSERT_UEYE(is_GetActSeqBuf(cam_id, &current_buffer_id, &current_buffer, &last));

        const int lock_tries = 4;
        std::chrono::milliseconds lock_time_out(static_cast<int>(1000/(fps*4))); // wait for a quarter of a frame if not lockable, should not occur in event mode
        UINT ret;
        for(int i = 0; i < lock_tries; i++) // try locking the buffer
        {
            ret = is_LockSeqBuf(cam_id, IS_IGNORE_PARAMETER, last);
            if(IS_SEQ_BUFFER_IS_LOCKED == ret)
                std::this_thread::sleep_for(lock_time_out);
            else
                break;
        }
        ASSERT_UEYE(ret);
        locked_image = last;
    }
    catch(const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "VIDEOIO(UEYE:" << cam_id << "): " <<  e.what());
        close();
        return false;
    }
    return true;
}

bool VideoCapture_uEye::retrieveFrame(int /*outputType*/, OutputArray frame)
{
    if(!locked_image)
        return false;
    Mat(height, width, CV_8UC3, locked_image, pitch).copyTo(frame);
    try
    {
        unlock_image_buffer();
    }
    catch(const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "VIDEOIO(UEYE:" << cam_id << "): " <<  e.what());
        return false;
    }

    return true;
}

void VideoCapture_uEye::start_camera()
{
    stop_camera();

    IS_RECT aoi;
    ASSERT_UEYE(is_AOI(cam_id, IS_AOI_IMAGE_GET_AOI, &aoi, sizeof(aoi)));

    UINT x_is_abs_pos;
    UINT y_is_abs_pos;

    ASSERT_UEYE(is_AOI(cam_id, IS_AOI_IMAGE_GET_POS_X_ABS, &x_is_abs_pos , sizeof(x_is_abs_pos)));
    ASSERT_UEYE(is_AOI(cam_id, IS_AOI_IMAGE_GET_POS_Y_ABS, &y_is_abs_pos , sizeof(y_is_abs_pos)));

    ASSERT_UEYE(is_GetSensorInfo(cam_id, &sensor_info));
    width  = x_is_abs_pos? sensor_info.nMaxWidth: aoi.s32Width;
    height = y_is_abs_pos? sensor_info.nMaxHeight: aoi.s32Height;

    // allocate ring_buffer
    int bpp = 24;
    for(auto& image_memory: ring_buffer)
    {
        ASSERT_UEYE(is_AllocImageMem(cam_id, width, height, bpp, &image_memory.data, &image_memory.id));
        ASSERT_UEYE(is_AddToSequence(cam_id, image_memory.data, image_memory.id));
    }

    // TODO: this could be set according to sensor_info.nColorMode and CAP_PROP_FOURCC
    ASSERT_UEYE(is_SetColorMode(cam_id, IS_CM_BGR8_PACKED));
    ASSERT_UEYE(is_GetImageMemPitch (cam_id, &pitch));

    ASSERT_UEYE(is_CaptureVideo(cam_id, IS_DONT_WAIT));
}

void VideoCapture_uEye::stop_camera()
{
    if(is_CaptureVideo(cam_id, IS_GET_LIVE))
        ASSERT_UEYE(is_StopLiveVideo(cam_id, IS_FORCE_VIDEO_STOP));

    if(locked_image)
        unlock_image_buffer();
    ASSERT_UEYE(is_ClearSequence(cam_id));
    for(auto buffer: ring_buffer)
    {
        if(buffer.data)
        {
            ASSERT_UEYE(is_FreeImageMem(cam_id, buffer.data, buffer.id));
            buffer.data = nullptr;
        }
    }
}

void VideoCapture_uEye::close()
{
    if(!isOpened())
        return;
    CV_LOG_DEBUG(NULL, "VIDEOIO(UEYE:" << cam_id << "): closing...");
    // During closing we do not care about correct error handling as much.
    // Either something has gone wrong already or it has been called from the
    // destructor. Just make sure that all calls are done.
    try
    {
        stop_camera();
    }
    catch(const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "VIDEOIO(UEYE:" << cam_id << "): " <<  e.what());
    }
    UINT frame_event = IS_SET_EVENT_FRAME;
    PRINT_ON_UEYE_ERROR(is_Event(cam_id, IS_EVENT_CMD_DISABLE, &frame_event, sizeof(frame_event)));
    PRINT_ON_UEYE_ERROR(is_Event(cam_id, IS_EVENT_CMD_EXIT, &frame_event, sizeof(frame_event)));
    PRINT_ON_UEYE_ERROR(is_ExitCamera(cam_id));
    cam_id = 255;
}

void VideoCapture_uEye::unlock_image_buffer()
{
    char* tmp_buffer = nullptr;
    std::swap(locked_image, tmp_buffer);
    ASSERT_UEYE(is_UnlockSeqBuf(cam_id, IS_IGNORE_PARAMETER, tmp_buffer));
}
} // namespace cv

// plugin glue
#if defined(BUILD_PLUGIN)

#define ABI_VERSION 0
#define API_VERSION 0
#include "plugin_api.hpp"

namespace cv
{

namespace
{
#define CV_PLUGIN_NULL_FAIL(ptr) if(!ptr) return CV_ERROR_FAIL;
#define CV_PLUGIN_CALL_BEGIN CV_PLUGIN_NULL_FAIL(handle) try {
#define CV_PLUGIN_CALL_END } catch (...) { return CV_ERROR_FAIL; }

CvResult CV_API_CALL cv_capture_open(const char*, int cam_id, CV_OUT CvPluginCapture* handle)
{
    CV_PLUGIN_CALL_BEGIN

    *handle = NULL;
    std::unique_ptr<VideoCapture_uEye> cap(new VideoCapture_uEye(cam_id));
    if (cap->isOpened())
    {
        *handle = (CvPluginCapture)cap.release();
        return CV_ERROR_OK;
    }
    return CV_ERROR_FAIL;

    CV_PLUGIN_CALL_END
}

CvResult CV_API_CALL cv_capture_release(CvPluginCapture handle)
{
    CV_PLUGIN_NULL_FAIL(handle)

    VideoCapture_uEye* instance = (VideoCapture_uEye*)handle;
    delete instance;
    return CV_ERROR_OK;
}


CvResult CV_API_CALL cv_capture_get_prop(CvPluginCapture handle, int prop, CV_OUT double* val)
{
    CV_PLUGIN_NULL_FAIL(val)
    CV_PLUGIN_CALL_BEGIN

    VideoCapture_uEye* instance = (VideoCapture_uEye*)handle;
    *val = instance->getProperty(prop);
    return CV_ERROR_OK;

    CV_PLUGIN_CALL_END
}

CvResult CV_API_CALL cv_capture_set_prop(CvPluginCapture handle, int prop, double val)
{
    CV_PLUGIN_CALL_BEGIN

    VideoCapture_uEye* instance = (VideoCapture_uEye*)handle;
    return instance->setProperty(prop, val) ? CV_ERROR_OK : CV_ERROR_FAIL;

    CV_PLUGIN_CALL_END
}

CvResult CV_API_CALL cv_capture_grab(CvPluginCapture handle)
{
    CV_PLUGIN_CALL_BEGIN

    VideoCapture_uEye* instance = (VideoCapture_uEye*)handle;
    return instance->grabFrame() ? CV_ERROR_OK : CV_ERROR_FAIL;

    CV_PLUGIN_CALL_END
}

CvResult CV_API_CALL cv_capture_retrieve(CvPluginCapture handle, int stream_idx, cv_videoio_retrieve_cb_t callback, void* userdata)
{
    CV_PLUGIN_CALL_BEGIN

    VideoCapture_uEye* instance = (VideoCapture_uEye*)handle;
    Mat img;
    if (instance->retrieveFrame(stream_idx, img))
        return callback(stream_idx, img.data, (int)img.step, img.cols, img.rows, img.channels(), userdata);
    return CV_ERROR_FAIL;

    CV_PLUGIN_CALL_END
}

CvResult CV_API_CALL cv_writer_open(const char* /*filename*/, int /*fourcc*/, double /*fps*/, int /*width*/, int /*height*/, int /*isColor*/,
                                    CV_OUT CvPluginWriter* /*handle*/)
{
    return CV_ERROR_FAIL;
}

CvResult CV_API_CALL cv_writer_release(CvPluginWriter /*handle*/)
{
    return CV_ERROR_FAIL;
}

CvResult CV_API_CALL cv_writer_get_prop(CvPluginWriter /*handle*/, int /*prop*/, CV_OUT double* /*val*/)
{
    return CV_ERROR_FAIL;
}

CvResult CV_API_CALL cv_writer_set_prop(CvPluginWriter /*handle*/, int /*prop*/, double /*val*/)
{
    return CV_ERROR_FAIL;
}

CvResult CV_API_CALL cv_writer_write(CvPluginWriter /*handle*/, const unsigned char* /*data*/, int /*step*/, int /*width*/, int /*height*/, int /*cn*/)
{
    return CV_ERROR_FAIL;
}

const OpenCV_VideoIO_Plugin_API_preview plugin_api =
{
    {
        sizeof(OpenCV_VideoIO_Plugin_API_preview), ABI_VERSION, API_VERSION,
        CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS,
        "uEye OpenCV Video I/O plugin"
    },
    {
        /*  1*/CAP_UEYE,
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
    }
};
} // namespace
} // namespace cv

const OpenCV_VideoIO_Plugin_API_preview* opencv_videoio_plugin_init_v0(int requested_abi_version, int requested_api_version, void* /*reserved=NULL*/) CV_NOEXCEPT
{
    if (requested_abi_version == ABI_VERSION && requested_api_version <= API_VERSION)
        return &cv::plugin_api;
    return NULL;
}

#endif // BUILD_PLUGIN
