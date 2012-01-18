#if !defined(ANDROID_r2_2_2) && !defined(ANDROID_r2_3_3) && !defined(ANDROID_r3_0_1) && !defined(ANDROID_r4_0_0) && !defined(ANDROID_r4_0_3)
# error Building camera wrapper for your version of Android is not supported by OpenCV. You need to modify OpenCV sources in order to compile camera wrapper for your version of Android.
#endif

#include <camera/Camera.h>
#include <camera/CameraParameters.h>

#if defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3)
# include <system/camera.h>
#endif //defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3)

#include "camera_wrapper.h"
#include "../include/camera_properties.h"

#if defined(ANDROID_r3_0_1) || defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3)
//Include SurfaceTexture.h file with the SurfaceTexture class
# include <gui/SurfaceTexture.h>
# define MAGIC_OPENCV_TEXTURE_ID (0x10)
#else // defined(ANDROID_r3_0_1) || defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3)
//TODO: This is either 2.2 or 2.3. Include the headers for ISurface.h access
# include <surfaceflinger/ISurface.h>
#endif  // defined(ANDROID_r3_0_1) || defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3)

#include <string>

//undef logging macro from /system/core/libcutils/loghack.h
#ifdef LOGD
# undef LOGD
#endif

#ifdef LOGI
# undef LOGI
#endif

#ifdef LOGW
# undef LOGW
#endif

#ifdef LOGE
# undef LOGE
#endif


// LOGGING
#include <android/log.h>
#define CAMERA_LOG_TAG "OpenCV_NativeCamera"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, CAMERA_LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, CAMERA_LOG_TAG, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, CAMERA_LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, CAMERA_LOG_TAG, __VA_ARGS__))

using namespace android;

void debugShowFPS()
{
    static int mFrameCount = 0;
    static int mLastFrameCount = 0;
    static nsecs_t mLastFpsTime = systemTime();
    static float mFps = 0;

    mFrameCount++;

    if (( mFrameCount % 30 ) != 0)
        return;

    nsecs_t now = systemTime();
    nsecs_t diff = now - mLastFpsTime;

    if (diff==0)
        return;

    mFps =  ((mFrameCount - mLastFrameCount) * float(s2ns(1))) / diff;
    mLastFpsTime = now;
    mLastFrameCount = mFrameCount;
    LOGI("### Camera FPS ### [%d] Frames, %.2f FPS", mFrameCount, mFps);
}

class CameraHandler: public CameraListener
{
protected:
    int cameraId;
    sp<Camera> camera;
    CameraParameters params;
    CameraCallback cameraCallback;
    void* userData;

    int emptyCameraCallbackReported;

    void doCall(void* buffer, size_t bufferSize)
    {
        if (cameraCallback == 0)
        {
            if (!emptyCameraCallbackReported)
                LOGE("CameraHandler::doCall(void*, size_t): Camera callback is empty!");

            emptyCameraCallbackReported++;
        }
        else
        {
            bool res = (*cameraCallback)(buffer, bufferSize, userData);

            if(!res)
            {
                LOGE("CameraHandler::doCall(void*, size_t): cameraCallback returns false (camera connection will be closed)");
                closeCameraConnect();
            }
        }
    }

    void doCall(const sp<IMemory>& dataPtr)
    {
        if (dataPtr == NULL)
        {
            LOGE("CameraHandler::doCall(const sp<IMemory>&): dataPtr==NULL (no frame to handle)");
            return;
        }

        size_t size = dataPtr->size();
        if (size <= 0)
        {
            LOGE("CameraHandler::doCall(const sp<IMemory>&): IMemory object is of zero size");
            return;
        }

        void* buffer = (void *)dataPtr->pointer();
        if (!buffer)
        {
            LOGE("CameraHandler::doCall(const sp<IMemory>&): Buffer pointer is NULL");
            return;
        }

        doCall(buffer, size);
    }

    virtual void postDataTimestamp(nsecs_t timestamp, int32_t msgType, const sp<IMemory>& dataPtr)
    {
        static uint32_t count = 0;
        count++;

        LOGE("Recording cb: %d %lld %%p Offset:%%d Stride:%%d\n", msgType, timestamp);

        if (dataPtr == NULL)
        {
            LOGE("postDataTimestamp: dataPtr IS ZERO -- returning");
            camera->releaseRecordingFrame(dataPtr);
            LOGE("postDataTimestamp:  camera->releaseRecordingFrame(dataPtr) is done");
            return;
        }

        uint8_t *ptr = (uint8_t*) dataPtr->pointer();
        if (ptr)
            LOGE("VID_CB: 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x", ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8], ptr[9]);
        else
            LOGE("postDataTimestamp: Ptr is zero");

        camera->releaseRecordingFrame(dataPtr);
    }

public:
    CameraHandler(CameraCallback callback = 0, void* _userData = 0):
        cameraId(0),
        cameraCallback(callback),
        userData(_userData),
        emptyCameraCallbackReported(0)
    {
        LOGD("Instantiated new CameraHandler (%p, %p)", callback, _userData);
    }

    virtual ~CameraHandler()
    {
        LOGD("CameraHandler destructor is called");
    }

    virtual void notify(int32_t msgType, int32_t ext1, int32_t ext2)
    {
        LOGE("CameraHandler::Notify: msgType=%d ext1=%d ext2=%d\n", msgType, ext1, ext2);
#if 0
        if ( msgType & CAMERA_MSG_FOCUS )
            LOGE("CameraHandler::Notify  AutoFocus %s in %llu us\n", (ext1) ? "OK" : "FAIL", timevalDelay(&autofocus_start));

        if ( msgType & CAMERA_MSG_SHUTTER )
            LOGE("CameraHandler::Notify  Shutter done in %llu us\n", timeval_delay(&picture_start));
#endif
    }

    virtual void postData(int32_t msgType, const sp<IMemory>& dataPtr
#if defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3)
                          ,camera_frame_metadata_t* metadata
#endif
                          )
    {
        debugShowFPS();

        if ( msgType & CAMERA_MSG_PREVIEW_FRAME )
        {
            doCall(dataPtr);
            return;
        }

        //if (msgType != CAMERA_MSG_PREVIEW_FRAME)
            //LOGE("CameraHandler::postData  Recieved message %d is not equal to CAMERA_MSG_PREVIEW_FRAME (%d)", (int) msgType, CAMERA_MSG_PREVIEW_FRAME);

        if ( msgType & CAMERA_MSG_RAW_IMAGE )
            LOGE("CameraHandler::postData  Unexpected data format: RAW\n");

        if (msgType & CAMERA_MSG_POSTVIEW_FRAME)
            LOGE("CameraHandler::postData  Unexpected data format: Postview frame\n");

        if (msgType & CAMERA_MSG_COMPRESSED_IMAGE )
            LOGE("CameraHandler::postData  Unexpected data format: JPEG");
    }

    static CameraHandler* initCameraConnect(const CameraCallback& callback, int cameraId, void* userData, CameraParameters* prevCameraParameters);
    void closeCameraConnect();
    double getProperty(int propIdx);
    void setProperty(int propIdx, double value);
    static void applyProperties(CameraHandler** ppcameraHandler);

    std::string cameraPropertySupportedPreviewSizesString;
    std::string cameraPropertyPreviewFormatString;
};


CameraHandler* CameraHandler::initCameraConnect(const CameraCallback& callback, int cameraId, void* userData, CameraParameters* prevCameraParameters)
{
    LOGD("CameraHandler::initCameraConnect(%p, %d, %p, %p)", callback, cameraId, userData, prevCameraParameters);

    sp<Camera> camera = 0;

#ifdef ANDROID_r2_2_2
    camera = Camera::connect();
#else 
    /* This is 2.3 or higher. The connect method has cameraID parameter */
    camera = Camera::connect(cameraId);
#endif

    if ( 0 == camera.get() )
    {
        LOGE("initCameraConnect: Unable to connect to CameraService\n");
        return 0;
    }

    CameraHandler* handler = new CameraHandler(callback, userData);
    camera->setListener(handler);

    handler->camera = camera;
    handler->cameraId = cameraId;

    if (prevCameraParameters != 0)
    {
        LOGI("initCameraConnect: Setting paramers from previous camera handler");
        camera->setParameters(prevCameraParameters->flatten());
        handler->params.unflatten(prevCameraParameters->flatten());
    }
    else
    {
        android::String8 params_str = camera->getParameters();
        LOGI("initCameraConnect: [%s]", params_str.string());

        handler->params.unflatten(params_str);

        LOGD("Supported Cameras: %s", handler->params.get("camera-indexes"));
        LOGD("Supported Picture Sizes: %s", handler->params.get(CameraParameters::KEY_SUPPORTED_PICTURE_SIZES));
        LOGD("Supported Picture Formats: %s", handler->params.get(CameraParameters::KEY_SUPPORTED_PICTURE_FORMATS));
        LOGD("Supported Preview Sizes: %s", handler->params.get(CameraParameters::KEY_SUPPORTED_PREVIEW_SIZES));
        LOGD("Supported Preview Formats: %s", handler->params.get(CameraParameters::KEY_SUPPORTED_PREVIEW_FORMATS));
        LOGD("Supported Preview Frame Rates: %s", handler->params.get(CameraParameters::KEY_SUPPORTED_PREVIEW_FRAME_RATES));
        LOGD("Supported Thumbnail Sizes: %s", handler->params.get(CameraParameters::KEY_SUPPORTED_JPEG_THUMBNAIL_SIZES));
        LOGD("Supported Whitebalance Modes: %s", handler->params.get(CameraParameters::KEY_SUPPORTED_WHITE_BALANCE));
        LOGD("Supported Effects: %s", handler->params.get(CameraParameters::KEY_SUPPORTED_EFFECTS));
        LOGD("Supported Scene Modes: %s", handler->params.get(CameraParameters::KEY_SUPPORTED_SCENE_MODES));
        LOGD("Supported Focus Modes: %s", handler->params.get(CameraParameters::KEY_SUPPORTED_FOCUS_MODES));
        LOGD("Supported Antibanding Options: %s", handler->params.get(CameraParameters::KEY_SUPPORTED_ANTIBANDING));
        LOGD("Supported Flash Modes: %s", handler->params.get(CameraParameters::KEY_SUPPORTED_FLASH_MODES));


        //check if yuv420sp format available. Set this format as preview format.
        const char* available_formats = handler->params.get(CameraParameters::KEY_SUPPORTED_PREVIEW_FORMATS);
        if (available_formats != 0)
        {
            const char* format_to_set = 0;
            const char* pos = available_formats;
            const char* ptr = pos;
            while(true)
            {
                while(*ptr != 0 && *ptr != ',') ++ptr;
                if (ptr != pos)
                {
                    if (0 == strncmp(pos, "yuv420sp", ptr - pos))
                    {
                        format_to_set = "yuv420sp";
                        break;
                    }
                    if (0 == strncmp(pos, "yuv420i", ptr - pos))
                        format_to_set = "yuv420i";
                }
                if (*ptr == 0)
                    break;
                pos = ++ptr;
            }

            if (0 != format_to_set)
            {
                handler->params.setPreviewFormat(format_to_set);

                status_t resParams = handler->camera->setParameters(handler->params.flatten());

                if (resParams != 0)
                    LOGE("initCameraConnect: failed to set preview format to %s", format_to_set);
                else
                    LOGD("initCameraConnect: preview format is set to %s", format_to_set);
            }
        }
    }


    status_t pdstatus;
#if defined(ANDROID_r2_2_2)
    pdstatus = camera->setPreviewDisplay(sp<ISurface>(0 /*new DummySurface*/));
    if (pdstatus != 0)
        LOGE("initCameraConnect: failed setPreviewDisplay(0) call; camera migth not work correctly on some devices");
#elif defined(ANDROID_r2_3_3)
    /* Do nothing in case of 2.3 for now */

#elif defined(ANDROID_r3_0_1) || defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3)
    sp<SurfaceTexture> surfaceTexture = new SurfaceTexture(MAGIC_OPENCV_TEXTURE_ID);
    pdstatus = camera->setPreviewTexture(surfaceTexture);
    if (pdstatus != 0)
        LOGE("initCameraConnect: failed setPreviewTexture call; camera migth not work correctly");
#endif

#if !(defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3))
# if 1
    ////ATTENTION: switching between two versions: with and without copying memory inside Android OS
    //// see the method  CameraService::Client::copyFrameAndPostCopiedFrame and where it is used
    camera->setPreviewCallbackFlags( FRAME_CALLBACK_FLAG_ENABLE_MASK | FRAME_CALLBACK_FLAG_COPY_OUT_MASK);//with copy
# else
    camera->setPreviewCallbackFlags( FRAME_CALLBACK_FLAG_ENABLE_MASK );//without copy
# endif
#else
    camera->setPreviewCallbackFlags( CAMERA_FRAME_CALLBACK_FLAG_ENABLE_MASK | CAMERA_FRAME_CALLBACK_FLAG_COPY_OUT_MASK);//with copy
#endif //!(defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3))

    status_t resStart = camera->startPreview();

    if (resStart != 0)
    {
        LOGE("initCameraConnect: startPreview() fails. Closing camera connection...");
        handler->closeCameraConnect();
        handler = 0;
    }

    return handler;
}

void CameraHandler::closeCameraConnect()
{
    if (camera == NULL)
    {
        LOGI("... camera is already NULL");
        return;
    }

    camera->stopPreview();
    camera->disconnect();
    camera.clear();

    camera=NULL;
    // ATTENTION!!!!!!!!!!!!!!!!!!!!!!!!!!
    // When we set 
    //    camera=NULL
    // above, the pointed instance of android::Camera object is destructed,
    // since this member `camera' has type android::sp<Camera> (android smart pointer template class), 
    // and this is the only pointer to it.
    //
    // BUT this instance of CameraHandler is set as a listener for that android::Camera object
    // (see the function CameraHandler::initCameraConnect above),
    // so this instance of CameraHandler is pointed from that android::Camera object as
    //     sp<CameraListener>  mListener
    // and there is no other android smart pointers to this.
    //
    // It means, when that instance of the android::Camera object is destructed,
    // it calls destructor for this CameraHandler instance too.
    //
    // So, this line `camera=NULL' causes to the call `delete this' 
    // (see destructor of the template class android::sp)
    //
    // So, we must not call `delete this' after the line, since it just has been called indeed
}

double CameraHandler::getProperty(int propIdx)
{
    LOGD("CameraHandler::getProperty(%d)", propIdx);

    switch (propIdx)
    {
    case ANDROID_CAMERA_PROPERTY_FRAMEWIDTH:
    {
        int w,h;
        params.getPreviewSize(&w, &h);
        return w;
    }
    case ANDROID_CAMERA_PROPERTY_FRAMEHEIGHT:
    {
        int w,h;
        params.getPreviewSize(&w, &h);
        return h;
    }
    case ANDROID_CAMERA_PROPERTY_SUPPORTED_PREVIEW_SIZES_STRING:
    {
        cameraPropertySupportedPreviewSizesString = params.get(CameraParameters::KEY_SUPPORTED_PREVIEW_SIZES);
        union {const char* str;double res;} u;
        memset(&u.res, 0, sizeof(u.res));
        u.str = cameraPropertySupportedPreviewSizesString.c_str();
        return u.res;
    }

    case ANDROID_CAMERA_PROPERTY_PREVIEW_FORMAT_STRING:
    {
        const char* fmt = params.get(CameraParameters::KEY_PREVIEW_FORMAT);
        if (fmt == CameraParameters::PIXEL_FORMAT_YUV422SP)
            fmt = "yuv422sp";
        else if (fmt == CameraParameters::PIXEL_FORMAT_YUV420SP)
            fmt = "yuv420sp";
        else if (fmt == CameraParameters::PIXEL_FORMAT_YUV422I)
            fmt = "yuv422i";
        else if (fmt == CameraParameters::PIXEL_FORMAT_RGB565)
            fmt = "rgb565";
        else if (fmt == CameraParameters::PIXEL_FORMAT_JPEG)
            fmt = "jpeg";
        cameraPropertyPreviewFormatString = fmt;

        union {const char* str;double res;} u;
        memset(&u.res, 0, sizeof(u.res));
        u.str = cameraPropertyPreviewFormatString.c_str();
        return u.res;
    }

    };
    return -1;
}

void CameraHandler::setProperty(int propIdx, double value)
{
    LOGD("CameraHandler::setProperty(%d, %f)", propIdx, value);

    switch (propIdx)
    {
    case ANDROID_CAMERA_PROPERTY_FRAMEWIDTH:
    {
        int w,h;
        params.getPreviewSize(&w, &h);
        w = (int)value;
        params.setPreviewSize(w, h);
    }
    break;
    case ANDROID_CAMERA_PROPERTY_FRAMEHEIGHT:
    {
        int w,h;
        params.getPreviewSize(&w, &h);
        h = (int)value;
        params.setPreviewSize(w, h);
    }
    break;
    };
}

void CameraHandler::applyProperties(CameraHandler** ppcameraHandler)
{
    LOGD("CameraHandler::applyProperties()");

    if (ppcameraHandler == 0)
    {
        LOGE("applyProperties: Passed NULL ppcameraHandler");
        return;
    }

    if (*ppcameraHandler == 0)
    {
        LOGE("applyProperties: Passed null *ppcameraHandler");
        return;
    }

    LOGD("CameraHandler::applyProperties()");
    CameraHandler* previousCameraHandler=*ppcameraHandler;
    CameraParameters curCameraParameters(previousCameraHandler->params.flatten());

    CameraCallback cameraCallback=previousCameraHandler->cameraCallback;
    void* userData=previousCameraHandler->userData;
    int cameraId=previousCameraHandler->cameraId;

    LOGD("CameraHandler::applyProperties(): before previousCameraHandler->closeCameraConnect");
    previousCameraHandler->closeCameraConnect();
    LOGD("CameraHandler::applyProperties(): after previousCameraHandler->closeCameraConnect");


    LOGD("CameraHandler::applyProperties(): before initCameraConnect");
    CameraHandler* handler=initCameraConnect(cameraCallback, cameraId, userData, &curCameraParameters);
    LOGD("CameraHandler::applyProperties(): after initCameraConnect, handler=0x%x", (int)handler);
    if (handler == NULL) {
        LOGE("ERROR in applyProperties --- cannot reinit camera");
        handler=initCameraConnect(cameraCallback, cameraId, userData, NULL);
        LOGD("CameraHandler::applyProperties(): repeate initCameraConnect after ERROR, handler=0x%x", (int)handler);
        if (handler == NULL) {
            LOGE("ERROR in applyProperties --- cannot reinit camera AGAIN --- cannot do anything else");
        }
    }
    (*ppcameraHandler)=handler;
}


extern "C" {

void* initCameraConnectC(void* callback, int cameraId, void* userData)
{
    return CameraHandler::initCameraConnect((CameraCallback)callback, cameraId, userData, NULL);
}

void closeCameraConnectC(void** camera)
{
    CameraHandler** cc = (CameraHandler**)camera;
    (*cc)->closeCameraConnect();
    *cc = 0;
}

double getCameraPropertyC(void* camera, int propIdx)
{
    return ((CameraHandler*)camera)->getProperty(propIdx);
}

void setCameraPropertyC(void* camera, int propIdx, double value)
{
    ((CameraHandler*)camera)->setProperty(propIdx,value);
}

void applyCameraPropertiesC(void** camera)
{
    CameraHandler::applyProperties((CameraHandler**)camera);
}

}
