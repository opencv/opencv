#define USE_RECORDING_INSTEAD_PREVIEW 0

#if !defined(ANDROID_r2_2_2) && !defined(ANDROID_r2_3_3) && !defined(ANDROID_r3_0_1)
#error unsupported version of Android
#endif

#include <camera/CameraHardwareInterface.h>
#include "camera_wrapper.h"
#include "../camera_wrapper_connector/camera_properties.h"
#include <string>

using namespace android;

void debugShowFPS()
{
    static int mFrameCount = 0;
    static int mLastFrameCount = 0;
    static nsecs_t mLastFpsTime = systemTime();;
    static float mFps = 0;

    mFrameCount++;

    if ( ( mFrameCount % 30 ) == 0 ) {
        nsecs_t now = systemTime();
        nsecs_t diff = now - mLastFpsTime;
        if (diff==0)
            return;

        mFps =  ((mFrameCount - mLastFrameCount) * float(s2ns(1))) / diff;
        mLastFpsTime = now;
        mLastFrameCount = mFrameCount;
        LOGI("####### [%d] Frames, %f FPS", mFrameCount, mFps);
    }
}

class CameraHandler: public CameraListener
{
protected:
    sp<Camera> camera;
    CameraCallback cameraCallback;
    CameraParameters params;
    void* userData;
    int cameraId;

    bool isEmptyCameraCallbackReported;
    virtual void doCall(void* buffer, size_t bufferSize)
    {
        if (cameraCallback == 0)
        {
            if (!isEmptyCameraCallbackReported)
                LOGE("Camera callback is empty!");

            isEmptyCameraCallbackReported = true;
            return;
        }

        bool res = (*cameraCallback)(buffer, bufferSize, userData);

        if(!res) closeCameraConnect();
    }

    virtual void doCall(const sp<IMemory>& dataPtr)
    {
        LOGI("doCall started");

        if (dataPtr == NULL)
        {
            LOGE("CameraBuffer: dataPtr==NULL");
            return;
        }

        size_t size = dataPtr->size();
        if (size <= 0)
        {
            LOGE("CameraBuffer: IMemory object is of zero size");
            return;
        }

        unsigned char* buffer = (unsigned char *)dataPtr->pointer();
        if (!buffer)
        {
            LOGE("CameraBuffer: Buffer pointer is invalid");
            return;
        }

        doCall(buffer, size);
    }

public:
    CameraHandler(CameraCallback callback = 0, void* _userData = 0):cameraCallback(callback), userData(_userData), cameraId(0),  isEmptyCameraCallbackReported(false) {}
    virtual ~CameraHandler()
    {
	    LOGW("CameraHandler destructor is called!");
    }

    virtual void notify(int32_t msgType, int32_t ext1, int32_t ext2)
    {
        LOGE("Notify cb: %d %d %d\n", msgType, ext1, ext2);
#if 0
        if ( msgType & CAMERA_MSG_FOCUS )
            LOGE("AutoFocus %s in %llu us\n", (ext1) ? "OK" : "FAIL", timevalDelay(&autofocus_start));

        if ( msgType & CAMERA_MSG_SHUTTER )
            LOGE("Shutter done in %llu us\n", timeval_delay(&picture_start));
#endif
    }

    virtual void postData(int32_t msgType, const sp<IMemory>& dataPtr)
    {
        debugShowFPS();

        if ( msgType & CAMERA_MSG_PREVIEW_FRAME )
        {
            doCall(dataPtr);
            return;
        }

        if (msgType != CAMERA_MSG_PREVIEW_FRAME)
            LOGE("Recieved not CAMERA_MSG_PREVIEW_FRAME message %d", (int) msgType);

        if ( msgType & CAMERA_MSG_RAW_IMAGE )
            LOGE("Unexpected data format: RAW\n");

        if (msgType & CAMERA_MSG_POSTVIEW_FRAME)
            LOGE("Unexpected data format: Postview frame\n");

        if (msgType & CAMERA_MSG_COMPRESSED_IMAGE )
            LOGE("Unexpected data format: JPEG");
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

    static CameraHandler* initCameraConnect(const CameraCallback& callback, int cameraId, void* userData, CameraParameters* prevCameraParameters);
    void closeCameraConnect();
    double getProperty(int propIdx);
    void setProperty(int propIdx, double value);
    static void applyProperties(CameraHandler** ppcameraHandler);

    std::string cameraPropertySupportedPreviewSizesString;
};


CameraHandler* CameraHandler::initCameraConnect(const CameraCallback& callback, int cameraId, void* userData, CameraParameters* prevCameraParameters)
{
//    if (camera != NULL)
//    {
//        LOGE("initCameraConnect: camera have been connected already");
//        return false;
//    }

    sp<Camera> camera = 0;

#ifdef ANDROID_r2_2_2
    camera = Camera::connect();
#endif
#ifdef ANDROID_r2_3_3
    camera = Camera::connect(cameraId);
#endif

    if ( NULL == camera.get() )
    {
        LOGE("initCameraConnect: Unable to connect to CameraService\n");
        return 0;
    }

    CameraHandler* handler = new CameraHandler(callback, userData);
    camera->setListener(handler);

    handler->camera = camera;
    handler->cameraId=cameraId;
#if 1 
    //setting paramers from previous camera handler
    if (prevCameraParameters != NULL) {
	    camera->setParameters(prevCameraParameters->flatten());
    }
#endif
    handler->params.unflatten(camera->getParameters());


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


    //TODO: check if yuv420i format available. Set this format as preview format.

#if USE_RECORDING_INSTEAD_PREVIEW
    status_t err = camera->setPreviewDisplay(sp<ISurface>(NULL /*new DummySurface1*/));
#endif

    ////ATTENTION: switching between two versions: with and without copying memory inside Android OS
    //// see the method  CameraService::Client::copyFrameAndPostCopiedFrame and where it is used
#if 1
    camera->setPreviewCallbackFlags( FRAME_CALLBACK_FLAG_ENABLE_MASK | FRAME_CALLBACK_FLAG_COPY_OUT_MASK);//with copy
#else
    camera->setPreviewCallbackFlags( FRAME_CALLBACK_FLAG_ENABLE_MASK );//without copy
#endif

#if USE_RECORDING_INSTEAD_PREVIEW
    status_t resStart = camera->startRecording();
#else
    status_t resStart = camera->startPreview();
#endif

    if (resStart != 0)
    {
        handler->closeCameraConnect();
        handler = 0;
    }
    return handler;
}

void CameraHandler::closeCameraConnect()
{
    if (camera == NULL)
    {
        LOGI("... camera is NULL");
        return;
    }

    //TODO: ATTENTION! should we do it ALWAYS???
#if USE_RECORDING_INSTEAD_PREVIEW
    camera->stopRecording();
#else
    camera->stopPreview();
#endif

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
    switch (propIdx)
    {
    case ANDROID_CAMERA_PROPERTY_FRAMEWIDTH:
    {
        int w,h;
        params.getPreviewSize(&w,&h);
        return w;
    }
    case ANDROID_CAMERA_PROPERTY_FRAMEHEIGHT:
    {
        int w,h;
        params.getPreviewSize(&w,&h);
        return h;
    }
    case ANDROID_CAMERA_PROPERTY_SUPPORTED_PREVIEW_SIZES_STRING:
    {
	    cameraPropertySupportedPreviewSizesString=params.get(CameraParameters::KEY_SUPPORTED_PREVIEW_SIZES);
	    double res;
	    memset(&res, 0, sizeof(res));
	    (*( (void**)&res ))= (void*)( cameraPropertySupportedPreviewSizesString.c_str() );
	    
	    return res;
    }

    };
    return -1;
}

void CameraHandler::setProperty(int propIdx, double value)
{
    switch (propIdx)
    {
    case ANDROID_CAMERA_PROPERTY_FRAMEWIDTH:
    {
        int w,h;
        params.getPreviewSize(&w,&h);
        w = (int)value;
        params.setPreviewSize(w,h);
    }
    break;
    case ANDROID_CAMERA_PROPERTY_FRAMEHEIGHT:
    {
        int w,h;
        params.getPreviewSize(&w,&h);
        h = (int)value;
        params.setPreviewSize(w,h);
    }
    break;
    };
}

void CameraHandler::applyProperties(CameraHandler** ppcameraHandler)
{
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
