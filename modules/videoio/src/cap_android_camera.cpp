// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Contributed by Giles Payne

#include "precomp.hpp"

#include <memory>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <android/log.h>
#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraError.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraMetadataTags.h>
#include <media/NdkImageReader.h>

using namespace cv;

#define TAG "NativeCamera"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define MAX_BUF_COUNT 4

#define COLOR_FormatUnknown -1
#define COLOR_FormatYUV420Planar 19
#define COLOR_FormatYUV420SemiPlanar 21

template <typename T> class RangeValue {
public:
    T min, max;
    /**
     * return absolute value from relative value
     * * value: in percent (50 for 50%)
     * */
    T value(int percent) {
        return static_cast<T>(min + (max - min) * percent / 100);
    }
    RangeValue() { min = max = static_cast<T>(0); }
    bool Supported(void) const { return (min != max); }
};

static inline void deleter_ACameraManager(ACameraManager *cameraManager) {
    ACameraManager_delete(cameraManager);
}

static inline void deleter_ACameraIdList(ACameraIdList *cameraIdList) {
    ACameraManager_deleteCameraIdList(cameraIdList);
}

static inline void deleter_ACameraDevice(ACameraDevice *cameraDevice) {
    ACameraDevice_close(cameraDevice);
}

static inline void deleter_ACameraMetadata(ACameraMetadata *cameraMetadata) {
    ACameraMetadata_free(cameraMetadata);
}

static inline void deleter_AImageReader(AImageReader *imageReader) {
    AImageReader_delete(imageReader);
}

static inline void deleter_ACaptureSessionOutputContainer(ACaptureSessionOutputContainer *outputContainer) {
    ACaptureSessionOutputContainer_free(outputContainer);
}

static inline void deleter_ACameraCaptureSession(ACameraCaptureSession *captureSession) {
    ACameraCaptureSession_close(captureSession);
}

static inline void deleter_AImage(AImage *image) {
    AImage_delete(image);
}

static inline void deleter_ANativeWindow(ANativeWindow *nativeWindow) {
    ANativeWindow_release(nativeWindow);
}

static inline void deleter_ACaptureSessionOutput(ACaptureSessionOutput *sessionOutput) {
    ACaptureSessionOutput_free(sessionOutput);
}

static inline void deleter_ACameraOutputTarget(ACameraOutputTarget *outputTarget) {
    ACameraOutputTarget_free(outputTarget);
}

static inline void deleter_ACaptureRequest(ACaptureRequest *captureRequest) {
    ACaptureRequest_free(captureRequest);
}

/*
 * CameraDevice callbacks
 */
static void OnDeviceDisconnect(void* /* ctx */, ACameraDevice* dev) {
    std::string id(ACameraDevice_getId(dev));
    LOGW("Device %s disconnected", id.c_str());
}

static void OnDeviceError(void* /* ctx */, ACameraDevice* dev, int err) {
    std::string id(ACameraDevice_getId(dev));
    LOGI("Camera Device Error: %#x, Device %s", err, id.c_str());

    switch (err) {
        case ERROR_CAMERA_IN_USE:
            LOGI("Camera in use");
            break;
        case ERROR_CAMERA_SERVICE:
            LOGI("Fatal Error occured in Camera Service");
            break;
        case ERROR_CAMERA_DEVICE:
            LOGI("Fatal Error occured in Camera Device");
            break;
        case ERROR_CAMERA_DISABLED:
            LOGI("Camera disabled");
            break;
        case ERROR_MAX_CAMERAS_IN_USE:
            LOGI("System limit for maximum concurrent cameras used was exceeded");
            break;
        default:
            LOGI("Unknown Camera Device Error: %#x", err);
    }
}

enum class CaptureSessionState {
    INITIALIZING,  // session is ready
    READY,         // session is ready
    ACTIVE,        // session is busy
    CLOSED         // session was closed
};

void OnSessionClosed(void* context, ACameraCaptureSession* session);

void OnSessionReady(void* context, ACameraCaptureSession* session);

void OnSessionActive(void* context, ACameraCaptureSession* session);

void OnCaptureCompleted(void* context,
                        ACameraCaptureSession* session,
                        ACaptureRequest* request,
                        const ACameraMetadata* result);

void OnCaptureFailed(void* context,
                     ACameraCaptureSession* session,
                     ACaptureRequest* request,
                     ACameraCaptureFailure* failure);

#define CAPTURE_TIMEOUT_SECONDS 2

/**
 * Range of Camera Exposure Time:
 *     Camera's capability range have a very long range which may be disturbing
 *     on camera. For this sample purpose, clamp to a range showing visible
 *     video on preview: 100000ns ~ 250000000ns
 */
static const long kMinExposureTime = 1000000L;
static const long kMaxExposureTime = 250000000L;

class AndroidCameraCapture : public IVideoCapture
{
    int cachedIndex;
    std::shared_ptr<ACameraManager> cameraManager;
    std::shared_ptr<ACameraDevice> cameraDevice;
    std::shared_ptr<AImageReader> imageReader;
    std::shared_ptr<ACaptureSessionOutputContainer> outputContainer;
    std::shared_ptr<ACaptureSessionOutput> sessionOutput;
    std::shared_ptr<ANativeWindow> nativeWindow;
    std::shared_ptr<ACameraOutputTarget> outputTarget;
    std::shared_ptr<ACaptureRequest> captureRequest;
    std::shared_ptr<ACameraCaptureSession> captureSession;
    CaptureSessionState sessionState = CaptureSessionState::INITIALIZING;
    int32_t frameWidth = 0;
    int32_t frameHeight = 0;
    int32_t colorFormat;
    std::vector<uint8_t> buffer;
    bool sessionOutputAdded = false;
    bool targetAdded = false;
    // properties
    bool convertToRgb = false;
    bool settingWidth = false;
    bool settingHeight = false;
    int desiredWidth = 640;
    int desiredHeight = 480;
    bool autoExposure = true;
    int64_t exposureTime = 0L;
    RangeValue<int64_t> exposureRange;
    int32_t sensitivity = 0;
    RangeValue<int32_t> sensitivityRange;

public:
    // for synchronization with NDK capture callback
    bool waitingCapture = false;
    bool captureSuccess = false;
    std::mutex mtx;
    std::condition_variable condition;

public:
    AndroidCameraCapture() {}

    ~AndroidCameraCapture() { cleanUp(); }

    ACameraDevice_stateCallbacks* GetDeviceListener() {
        static ACameraDevice_stateCallbacks cameraDeviceListener = {
            .onDisconnected = ::OnDeviceDisconnect,
            .onError = ::OnDeviceError,
        };
        return &cameraDeviceListener;
    }

    ACameraCaptureSession_stateCallbacks sessionListener;

    ACameraCaptureSession_stateCallbacks* GetSessionListener() {
        sessionListener = {
            .context = this,
            .onClosed = ::OnSessionClosed,
            .onReady = ::OnSessionReady,
            .onActive = ::OnSessionActive,
        };
        return &sessionListener;
    }

    ACameraCaptureSession_captureCallbacks captureListener;

    ACameraCaptureSession_captureCallbacks* GetCaptureCallback() {
        captureListener = {
            .context = this,
            .onCaptureStarted = nullptr,
            .onCaptureProgressed = nullptr,
            .onCaptureCompleted = ::OnCaptureCompleted,
            .onCaptureFailed = ::OnCaptureFailed,
            .onCaptureSequenceCompleted = nullptr,
            .onCaptureSequenceAborted = nullptr,
            .onCaptureBufferLost = nullptr,
        };
        return &captureListener;
    }

    void setSessionState(CaptureSessionState newSessionState) {
        this->sessionState = newSessionState;
    }

    bool isOpened() const CV_OVERRIDE { return imageReader.get() != nullptr && captureSession.get() != nullptr; }

    int getCaptureDomain() CV_OVERRIDE { return CAP_ANDROID; }

    bool grabFrame() CV_OVERRIDE
    {
        AImage* img;
        {
            std::unique_lock<std::mutex> lock(mtx);
            media_status_t mStatus = AImageReader_acquireLatestImage(imageReader.get(), &img);
            if (mStatus != AMEDIA_OK) {
                if (mStatus == AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE) {
                    // this error is not fatal - we just need to wait for a buffer to become available
                    LOGW("No Buffer Available error occured - waiting for callback");
                    waitingCapture = true;
                    captureSuccess = false;
                    bool captured = condition.wait_for(lock, std::chrono::seconds(CAPTURE_TIMEOUT_SECONDS), [this]{ return captureSuccess; });
                    waitingCapture = false;
                    if (captured) {
                        mStatus = AImageReader_acquireLatestImage(imageReader.get(), &img);
                        if (mStatus != AMEDIA_OK) {
                            LOGE("Acquire image failed with error code: %d", mStatus);
                            return false;
                        }
                    } else {
                        LOGE("Capture failed or callback timed out");
                        return false;
                    }
                } else {
                    LOGE("Acquire image failed with error code: %d", mStatus);
                    return false;
                }
            }
        }
        std::shared_ptr<AImage> image = std::shared_ptr<AImage>(img, deleter_AImage);
        int32_t srcFormat = -1;
        AImage_getFormat(image.get(), &srcFormat);
        if (srcFormat != AIMAGE_FORMAT_YUV_420_888) {
            LOGE("Incorrect image format");
            return false;
        }
        int32_t srcPlanes = 0;
        AImage_getNumberOfPlanes(image.get(), &srcPlanes);
        if (srcPlanes != 3) {
            LOGE("Incorrect number of planes in image data");
            return false;
        }
        int32_t yStride, uvStride;
        uint8_t *yPixel, *uPixel, *vPixel;
        int32_t yLen, uLen, vLen;
        int32_t uvPixelStride;
        AImage_getPlaneRowStride(image.get(), 0, &yStride);
        AImage_getPlaneRowStride(image.get(), 1, &uvStride);
        AImage_getPlaneData(image.get(), 0, &yPixel, &yLen);
        AImage_getPlaneData(image.get(), 1, &vPixel, &vLen);
        AImage_getPlaneData(image.get(), 2, &uPixel, &uLen);
        AImage_getPlanePixelStride(image.get(), 1, &uvPixelStride);

        if ( (uvPixelStride == 2) && (vPixel == uPixel + 1) && (yLen == frameWidth * frameHeight) && (uLen == ((yLen / 2) - 1)) && (vLen == uLen) ) {
            colorFormat = COLOR_FormatYUV420SemiPlanar;
        } else if ( (uvPixelStride == 1) && (vPixel = uPixel + uLen) && (yLen == frameWidth * frameHeight) && (uLen == yLen / 4) && (vLen == uLen) ) {
            colorFormat = COLOR_FormatYUV420Planar;
        } else {
            colorFormat = COLOR_FormatUnknown;
            LOGE("Unsupported format");
            return false;
        }

        buffer.clear();
        buffer.insert(buffer.end(), yPixel, yPixel + yLen);
        buffer.insert(buffer.end(), uPixel, uPixel + yLen / 2);
        return true;
    }

    bool retrieveFrame(int, OutputArray out) CV_OVERRIDE
    {
        if (buffer.empty()) {
            return false;
        }
        Mat yuv(frameHeight + frameHeight/2, frameWidth, CV_8UC1, buffer.data());
        if (colorFormat == COLOR_FormatYUV420Planar) {
            cv::cvtColor(yuv, out, convertToRgb ? cv::COLOR_YUV2RGB_YV12 : cv::COLOR_YUV2BGR_YV12);
        } else if (colorFormat == COLOR_FormatYUV420SemiPlanar) {
            cv::cvtColor(yuv, out, convertToRgb ? COLOR_YUV2RGB_NV21 : cv::COLOR_YUV2BGR_NV21);
        } else {
            LOGE("Unsupported video format: %d", colorFormat);
            return false;
        }
        return true;
    }

    double getProperty(int property_id) const CV_OVERRIDE
    {
        switch (property_id) {
            case CV_CAP_PROP_FRAME_WIDTH:
                return isOpened() ? frameWidth : desiredWidth;
            case CV_CAP_PROP_FRAME_HEIGHT:
                return isOpened() ? frameHeight : desiredHeight;
            case CV_CAP_PROP_CONVERT_RGB:
                return convertToRgb ? 1 : 0;
            case CAP_PROP_AUTO_EXPOSURE:
                return autoExposure ? 1 : 0;
            case CV_CAP_PROP_EXPOSURE:
                return exposureTime;
            case CV_CAP_PROP_ISO_SPEED:
                return sensitivity;
            default:
                break;
        }
        // unknown parameter or value not available
        return -1;
    }

    bool setProperty(int property_id, double value) CV_OVERRIDE
    {
        switch (property_id) {
            case CV_CAP_PROP_FRAME_WIDTH:
                desiredWidth = value;
                settingWidth = true;
                if (settingWidth && settingHeight) {
                    setWidthHeight();
                    settingWidth = false;
                    settingHeight = false;
                }
                return true;
            case CV_CAP_PROP_FRAME_HEIGHT:
                desiredHeight = value;
                settingHeight = true;
                if (settingWidth && settingHeight) {
                    setWidthHeight();
                    settingWidth = false;
                    settingHeight = false;
                }
                return true;
            case CV_CAP_PROP_CONVERT_RGB:
                convertToRgb = (value != 0);
                return true;
            case CAP_PROP_AUTO_EXPOSURE:
                autoExposure = (value != 0);
                if (isOpened()) {
                    uint8_t aeMode = autoExposure ? ACAMERA_CONTROL_AE_MODE_ON : ACAMERA_CONTROL_AE_MODE_OFF;
                    camera_status_t status = ACaptureRequest_setEntry_u8(captureRequest.get(), ACAMERA_CONTROL_AE_MODE, 1, &aeMode);
                    return status == ACAMERA_OK;
                }
                return true;
            case CV_CAP_PROP_EXPOSURE:
                if (isOpened() && exposureRange.Supported()) {
                    exposureTime = (int64_t)value;
                    camera_status_t status = ACaptureRequest_setEntry_i64(captureRequest.get(), ACAMERA_SENSOR_EXPOSURE_TIME, 1, &exposureTime);
                    return status == ACAMERA_OK;
                }
                break;
            case CV_CAP_PROP_ISO_SPEED:
                if (isOpened() && sensitivityRange.Supported()) {
                    sensitivity = (int32_t)value;
                    camera_status_t status = ACaptureRequest_setEntry_i32(captureRequest.get(), ACAMERA_SENSOR_SENSITIVITY, 1, &sensitivity);
                    return status == ACAMERA_OK;
                }
                break;
            default:
                break;
        }
        return false;
    }

    void setWidthHeight() {
        cleanUp();
        initCapture(cachedIndex);
    }

    // calculate a score based on how well the width and height match the desired width and height
    // basically draw the 2 rectangle on top of each other and take the ratio of the non-overlapping
    // area to the overlapping area
    double getScore(int32_t width, int32_t height) {
        double area1 = width * height;
        double area2 = desiredWidth * desiredHeight;
        if ((width < desiredWidth) == (height < desiredHeight)) {
            return (width < desiredWidth) ? (area2 - area1)/area1 : (area1 - area2)/area2;
        } else {
            int32_t overlappedWidth = std::min(width, desiredWidth);
            int32_t overlappedHeight = std::min(height, desiredHeight);
            double overlappedArea = overlappedWidth * overlappedHeight;
            return (area1 + area2 - overlappedArea)/overlappedArea;
        }
    }

    bool initCapture(int index)
    {
        cachedIndex = index;
        cameraManager = std::shared_ptr<ACameraManager>(ACameraManager_create(), deleter_ACameraManager);
        if (!cameraManager) {
            return false;
        }
        ACameraIdList* cameraIds = nullptr;
        camera_status_t cStatus = ACameraManager_getCameraIdList(cameraManager.get(), &cameraIds);
        if (cStatus != ACAMERA_OK) {
            LOGE("Get camera list failed with error code: %d", cStatus);
            return false;
        }
        std::shared_ptr<ACameraIdList> cameraIdList = std::shared_ptr<ACameraIdList>(cameraIds, deleter_ACameraIdList);
        if (index < 0 || index >= cameraIds->numCameras) {
            LOGE("Camera index out of range %d (Number of cameras: %d)", index, cameraIds->numCameras);
            return false;
        }
        ACameraDevice* camera = nullptr;
        cStatus = ACameraManager_openCamera(cameraManager.get(), cameraIdList.get()->cameraIds[index], GetDeviceListener(), &camera);
        if (cStatus != ACAMERA_OK) {
            LOGE("Open camera failed with error code: %d", cStatus);
            return false;
        }
        cameraDevice = std::shared_ptr<ACameraDevice>(camera, deleter_ACameraDevice);
        ACameraMetadata* metadata;
        cStatus = ACameraManager_getCameraCharacteristics(cameraManager.get(), cameraIdList.get()->cameraIds[index], &metadata);
        if (cStatus != ACAMERA_OK) {
            LOGE("Get camera characteristics failed with error code: %d", cStatus);
            return false;
        }
        std::shared_ptr<ACameraMetadata> cameraMetadata = std::shared_ptr<ACameraMetadata>(metadata, deleter_ACameraMetadata);
        ACameraMetadata_const_entry entry;
        ACameraMetadata_getConstEntry(cameraMetadata.get(), ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &entry);

        double bestScore = std::numeric_limits<double>::max();
        int32_t bestMatchWidth = 0;
        int32_t bestMatchHeight = 0;

        for (uint32_t i = 0; i < entry.count; i += 4) {
            int32_t input = entry.data.i32[i + 3];
            int32_t format = entry.data.i32[i + 0];
            if (input) {
                continue;
            }
            if (format == AIMAGE_FORMAT_YUV_420_888) {
                int32_t width = entry.data.i32[i + 1];
                int32_t height = entry.data.i32[i + 2];
                if (width == desiredWidth && height == desiredHeight) {
                    bestMatchWidth = width;
                    bestMatchHeight = height;
                    bestScore = 0;
                    break;
                } else {
                    double score = getScore(width, height);
                    if (score < bestScore) {
                        bestMatchWidth = width;
                        bestMatchHeight = height;
                        bestScore = score;
                    }
                }
            }
        }

        ACameraMetadata_const_entry val = { 0, };
        camera_status_t status = ACameraMetadata_getConstEntry(cameraMetadata.get(), ACAMERA_SENSOR_INFO_EXPOSURE_TIME_RANGE, &val);
        if (status == ACAMERA_OK) {
            exposureRange.min = val.data.i64[0];
            if (exposureRange.min < kMinExposureTime) {
                exposureRange.min = kMinExposureTime;
            }
            exposureRange.max = val.data.i64[1];
            if (exposureRange.max > kMaxExposureTime) {
                exposureRange.max = kMaxExposureTime;
            }
            exposureTime = exposureRange.value(2);
        } else {
            LOGW("Unsupported ACAMERA_SENSOR_INFO_EXPOSURE_TIME_RANGE");
            exposureRange.min = exposureRange.max = 0l;
            exposureTime = 0l;
        }
        status = ACameraMetadata_getConstEntry(cameraMetadata.get(), ACAMERA_SENSOR_INFO_SENSITIVITY_RANGE, &val);
        if (status == ACAMERA_OK){
            sensitivityRange.min = val.data.i32[0];
            sensitivityRange.max = val.data.i32[1];
            sensitivity = sensitivityRange.value(2);
        } else {
            LOGW("Unsupported ACAMERA_SENSOR_INFO_SENSITIVITY_RANGE");
            sensitivityRange.min = sensitivityRange.max = 0;
            sensitivity = 0;
        }

        AImageReader* reader;
        media_status_t mStatus = AImageReader_new(bestMatchWidth, bestMatchHeight, AIMAGE_FORMAT_YUV_420_888, MAX_BUF_COUNT, &reader);
        if (mStatus != AMEDIA_OK) {
            LOGE("ImageReader creation failed with error code: %d", mStatus);
            return false;
        }
        frameWidth = bestMatchWidth;
        frameHeight = bestMatchHeight;
        imageReader = std::shared_ptr<AImageReader>(reader, deleter_AImageReader);

        ANativeWindow *window;
        mStatus = AImageReader_getWindow(imageReader.get(), &window);
        if (mStatus != AMEDIA_OK) {
            LOGE("Could not get ANativeWindow: %d", mStatus);
            return false;
        }
        nativeWindow = std::shared_ptr<ANativeWindow>(window, deleter_ANativeWindow);

        ACaptureSessionOutputContainer* container;
        cStatus = ACaptureSessionOutputContainer_create(&container);
        if (cStatus != ACAMERA_OK) {
            LOGE("CaptureSessionOutputContainer creation failed with error code: %d", cStatus);
            return false;
        }
        outputContainer = std::shared_ptr<ACaptureSessionOutputContainer>(container, deleter_ACaptureSessionOutputContainer);

        ANativeWindow_acquire(nativeWindow.get());
        ACaptureSessionOutput* output;
        cStatus = ACaptureSessionOutput_create(nativeWindow.get(), &output);
        if (cStatus != ACAMERA_OK) {
            LOGE("CaptureSessionOutput creation failed with error code: %d", cStatus);
            return false;
        }
        sessionOutput = std::shared_ptr<ACaptureSessionOutput>(output, deleter_ACaptureSessionOutput);
        ACaptureSessionOutputContainer_add(outputContainer.get(), sessionOutput.get());
        sessionOutputAdded = true;

        ACameraOutputTarget* target;
        cStatus = ACameraOutputTarget_create(nativeWindow.get(), &target);
        if (cStatus != ACAMERA_OK) {
            LOGE("CameraOutputTarget creation failed with error code: %d", cStatus);
            return false;
        }
        outputTarget = std::shared_ptr<ACameraOutputTarget>(target, deleter_ACameraOutputTarget);

        ACaptureRequest * request;
        cStatus = ACameraDevice_createCaptureRequest(cameraDevice.get(), TEMPLATE_PREVIEW, &request);
        if (cStatus != ACAMERA_OK) {
            LOGE("CaptureRequest creation failed with error code: %d", cStatus);
            return false;
        }
        captureRequest = std::shared_ptr<ACaptureRequest>(request, deleter_ACaptureRequest);

        cStatus = ACaptureRequest_addTarget(captureRequest.get(), outputTarget.get());
        if (cStatus != ACAMERA_OK) {
            LOGE("Add target to CaptureRequest failed with error code: %d", cStatus);
            return false;
        }
        targetAdded = true;

        ACameraCaptureSession *session;
        cStatus = ACameraDevice_createCaptureSession(cameraDevice.get(), outputContainer.get(), GetSessionListener(), &session);
        if (cStatus != ACAMERA_OK) {
            LOGE("CaptureSession creation failed with error code: %d", cStatus);
            return false;
        }
        captureSession = std::shared_ptr<ACameraCaptureSession>(session, deleter_ACameraCaptureSession);
        uint8_t aeMode = autoExposure ? ACAMERA_CONTROL_AE_MODE_ON : ACAMERA_CONTROL_AE_MODE_OFF;
        ACaptureRequest_setEntry_u8(captureRequest.get(), ACAMERA_CONTROL_AE_MODE, 1, &aeMode);
        ACaptureRequest_setEntry_i32(captureRequest.get(), ACAMERA_SENSOR_SENSITIVITY, 1, &sensitivity);
        if (!autoExposure) {
            ACaptureRequest_setEntry_i64(captureRequest.get(), ACAMERA_SENSOR_EXPOSURE_TIME, 1, &exposureTime);
        }

        cStatus = ACameraCaptureSession_setRepeatingRequest(captureSession.get(), GetCaptureCallback(), 1, &request, nullptr);
        if (cStatus != ACAMERA_OK) {
            LOGE("CameraCaptureSession set repeating request failed with error code: %d", cStatus);
            return false;
        }
        return true;
    }

    void cleanUp() {
        captureListener.context = nullptr;
        sessionListener.context = nullptr;
        if (sessionState == CaptureSessionState::ACTIVE) {
            ACameraCaptureSession_stopRepeating(captureSession.get());
        }
        captureSession = nullptr;
        if (targetAdded) {
            ACaptureRequest_removeTarget(captureRequest.get(), outputTarget.get());
            targetAdded = false;
        }
        captureRequest = nullptr;
        outputTarget = nullptr;
        if (sessionOutputAdded) {
            ACaptureSessionOutputContainer_remove(outputContainer.get(), sessionOutput.get());
            sessionOutputAdded = false;
        }
        sessionOutput = nullptr;
        nativeWindow = nullptr;
        outputContainer = nullptr;
        cameraDevice = nullptr;
        cameraManager = nullptr;
        imageReader = nullptr;
    }
};

/********************************  Session management  *******************************/

void OnSessionClosed(void* context, ACameraCaptureSession* session) {
    if (context == nullptr) return;
    LOGW("session %p closed", session);
    reinterpret_cast<AndroidCameraCapture*>(context)->setSessionState(CaptureSessionState::CLOSED);
}

void OnSessionReady(void* context, ACameraCaptureSession* session) {
    if (context == nullptr) return;
    LOGW("session %p ready", session);
    reinterpret_cast<AndroidCameraCapture*>(context)->setSessionState(CaptureSessionState::READY);
}

void OnSessionActive(void* context, ACameraCaptureSession* session) {
    if (context == nullptr) return;
    LOGW("session %p active", session);
    reinterpret_cast<AndroidCameraCapture*>(context)->setSessionState(CaptureSessionState::ACTIVE);
}

void OnCaptureCompleted(void* context,
                        ACameraCaptureSession* session,
                        ACaptureRequest* /* request */,
                        const ACameraMetadata* /* result */) {
    if (context == nullptr) return;
    LOGV("session %p capture completed", session);
    AndroidCameraCapture* cameraCapture = reinterpret_cast<AndroidCameraCapture*>(context);
    std::unique_lock<std::mutex> lock(cameraCapture->mtx);

    if (cameraCapture->waitingCapture) {
        cameraCapture->waitingCapture = false;
        cameraCapture->captureSuccess = true;
        cameraCapture->condition.notify_one();
    }
}

void OnCaptureFailed(void* context,
                     ACameraCaptureSession* session,
                     ACaptureRequest* /* request */,
                     ACameraCaptureFailure* /* failure */) {
    if (context == nullptr) return;
    LOGV("session %p capture failed", session);
    AndroidCameraCapture* cameraCapture = reinterpret_cast<AndroidCameraCapture*>(context);
    std::unique_lock<std::mutex> lock(cameraCapture->mtx);

    if (cameraCapture->waitingCapture) {
        cameraCapture->waitingCapture = false;
        cameraCapture->captureSuccess = false;
        cameraCapture->condition.notify_one();
    }
}

/****************** Implementation of interface functions ********************/

Ptr<IVideoCapture> cv::createAndroidCapture_cam( int index ) {
    Ptr<AndroidCameraCapture> res = makePtr<AndroidCameraCapture>();
    if (res && res->initCapture(index))
        return res;
    return Ptr<IVideoCapture>();
}
