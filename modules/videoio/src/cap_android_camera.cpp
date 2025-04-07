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

#define FOURCC_BGR CV_FOURCC_MACRO('B','G','R','3')
#define FOURCC_RGB CV_FOURCC_MACRO('R','G','B','3')
#define FOURCC_BGRA CV_FOURCC_MACRO('B','G','R','4')
#define FOURCC_RGBA CV_FOURCC_MACRO('R','G','B','4')
#define FOURCC_GRAY CV_FOURCC_MACRO('G','R','E','Y')
#define FOURCC_NV21 CV_FOURCC_MACRO('N','V','2','1')
#define FOURCC_YV12 CV_FOURCC_MACRO('Y','V','1','2')
#define FOURCC_UNKNOWN  0xFFFFFFFF

template <typename T> struct RangeValue {
    T min, max;

    RangeValue(T minv = 0, T maxv = 0) : min(minv), max(maxv) {}
    bool isValid() const { return (min != max); }
    T percentage(int percent) const {
        return static_cast<T>(min + ((max - min) * percent) / 100);
    }
    T clamp( T value ) const {
        return (value > max) ? max : ((value < min) ? min : value);
    }
};

template <typename T>
using AObjPtr = std::unique_ptr<T, std::function<void(T *)>>;

enum class CaptureSessionState {
    INITIALIZING,  // session is ready
    READY,         // session is ready
    ACTIVE,        // session is busy
    CLOSED         // session was closed
};

#define CAPTURE_TIMEOUT_SECONDS 2
#define CAPTURE_POLL_INTERVAL_MS 5

/**
 * Range of Camera Exposure Time:
 *     Camera's capability range have a very long range which may be disturbing
 *     on camera. For this sample purpose, clamp to a range showing visible
 *     video on preview: 100000ns ~ 250000000ns
 */
static const RangeValue<int64_t> exposureTimeLimits = { 1000000, 250000000 };

static double elapsedTimeFrom(std::chrono::time_point<std::chrono::system_clock> start) {
    return std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
}

class AndroidCameraCapture : public IVideoCapture
{
    int deviceIndex;
    AObjPtr<ACameraManager> cameraManager { nullptr, ACameraManager_delete };
    AObjPtr<ACameraDevice> cameraDevice { nullptr, ACameraDevice_close };
    AObjPtr<AImageReader> imageReader { nullptr, AImageReader_delete };
    AObjPtr<ACaptureSessionOutputContainer> outputContainer { nullptr, ACaptureSessionOutputContainer_free };
    AObjPtr<ACaptureSessionOutput> sessionOutput { nullptr, ACaptureSessionOutput_free };
    AObjPtr<ANativeWindow> nativeWindow { nullptr, ANativeWindow_release };
    AObjPtr<ACameraOutputTarget> outputTarget { nullptr, ACameraOutputTarget_free };
    AObjPtr<ACaptureRequest> captureRequest { nullptr, ACaptureRequest_free };
    AObjPtr<ACameraCaptureSession> captureSession { nullptr, ACameraCaptureSession_close };
    CaptureSessionState sessionState = CaptureSessionState::INITIALIZING;
    int32_t frameWidth = 0;
    int32_t frameStride = 0;
    int32_t frameHeight = 0;
    int32_t colorFormat = COLOR_FormatUnknown;
    std::vector<uint8_t> buffer;
    bool sessionOutputAdded = false;
    bool targetAdded = false;
    // properties
    uint32_t fourCC = FOURCC_UNKNOWN;
    int32_t desiredWidth = 640;
    int32_t desiredHeight = 480;
    enum SetupState { setupDone = 0, setupWidth = 0x01, setupHeight = 0x02 } widthHeightState = setupDone;
    uint8_t flashMode = ACAMERA_FLASH_MODE_OFF;
    uint8_t aeMode = ACAMERA_CONTROL_AE_MODE_ON;
    int64_t exposureTime = 0;
    RangeValue<int64_t> exposureRange;
    int32_t sensitivity = 0;
    RangeValue<int32_t> sensitivityRange;
    float zoomRatio = 1.0f;
    RangeValue<float> zoomRange;

    ACameraDevice_stateCallbacks deviceCallbacks = {};
    ACameraCaptureSession_stateCallbacks sessionCallbacks = {};
    ACameraCaptureSession_captureCallbacks captureCallbacks = {};

    static void OnDeviceDisconnect(void* ctx, ACameraDevice* dev);
    static void OnDeviceError(void* ctx, ACameraDevice* dev, int err);
    static void OnSessionClosed(void* context, ACameraCaptureSession* session);
    static void OnSessionReady(void* context, ACameraCaptureSession* session);
    static void OnSessionActive(void* context, ACameraCaptureSession* session);
    static void OnCaptureCompleted(void* context,
                                   ACameraCaptureSession* session,
                                   ACaptureRequest* request,
                                   const ACameraMetadata* result);
    static void OnCaptureFailed(void* context,
                                ACameraCaptureSession* session,
                                ACaptureRequest* request,
                                ACameraCaptureFailure* failure);

    // for synchronization with NDK capture callback
    bool waitingCapture = false;
    bool captureSuccess = false;
    std::mutex mtx;
    std::condition_variable condition;

public:
    AndroidCameraCapture(int index, const VideoCaptureParameters& params)
        : deviceIndex(index)
    {
        deviceCallbacks.context = this;
        deviceCallbacks.onError = OnDeviceError;
        deviceCallbacks.onDisconnected = OnDeviceDisconnect,

        sessionCallbacks.context = this;
        sessionCallbacks.onReady = OnSessionReady;
        sessionCallbacks.onActive = OnSessionActive;
        sessionCallbacks.onClosed = OnSessionClosed;

        captureCallbacks.context = this;
        captureCallbacks.onCaptureCompleted = OnCaptureCompleted;
        captureCallbacks.onCaptureFailed = OnCaptureFailed;

        desiredWidth = params.get<int32_t>(CAP_PROP_FRAME_WIDTH, desiredWidth);
        desiredHeight = params.get<int32_t>(CAP_PROP_FRAME_HEIGHT, desiredHeight);

        static const struct {
            int propId;
            uint32_t defaultValue;
        } items[] = {
            { CAP_PROP_AUTO_EXPOSURE, 1 },
            { CAP_PROP_FOURCC, FOURCC_UNKNOWN },
            { CAP_PROP_ANDROID_DEVICE_TORCH, 0 }
        };

        for (auto it = std::begin(items); it != std::end(items); ++it) {
            setProperty(it->propId, params.get<double>(it->propId, it->defaultValue));
        }
    }

    ~AndroidCameraCapture() { cleanUp(); }

    bool isOpened() const CV_OVERRIDE { return imageReader && captureSession; }

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
                    LOGW("No Buffer Available error occurred - waiting for callback");
                    waitingCapture = true;
                    captureSuccess = false;
                    auto start = std::chrono::system_clock::now();
                    bool captured = condition.wait_for(lock, std::chrono::seconds(
                                        CAPTURE_TIMEOUT_SECONDS), [this]{ return captureSuccess; });
                    waitingCapture = false;
                    if (captured) {
                        mStatus = AImageReader_acquireLatestImage(imageReader.get(), &img);
                        // even though an image has been captured we may not be able to acquire it
                        // straight away so we poll every 10ms
                        while (mStatus == AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE &&
                               elapsedTimeFrom(start) < CAPTURE_TIMEOUT_SECONDS) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(CAPTURE_POLL_INTERVAL_MS));
                            mStatus = AImageReader_acquireLatestImage(imageReader.get(), &img);
                        }
                        if (mStatus != AMEDIA_OK) {
                            LOGE("Acquire image failed with error code: %d", mStatus);
                            if (elapsedTimeFrom(start) >= CAPTURE_TIMEOUT_SECONDS) {
                                LOGE("Image acquisition timed out");
                            }
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
        AObjPtr<AImage> image(img, AImage_delete);
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
        AImage_getPlaneData(image.get(), 1, &uPixel, &uLen);
        AImage_getPlaneData(image.get(), 2, &vPixel, &vLen);
        AImage_getPlanePixelStride(image.get(), 1, &uvPixelStride);
        int32_t yBufferLen = yLen;

        if ( (uvPixelStride == 2) && (uPixel == vPixel + 1) &&
             (yLen == (yStride * (frameHeight - 1)) + frameWidth) &&
             (uLen == (uvStride * ((frameHeight / 2) - 1)) + frameWidth - 1) &&
             (uvStride == yStride)  && (vLen == uLen) ) {
            frameStride = yStride;
            yBufferLen = frameStride * frameHeight;
            colorFormat = COLOR_FormatYUV420SemiPlanar;
            if (fourCC == FOURCC_UNKNOWN) {
                fourCC = FOURCC_NV21;
            }
        } else if ( (uvPixelStride == 1) && (uPixel == vPixel + vLen) &&
                    (yLen == frameWidth * frameHeight) &&
                    (uLen == yLen / 4) && (vLen == uLen) ) {
            colorFormat = COLOR_FormatYUV420Planar;
            if (fourCC == FOURCC_UNKNOWN) {
                fourCC = FOURCC_YV12;
            }
        } else {
            colorFormat = COLOR_FormatUnknown;
            fourCC = FOURCC_UNKNOWN;
            LOGE("Unsupported format");
            return false;
        }

        buffer.clear();
        buffer.insert(buffer.end(), yPixel, yPixel + yBufferLen);
        buffer.insert(buffer.end(), vPixel, vPixel + yBufferLen / 2);
        return true;
    }

    bool retrieveFrame(int, OutputArray out) CV_OVERRIDE
    {
        if (buffer.empty()) {
            return false;
        }
        if (colorFormat == COLOR_FormatYUV420Planar) {
            const Mat yuv(frameHeight + frameHeight/2, frameWidth, CV_8UC1, buffer.data());
            switch (fourCC) {
                case FOURCC_BGRA:
                    cvtColor(yuv, out, COLOR_YUV2BGRA_YV12);
                    break;
                case FOURCC_RGBA:
                    cvtColor(yuv, out, COLOR_YUV2RGBA_YV12);
                    break;
                case FOURCC_BGR:
                    cvtColor(yuv, out, COLOR_YUV2BGR_YV12);
                    break;
                case FOURCC_RGB:
                    cvtColor(yuv, out, COLOR_YUV2RGB_YV12);
                    break;
                case FOURCC_GRAY:
                    cvtColor(yuv, out, COLOR_YUV2GRAY_YV12);
                    break;
                case FOURCC_YV12:
                    yuv.copyTo(out);
                    break;
                default:
                    LOGE("Unexpected FOURCC value: %d", fourCC);
                    return false;
            }
        } else if (colorFormat == COLOR_FormatYUV420SemiPlanar) {
            const Mat yuv(frameHeight + frameHeight/2, frameWidth, CV_8UC1, buffer.data(), frameStride);
            switch (fourCC) {
                case FOURCC_BGRA:
                    cvtColor(yuv, out, COLOR_YUV2BGRA_NV21);
                    break;
                case FOURCC_RGBA:
                    cvtColor(yuv, out, COLOR_YUV2RGBA_NV21);
                    break;
                case FOURCC_BGR:
                    cvtColor(yuv, out, COLOR_YUV2BGR_NV21);
                    break;
                case FOURCC_RGB:
                    cvtColor(yuv, out, COLOR_YUV2RGB_NV21);
                    break;
                case FOURCC_GRAY:
                    cvtColor(yuv, out, COLOR_YUV2GRAY_NV21);
                    break;
                case FOURCC_NV21:
                    yuv.copyTo(out);
                    break;
                default:
                    LOGE("Unexpected FOURCC value: %d", fourCC);
                    return false;
            }
        } else {
            LOGE("Unsupported video format: %d", colorFormat);
            return false;
        }
        return true;
    }

    double getProperty(int property_id) const CV_OVERRIDE
    {
        switch (property_id) {
            case CAP_PROP_FRAME_WIDTH:
                return isOpened() ? frameWidth : desiredWidth;
            case CAP_PROP_FRAME_HEIGHT:
                return isOpened() ? frameHeight : desiredHeight;
            case CAP_PROP_AUTO_EXPOSURE:
                return (aeMode == ACAMERA_CONTROL_AE_MODE_ON) ? 1 : 0;
            case CAP_PROP_EXPOSURE:
                return exposureTime;
            case CAP_PROP_ISO_SPEED:
                return sensitivity;
            case CAP_PROP_FOURCC:
                return fourCC;
            case CAP_PROP_ANDROID_DEVICE_TORCH:
                return (flashMode == ACAMERA_FLASH_MODE_TORCH) ? 1 : 0;
            case CAP_PROP_ZOOM:
                return zoomRange.isValid() ? zoomRatio : -1;
            default:
                break;
        }
        // unknown parameter or value not available
        return -1;
    }

    bool setProperty(int property_id, double value) CV_OVERRIDE
    {
        switch (property_id) {
            case CAP_PROP_FRAME_WIDTH:
                desiredWidth = static_cast<int32_t>(value);
                setWidthHeight(setupWidth);
                return true;
            case CAP_PROP_FRAME_HEIGHT:
                desiredHeight = static_cast<int32_t>(value);
                setWidthHeight(setupHeight);
                return true;
            case CAP_PROP_FOURCC:
                {
                    uint32_t newFourCC = cvRound(value);
                    if (fourCC == newFourCC) {
                        return true;
                    } else {
                        switch (newFourCC) {
                            case FOURCC_BGR:
                            case FOURCC_RGB:
                            case FOURCC_BGRA:
                            case FOURCC_RGBA:
                            case FOURCC_GRAY:
                                fourCC = newFourCC;
                                return true;
                            case FOURCC_YV12:
                                if (colorFormat == COLOR_FormatYUV420Planar) {
                                    fourCC = newFourCC;
                                    return true;
                                } else {
                                    LOGE("Unsupported FOURCC conversion COLOR_FormatYUV420SemiPlanar"
                                         " -> COLOR_FormatYUV420Planar");
                                    return false;
                                }
                            case FOURCC_NV21:
                                if (colorFormat == COLOR_FormatYUV420SemiPlanar) {
                                    fourCC = newFourCC;
                                    return true;
                                } else {
                                    LOGE("Unsupported FOURCC conversion COLOR_FormatYUV420Planar"
                                         " -> COLOR_FormatYUV420SemiPlanar");
                                    return false;
                                }
                            default:
                                LOGE("Unsupported FOURCC value: %d\n", fourCC);
                                return false;
                        }
                    }
                }
            case CAP_PROP_AUTO_EXPOSURE:
                aeMode = (value != 0) ? ACAMERA_CONTROL_AE_MODE_ON : ACAMERA_CONTROL_AE_MODE_OFF;
                if (isOpened()) {
                    return submitRequest(ACaptureRequest_setEntry_u8, ACAMERA_CONTROL_AE_MODE, aeMode);
                }
                return true;
            case CAP_PROP_EXPOSURE:
                if (isOpened() && exposureRange.isValid()) {
                    exposureTime = exposureRange.clamp(static_cast<int64_t>(value));
                    LOGI("Setting CAP_PROP_EXPOSURE will have no effect unless CAP_PROP_AUTO_EXPOSURE is off");
                    return submitRequest(ACaptureRequest_setEntry_i64, ACAMERA_SENSOR_EXPOSURE_TIME, exposureTime);
                }
                return false;
            case CAP_PROP_ISO_SPEED:
                if (isOpened() && sensitivityRange.isValid()) {
                    sensitivity = sensitivityRange.clamp(static_cast<int32_t>(value));
                    LOGI("Setting CAP_PROP_ISO_SPEED will have no effect unless CAP_PROP_AUTO_EXPOSURE is off");
                    return submitRequest(ACaptureRequest_setEntry_i32, ACAMERA_SENSOR_SENSITIVITY, sensitivity);
                }
                return false;
            case CAP_PROP_ZOOM:
                if (isOpened() && zoomRange.isValid()) {
                    zoomRatio = zoomRange.clamp(static_cast<float>(value));
                    return submitRequest(ACaptureRequest_setEntry_float, ACAMERA_CONTROL_ZOOM_RATIO, zoomRatio);
                }
                return true;
            case CAP_PROP_ANDROID_DEVICE_TORCH:
                flashMode = (value != 0) ? ACAMERA_FLASH_MODE_TORCH : ACAMERA_FLASH_MODE_OFF;
                if (isOpened()) {
                    return submitRequest(ACaptureRequest_setEntry_u8, ACAMERA_FLASH_MODE, flashMode);
                }
                return true;
            default:
                break;
        }
        return false;
    }

    bool initCapture()
    {
        cameraManager.reset(ACameraManager_create());
        if (!cameraManager) {
            LOGE("Cannot create camera manager!");
            return false;
        }
        ACameraIdList* cameraIds;
        camera_status_t cStatus = ACameraManager_getCameraIdList(cameraManager.get(), &cameraIds);
        if (cStatus != ACAMERA_OK) {
            LOGE("Get camera list failed with error code: %d", cStatus);
            return false;
        }
        AObjPtr<ACameraIdList> cameraIdList(cameraIds, ACameraManager_deleteCameraIdList);
        if (deviceIndex < 0 || deviceIndex >= cameraIds->numCameras) {
            LOGE("Camera index out of range %d (Number of cameras: %d)", deviceIndex, cameraIds->numCameras);
            return false;
        }
        const char *cameraId = cameraIdList.get()->cameraIds[deviceIndex];

        ACameraDevice* camera;
        cStatus = ACameraManager_openCamera(cameraManager.get(), cameraId, &deviceCallbacks, &camera);
        if (cStatus != ACAMERA_OK) {
            LOGE("Open camera failed with error code: %d", cStatus);
            return false;
        }
        cameraDevice.reset(camera);

        ACameraMetadata* metadata;
        cStatus = ACameraManager_getCameraCharacteristics(cameraManager.get(), cameraId, &metadata);
        if (cStatus != ACAMERA_OK) {
            LOGE("Get camera characteristics failed with error code: %d", cStatus);
            return false;
        }
        AObjPtr<ACameraMetadata> cameraMetadata(metadata, ACameraMetadata_free);

        getPropertyRanges(cameraMetadata.get());

        int32_t bestMatchWidth = 0, bestMatchHeight = 0;
        findResolutionMatch(cameraMetadata.get(), bestMatchWidth, bestMatchHeight);
        LOGI("Best resolution match: %dx%d", bestMatchWidth, bestMatchHeight);

        AImageReader* reader;
        media_status_t mStatus = AImageReader_new(bestMatchWidth, bestMatchHeight,
                                                  AIMAGE_FORMAT_YUV_420_888, MAX_BUF_COUNT, &reader);
        if (mStatus != AMEDIA_OK) {
            LOGE("ImageReader creation failed with error code: %d", mStatus);
            return false;
        }
        frameWidth = bestMatchWidth;
        frameHeight = bestMatchHeight;
        imageReader.reset(reader);

        ANativeWindow* window;
        mStatus = AImageReader_getWindow(imageReader.get(), &window);
        if (mStatus != AMEDIA_OK) {
            LOGE("Could not get ANativeWindow: %d", mStatus);
            return false;
        }
        nativeWindow.reset(window);
        ANativeWindow_acquire(nativeWindow.get());

        ACaptureSessionOutputContainer* container;
        cStatus = ACaptureSessionOutputContainer_create(&container);
        if (cStatus != ACAMERA_OK) {
            LOGE("CaptureSessionOutputContainer creation failed with error code: %d", cStatus);
            return false;
        }
        outputContainer.reset(container);

        ACaptureSessionOutput* output;
        cStatus = ACaptureSessionOutput_create(nativeWindow.get(), &output);
        if (cStatus != ACAMERA_OK) {
            LOGE("CaptureSessionOutput creation failed with error code: %d", cStatus);
            return false;
        }
        sessionOutput.reset(output);

        cStatus = ACaptureSessionOutputContainer_add(outputContainer.get(), sessionOutput.get());
        if (cStatus != ACAMERA_OK) {
            LOGE("CaptureSessionOutput Container add failed with error code: %d", cStatus);
            return false;
        }
        sessionOutputAdded = true;

        ACameraOutputTarget* target;
        cStatus = ACameraOutputTarget_create(nativeWindow.get(), &target);
        if (cStatus != ACAMERA_OK) {
            LOGE("CameraOutputTarget creation failed with error code: %d", cStatus);
            return false;
        }
        outputTarget.reset(target);

        ACaptureRequest* request;
        cStatus = ACameraDevice_createCaptureRequest(cameraDevice.get(), TEMPLATE_PREVIEW, &request);
        if (cStatus != ACAMERA_OK) {
            LOGE("CaptureRequest creation failed with error code: %d", cStatus);
            return false;
        }
        captureRequest.reset(request);

        cStatus = ACaptureRequest_addTarget(captureRequest.get(), outputTarget.get());
        if (cStatus != ACAMERA_OK) {
            LOGE("Add target to CaptureRequest failed with error code: %d", cStatus);
            return false;
        }
        targetAdded = true;

        ACameraCaptureSession* session;
        cStatus = ACameraDevice_createCaptureSession(cameraDevice.get(),
                                                     outputContainer.get(), &sessionCallbacks, &session);
        if (cStatus != ACAMERA_OK) {
            LOGE("CaptureSession creation failed with error code: %d", cStatus);
            return false;
        }
        captureSession.reset(session);

        ACaptureRequest_setEntry_u8(captureRequest.get(), ACAMERA_CONTROL_AE_MODE, 1, &aeMode);
        if (aeMode != ACAMERA_CONTROL_AE_MODE_ON) {
            ACaptureRequest_setEntry_i32(captureRequest.get(), ACAMERA_SENSOR_SENSITIVITY, 1, &sensitivity);
            ACaptureRequest_setEntry_i64(captureRequest.get(), ACAMERA_SENSOR_EXPOSURE_TIME, 1, &exposureTime);
        }
        if (zoomRange.isValid()) {
            ACaptureRequest_setEntry_float(captureRequest.get(), ACAMERA_CONTROL_ZOOM_RATIO, 1, &zoomRatio);
        }
        ACaptureRequest_setEntry_u8(captureRequest.get(), ACAMERA_FLASH_MODE, 1, &flashMode);

        cStatus = ACameraCaptureSession_setRepeatingRequest(captureSession.get(),
                                                            &captureCallbacks, 1, &request, nullptr);
        if (cStatus != ACAMERA_OK) {
            LOGE("CameraCaptureSession set repeating request failed with error code: %d", cStatus);
            return false;
        }
        return true;
    }

private:
    void getPropertyRanges(const ACameraMetadata* metadata)
    {
        camera_status_t cStatus;
        ACameraMetadata_const_entry val;

        cStatus = ACameraMetadata_getConstEntry(metadata, ACAMERA_SENSOR_INFO_EXPOSURE_TIME_RANGE, &val);
        if (cStatus == ACAMERA_OK) {
            exposureRange.min = exposureTimeLimits.clamp(val.data.i64[0]);
            exposureRange.max = exposureTimeLimits.clamp(val.data.i64[1]);
            exposureTime = exposureRange.percentage(2);
        } else {
            LOGW("Unsupported ACAMERA_SENSOR_INFO_EXPOSURE_TIME_RANGE");
            exposureRange.min = exposureRange.max = 0;
            exposureTime = 0;
        }

        cStatus = ACameraMetadata_getConstEntry(metadata, ACAMERA_SENSOR_INFO_SENSITIVITY_RANGE, &val);
        if (cStatus == ACAMERA_OK){
            sensitivityRange.min = val.data.i32[0];
            sensitivityRange.max = val.data.i32[1];
            sensitivity = sensitivityRange.percentage(2);
        } else {
            LOGW("Unsupported ACAMERA_SENSOR_INFO_SENSITIVITY_RANGE");
            sensitivityRange.min = sensitivityRange.max = 0;
            sensitivity = 0;
        }

        cStatus = ACameraMetadata_getConstEntry(metadata, ACAMERA_CONTROL_ZOOM_RATIO_RANGE, &val);
        if (cStatus == ACAMERA_OK){
            zoomRange.min = val.data.f[0];
            zoomRange.max = val.data.f[1];
            zoomRatio = zoomRange.clamp(zoomRatio);
        } else {
            LOGW("Unsupported ACAMERA_CONTROL_ZOOM_RATIO_RANGE");
            zoomRange.min = zoomRange.max = 0;
            zoomRatio = 1.0f;
        }
    }

    // calculate a score based on how well the width and height match the desired width and height
    // basically draw the 2 rectangle on top of each other and take the ratio of the non-overlapping
    // area to the overlapping area
    double getScore(int32_t width, int32_t height) const {
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

    void findResolutionMatch(const ACameraMetadata* metadata,
                             int32_t &bestMatchWidth, int32_t &bestMatchHeight) const {
        ACameraMetadata_const_entry entry = {};
        ACameraMetadata_getConstEntry(metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &entry);

        double bestScore = std::numeric_limits<double>::max();
        for (uint32_t i = 0; i < entry.count; i += 4) {
            int32_t input = entry.data.i32[i + 3];
            int32_t format = entry.data.i32[i + 0];

            if (!input && format == AIMAGE_FORMAT_YUV_420_888) {
                int32_t width = entry.data.i32[i + 1];
                int32_t height = entry.data.i32[i + 2];

                if (width == desiredWidth && height == desiredHeight) {
                    bestMatchWidth = width;
                    bestMatchHeight = height;
                    return;
                }

                double score = getScore(width, height);
                if (score < bestScore) {
                    bestMatchWidth = width;
                    bestMatchHeight = height;
                    bestScore = score;
                }
            }
        }
    }

    void setWidthHeight(SetupState newState) {
        if ((widthHeightState | newState) == (setupWidth | setupHeight)) {
            cleanUp();
            initCapture();
            newState = setupDone;
        }
        widthHeightState = newState;
    }

    void cleanUp() {
        if (sessionState == CaptureSessionState::ACTIVE) {
            ACameraCaptureSession_stopRepeating(captureSession.get());
        }
        captureSession.reset();
        if (targetAdded) {
            ACaptureRequest_removeTarget(captureRequest.get(), outputTarget.get());
            targetAdded = false;
        }
        captureRequest.reset();
        outputTarget.reset();
        if (sessionOutputAdded) {
            ACaptureSessionOutputContainer_remove(outputContainer.get(), sessionOutput.get());
            sessionOutputAdded = false;
        }
        sessionOutput.reset();
        nativeWindow.reset();
        outputContainer.reset();
        cameraDevice.reset();
        cameraManager.reset();
        imageReader.reset();
    }

    template<typename FuncT, typename T>
    bool submitRequest(FuncT setFn, uint32_t tag, const T &data)
    {
        ACaptureRequest *request = captureRequest.get();

        return request &&
               setFn(request, tag, 1, &data) == ACAMERA_OK &&
               ACameraCaptureSession_setRepeatingRequest(captureSession.get(),
                                                         &captureCallbacks,
                                                         1, &request, nullptr) == ACAMERA_OK;
    }
};

/********************************  Device management  *******************************/

void AndroidCameraCapture::OnDeviceDisconnect(void* /* ctx */, ACameraDevice* dev) {
    const char *id = ACameraDevice_getId(dev);
    LOGW("Device %s disconnected", id ? id : "<null>");
}

void AndroidCameraCapture::OnDeviceError(void* /* ctx */, ACameraDevice* dev, int err) {
    const char *id = ACameraDevice_getId(dev);
    LOGI("Camera Device Error: %#x, Device %s", err, id ? id : "<null>");

    switch (err) {
        case ERROR_CAMERA_IN_USE:
            LOGI("Camera in use");
            break;
        case ERROR_CAMERA_SERVICE:
            LOGI("Fatal Error occurred in Camera Service");
            break;
        case ERROR_CAMERA_DEVICE:
            LOGI("Fatal Error occurred in Camera Device");
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

/********************************  Session management  *******************************/

void AndroidCameraCapture::OnSessionClosed(void* context, ACameraCaptureSession* session) {
    if (context == nullptr) return;
    LOGW("session %p closed", session);
    static_cast<AndroidCameraCapture*>(context)->sessionState = CaptureSessionState::CLOSED;
}

void AndroidCameraCapture::OnSessionReady(void* context, ACameraCaptureSession* session) {
    if (context == nullptr) return;
    LOGW("session %p ready", session);
    static_cast<AndroidCameraCapture*>(context)->sessionState = CaptureSessionState::READY;
}

void AndroidCameraCapture::OnSessionActive(void* context, ACameraCaptureSession* session) {
    if (context == nullptr) return;
    LOGW("session %p active", session);
    static_cast<AndroidCameraCapture*>(context)->sessionState = CaptureSessionState::ACTIVE;
}

void AndroidCameraCapture::OnCaptureCompleted(void* context,
                                              ACameraCaptureSession* session,
                                              ACaptureRequest* /* request */,
                                              const ACameraMetadata* /* result */) {
    if (context == nullptr) return;
    LOGV("session %p capture completed", session);
    AndroidCameraCapture* cameraCapture = static_cast<AndroidCameraCapture*>(context);
    std::unique_lock<std::mutex> lock(cameraCapture->mtx);

    if (cameraCapture->waitingCapture) {
        cameraCapture->waitingCapture = false;
        cameraCapture->captureSuccess = true;
        cameraCapture->condition.notify_one();
    }
}

void AndroidCameraCapture::OnCaptureFailed(void* context,
                                           ACameraCaptureSession* session,
                                           ACaptureRequest* /* request */,
                                           ACameraCaptureFailure* /* failure */) {
    if (context == nullptr) return;
    LOGV("session %p capture failed", session);
    AndroidCameraCapture* cameraCapture = static_cast<AndroidCameraCapture*>(context);
    std::unique_lock<std::mutex> lock(cameraCapture->mtx);

    if (cameraCapture->waitingCapture) {
        cameraCapture->waitingCapture = false;
        cameraCapture->captureSuccess = false;
        cameraCapture->condition.notify_one();
    }
}

/****************** Implementation of interface functions ********************/

Ptr<IVideoCapture> cv::createAndroidCapture_cam(int index, const VideoCaptureParameters& params) {
    Ptr<AndroidCameraCapture> res = makePtr<AndroidCameraCapture>(index, params);
    if (res && res->initCapture())
        return res;
    return Ptr<IVideoCapture>();
}
