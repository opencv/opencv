#ifndef _CAMERAACTIVITY_H_
#define _CAMERAACTIVITY_H_

#include <camera_properties.h>
//#include <opencv2/core/core.hpp>

class CameraActivity
{
public:
    enum ErrorCode {
        NO_ERROR=0,
        ERROR_WRONG_FRAME_SIZE,
        ERROR_WRONG_POINTER_CAMERA_WRAPPER,
        ERROR_CAMERA_CONNECTED,
        ERROR_CANNOT_OPEN_CAMERA_WRAPPER_LIB,
        ERROR_CANNOT_GET_FUNCTION_FROM_CAMERA_WRAPPER_LIB,
        ERROR_CANNOT_INITIALIZE_CONNECTION,
        ERROR_ISNT_CONNECTED,
        ERROR_JAVA_VM_CANNOT_GET_CLASS,
        ERROR_JAVA_VM_CANNOT_GET_FIELD,
        ERROR_CANNOT_SET_PREVIEW_DISPLAY,

        ERROR_UNKNOWN=255
    };

    CameraActivity();
    virtual ~CameraActivity();
    virtual bool onFrameBuffer(void* buffer, int bufferSize);

    ErrorCode connect(int cameraId = 0);
    void disconnect();
    bool isConnected() const;

    double getProperty(int propIdx);
    void setProperty(int propIdx, double value);
    void applyProperties();

    int getFrameWidth();
    int getFrameHeight();

    static void setPathLibFolder(const char* path);
private:
    void* camera;
    int frameWidth;
    int frameHeight;
};

#endif
