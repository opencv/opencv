typedef bool (*CameraCallback)(void* buffer, size_t bufferSize, void* userData);

typedef void* (*InitCameraConnectC)(void* cameraCallback, int cameraId, void* userData);
typedef void (*CloseCameraConnectC)(void**);
typedef double (*GetCameraPropertyC)(void* camera, int propIdx);
typedef void (*SetCameraPropertyC)(void* camera, int propIdx, double value);
typedef void (*ApplyCameraPropertiesC)(void** camera);

extern "C"
{
void* initCameraConnectC(void* cameraCallback, int cameraId, void* userData);
void closeCameraConnectC(void**);
double getCameraPropertyC(void* camera, int propIdx);
void setCameraPropertyC(void* camera, int propIdx, double value);
void applyCameraPropertiesC(void** camera);
}
