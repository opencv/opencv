#include <dlfcn.h>
#include <android/log.h>
#include <string>
#include <vector>
#include "camera_activity.hpp"
#include "camera_wrapper.h"

#define LOG_TAG "CAMERA_ACTIVITY"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

///////
// Debug
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>


using namespace std;

class CameraWrapperConnector
{
public:
    static CameraActivity::ErrorCode connect(int cameraId, CameraActivity* pCameraActivity, void** camera);
    static CameraActivity::ErrorCode disconnect(void** camera);
    static CameraActivity::ErrorCode setProperty(void* camera, int propIdx, double value);
    static CameraActivity::ErrorCode getProperty(void* camera, int propIdx, double* value);
    static CameraActivity::ErrorCode applyProperties(void** ppcamera);

    static void setPathLibFolder(const std::string& path);

private:
    static std::string pathLibFolder;
    static bool isConnectedToLib;

    static std::string getPathLibFolder();
    static CameraActivity::ErrorCode connectToLib();
    static CameraActivity::ErrorCode getSymbolFromLib(void * libHandle, const char* symbolName, void** ppSymbol);
    static void fillListWrapperLibs(const string& folderPath, vector<string>& listLibs);

    static InitCameraConnectC pInitCameraC;
    static CloseCameraConnectC pCloseCameraC;
    static GetCameraPropertyC pGetPropertyC;
    static SetCameraPropertyC pSetPropertyC;
    static ApplyCameraPropertiesC pApplyPropertiesC;

    friend bool nextFrame(void* buffer, size_t bufferSize, void* userData);
};

std::string CameraWrapperConnector::pathLibFolder;
#define DEFAULT_WRAPPER_PACKAGE_NAME "com.NativeCamera"
#define DEFAULT_PATH_LIB_FOLDER "/data/data/" DEFAULT_WRAPPER_PACKAGE_NAME "/lib/"

bool CameraWrapperConnector::isConnectedToLib=false;
InitCameraConnectC  CameraWrapperConnector::pInitCameraC = 0;
CloseCameraConnectC  CameraWrapperConnector::pCloseCameraC = 0;
GetCameraPropertyC CameraWrapperConnector::pGetPropertyC = 0;
SetCameraPropertyC CameraWrapperConnector::pSetPropertyC = 0;
ApplyCameraPropertiesC CameraWrapperConnector::pApplyPropertiesC = 0;

#define INIT_CAMERA_SYMBOL_NAME "initCameraConnectC"
#define CLOSE_CAMERA_SYMBOL_NAME "closeCameraConnectC"
#define SET_CAMERA_PROPERTY_SYMBOL_NAME "setCameraPropertyC"
#define GET_CAMERA_PROPERTY_SYMBOL_NAME "getCameraPropertyC"
#define APPLY_CAMERA_PROPERTIES_SYMBOL_NAME "applyCameraPropertiesC"
#define PREFIX_CAMERA_WRAPPER_LIB "libnative_camera"


bool nextFrame(void* buffer, size_t bufferSize, void* userData)
{
    if (userData == NULL)
        return true;

    return ((CameraActivity*)userData)->onFrameBuffer(buffer, bufferSize);
}

CameraActivity::ErrorCode CameraWrapperConnector::connect(int cameraId, CameraActivity* pCameraActivity, void** camera)
{
    if (pCameraActivity == NULL)
    {
        LOGE("CameraWrapperConnector::connect error: wrong pointer to CameraActivity object");
        return CameraActivity::ERROR_WRONG_POINTER_CAMERA_WRAPPER;
    }

    CameraActivity::ErrorCode errcode=connectToLib();
    if (errcode) return errcode;

    void* cmr = (*pInitCameraC)((void*)nextFrame, cameraId, (void*)pCameraActivity);
    if (!cmr)
    {
        LOGE("CameraWrapperConnector::connectWrapper ERROR: the initializing function returned false");
        return CameraActivity::ERROR_CANNOT_INITIALIZE_CONNECTION;
    }

    *camera = cmr;
    return CameraActivity::NO_ERROR;
}

CameraActivity::ErrorCode CameraWrapperConnector::disconnect(void** camera)
{
    if (camera == NULL || *camera == NULL)
    {
        LOGE("CameraWrapperConnector::disconnect error: wrong pointer to camera object");
        return CameraActivity::ERROR_WRONG_POINTER_CAMERA_WRAPPER;
    }

    CameraActivity::ErrorCode errcode=connectToLib();
    if (errcode) return errcode;

    (*pCloseCameraC)(camera);

    return CameraActivity::NO_ERROR;
}

CameraActivity::ErrorCode CameraWrapperConnector::setProperty(void* camera, int propIdx, double value)
{
    if (camera == NULL)
    {
        LOGE("CameraWrapperConnector::setProperty error: wrong pointer to camera object");
        return CameraActivity::ERROR_WRONG_POINTER_CAMERA_WRAPPER;
    }

    (*pSetPropertyC)(camera, propIdx, value);

    return CameraActivity::NO_ERROR;
}

CameraActivity::ErrorCode CameraWrapperConnector::getProperty(void* camera, int propIdx, double* value)
{
    if (camera == NULL)
    {
        LOGE("CameraWrapperConnector::getProperty error: wrong pointer to camera object");
        return CameraActivity::ERROR_WRONG_POINTER_CAMERA_WRAPPER;
    }

    *value = (*pGetPropertyC)(camera, propIdx);
    return CameraActivity::NO_ERROR;
}

CameraActivity::ErrorCode CameraWrapperConnector::applyProperties(void** ppcamera)
{
    if ((ppcamera == NULL) || (*ppcamera == NULL))
    {
        LOGE("CameraWrapperConnector::applyProperties error: wrong pointer to camera object");
        return CameraActivity::ERROR_WRONG_POINTER_CAMERA_WRAPPER;
    }

    (*pApplyPropertiesC)(ppcamera);
    return CameraActivity::NO_ERROR;
}

CameraActivity::ErrorCode CameraWrapperConnector::connectToLib()
{
    if (isConnectedToLib) {
        return CameraActivity::NO_ERROR;
    }

    dlerror();
    string folderPath=getPathLibFolder();
    LOGD("CameraWrapperConnector::connectToLib: folderPath=%s", folderPath.c_str());

    vector<string> listLibs;
    fillListWrapperLibs(folderPath, listLibs);

    void * libHandle=0;
    string cur_path;
    for(size_t i=0; i < listLibs.size(); i++) {
        cur_path=folderPath + listLibs[i];
        LOGD("try to load library '%s'", listLibs[i].c_str());
        libHandle=dlopen(cur_path.c_str(), RTLD_LAZY);
        if (libHandle) {
            LOGD("Loaded library '%s'", cur_path.c_str());
            break;
        } else {
            LOGD("CameraWrapperConnector::connectToLib ERROR: cannot dlopen camera wrapper library %s, dlerror=\"%s\"",
                 cur_path.c_str(), dlerror());
        }
    }

    if (!libHandle) {
        LOGE("CameraWrapperConnector::connectToLib ERROR: cannot dlopen camera wrapper library");
        return CameraActivity::ERROR_CANNOT_OPEN_CAMERA_WRAPPER_LIB;
    }

    InitCameraConnectC pInit_C;
    CloseCameraConnectC pClose_C;
    GetCameraPropertyC pGetProp_C;
    SetCameraPropertyC pSetProp_C;
    ApplyCameraPropertiesC pApplyProp_C;

    CameraActivity::ErrorCode res;

    res = getSymbolFromLib(libHandle, (const char*)INIT_CAMERA_SYMBOL_NAME, (void**)(&pInit_C));
    if (res) return res;

    res = getSymbolFromLib(libHandle, CLOSE_CAMERA_SYMBOL_NAME, (void**)(&pClose_C));
    if (res) return res;

    res = getSymbolFromLib(libHandle, GET_CAMERA_PROPERTY_SYMBOL_NAME, (void**)(&pGetProp_C));
    if (res) return res;

    res = getSymbolFromLib(libHandle, SET_CAMERA_PROPERTY_SYMBOL_NAME, (void**)(&pSetProp_C));
    if (res) return res;
    \
    res = getSymbolFromLib(libHandle, APPLY_CAMERA_PROPERTIES_SYMBOL_NAME, (void**)(&pApplyProp_C));
    if (res) return res;

    pInitCameraC  = pInit_C;
    pCloseCameraC = pClose_C;
    pGetPropertyC = pGetProp_C;
    pSetPropertyC = pSetProp_C;
    pApplyPropertiesC = pApplyProp_C;
    isConnectedToLib=true;

    return CameraActivity::NO_ERROR;
}

CameraActivity::ErrorCode CameraWrapperConnector::getSymbolFromLib(void* libHandle, const char* symbolName, void** ppSymbol)
{
    dlerror();
    *(void **) (ppSymbol)=dlsym(libHandle, symbolName);

    const char* error_dlsym_init=dlerror();
    if (error_dlsym_init) {
        LOGE("CameraWrapperConnector::getSymbolFromLib ERROR: cannot get symbol of the function '%s' from the camera wrapper library, dlerror=\"%s\"",
             symbolName, error_dlsym_init);
        return CameraActivity::ERROR_CANNOT_GET_FUNCTION_FROM_CAMERA_WRAPPER_LIB;
    }
    return CameraActivity::NO_ERROR;
}

void CameraWrapperConnector::fillListWrapperLibs(const string& folderPath, vector<string>& listLibs)
{
    DIR *dp;
    struct dirent *ep;

    dp = opendir (folderPath.c_str());
    if (dp != NULL)
    {
        while ((ep = readdir (dp))) {
            const char* cur_name=ep->d_name;
            if (strstr(cur_name, PREFIX_CAMERA_WRAPPER_LIB)) {
                listLibs.push_back(cur_name);
                LOGE("||%s", cur_name);
            }
        }
        (void) closedir (dp);
    }
}

std::string CameraWrapperConnector::getPathLibFolder()
{
    if (!pathLibFolder.empty())
        return pathLibFolder;

    Dl_info dl_info;
    if(0 != dladdr((void *)nextFrame, &dl_info))
    {
        LOGD("Library name: %s", dl_info.dli_fname);
        LOGD("Library base address: %p", dl_info.dli_fbase);

        char addrBuf[18];
        sprintf(addrBuf, "%p-", dl_info.dli_fbase);
        int addrLength = strlen(addrBuf);

        char lineBuf[2048];
        FILE* file = fopen("/proc/self/smaps", "rt");

        if(file)
        {
	    while (fgets(lineBuf, sizeof lineBuf, file) != NULL)
	    {
                if(0 == strncmp(lineBuf, addrBuf, addrLength))
                {
                    //verify that line ends with library name
                    int lineLength = strlen(lineBuf);
                    int libNameLength = strlen(dl_info.dli_fname);

                    //trim end
                    for(int i = lineLength - 1; i >= 0 && isspace(lineBuf[i]); --i)
                    {
                        lineBuf[i] = 0;
                        --lineLength;
                    }

                    if (0 != strncmp(lineBuf + lineLength - libNameLength, dl_info.dli_fname, libNameLength))
                    {
                        LOGE("Strange error: line \"%s\" does not ends with library name %s", lineBuf, dl_info.dli_fname);
                        continue;
                    }

                    //extract path from smaps line
                    char* pathBegin = strchr(lineBuf, '/');
                    if (0 == pathBegin)
                    {
                        LOGE("Strange error: could not find path beginning in lin \"%s\"", lineBuf);
                        continue;
                    }

                    char* pathEnd = strrchr(pathBegin, '/');
                    pathEnd[1] = 0;

                    LOGD("Libraries folder found: %s", pathBegin);

                    fclose(file);
                    return pathBegin;
                }
	    }
	    fclose(file);
	    LOGE("Could not find library path.");
        }
        else
        {
	    LOGE("Could not read /proc/self/smaps");
        }
    }
    else
    {
        LOGE("Could not get library name and base address.");
    }

    return DEFAULT_PATH_LIB_FOLDER ;
}

void CameraWrapperConnector::setPathLibFolder(const string& path)
{
    pathLibFolder=path;
}


/////////////////////////////////////////////////////////////////////////////////////////////////

CameraActivity::CameraActivity() : camera(0), frameWidth(-1), frameHeight(-1)
{
}

CameraActivity::~CameraActivity()
{
    if (camera != 0)
        disconnect();
}

bool CameraActivity::onFrameBuffer(void* buffer, int bufferSize)
{
    LOGD("CameraActivity::onFrameBuffer - empty callback");
    return true;
}

void CameraActivity::disconnect()
{
    CameraWrapperConnector::disconnect(&camera);
}

bool CameraActivity::isConnected() const
{
    return camera != 0;
}

CameraActivity::ErrorCode CameraActivity::connect(int cameraId)
{
    ErrorCode rescode = CameraWrapperConnector::connect(cameraId, this, &camera);
    if (rescode) return rescode;

    return NO_ERROR;
}

double CameraActivity::getProperty(int propIdx)
{
    double propVal;
    ErrorCode rescode = CameraWrapperConnector::getProperty(camera, propIdx, &propVal);
    if (rescode) return -1;
    return propVal;
}

void CameraActivity::setProperty(int propIdx, double value)
{
    CameraWrapperConnector::setProperty(camera, propIdx, value);
}

void CameraActivity::applyProperties()
{
    frameWidth = -1;
    frameHeight = -1;
    CameraWrapperConnector::applyProperties(&camera);
}

int CameraActivity::getFrameWidth()
{
    if (frameWidth < 0)
        frameWidth = getProperty(ANDROID_CAMERA_PROPERTY_FRAMEWIDTH);
    return frameWidth;
}

int CameraActivity::getFrameHeight()
{
    if (frameHeight < 0)
        frameHeight = getProperty(ANDROID_CAMERA_PROPERTY_FRAMEHEIGHT);
    return frameHeight;
}

void CameraActivity::setPathLibFolder(const char* path)
{
    CameraWrapperConnector::setPathLibFolder(path);
}
