#include "OpenCVLibraryInfo.h"
#include "EngineCommon.h"
#include <utils/Log.h>
#include <dlfcn.h>

JNIEXPORT jlong JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_open
  (JNIEnv * env, jobject, jstring str)
{
    const char* infoLibPath = env->GetStringUTFChars(str, NULL);
    if (infoLibPath == NULL)
        return 0;

    LOGD("Trying to load info library \"%s\"", infoLibPath);

    void* handle;

    handle = dlopen(infoLibPath, RTLD_LAZY);
    if (handle == NULL)
        LOGI("Info library not found by path \"%s\"", infoLibPath);

    return (jlong)handle;
}

JNIEXPORT jstring JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_getPackageName
  (JNIEnv* env, jobject, jlong handle)
{
    InfoFunctionType info_func;
    const char* result;
    const char* error;

    dlerror();
    info_func = (InfoFunctionType)dlsym((void*)handle, "GetPackageName");
    if ((error = dlerror()) == NULL)
        result = (*info_func)();
    else
    {
        LOGE("dlsym error: \"%s\"", error);
        result = "unknown";
    }

    return env->NewStringUTF(result);
}

JNIEXPORT jstring JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_getLibraryList
  (JNIEnv* env, jobject, jlong handle)
{
    InfoFunctionType info_func;
    const char* result;
    const char* error;

    dlerror();
    info_func = (InfoFunctionType)dlsym((void*)handle, "GetLibraryList");
    if ((error = dlerror()) == NULL)
        result = (*info_func)();
    else
    {
        LOGE("dlsym error: \"%s\"", error);
        result = "unknown";
    }

    return env->NewStringUTF(result);
}

JNIEXPORT jstring JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_getVersionName
  (JNIEnv* env, jobject, jlong handle)
{
    InfoFunctionType info_func;
    const char* result;
    const char* error;

    dlerror();
    info_func = (InfoFunctionType)dlsym((void*)handle, "GetRevision");
    if ((error = dlerror()) == NULL)
        result = (*info_func)();
    else
    {
        LOGE("dlsym error: \"%s\"", error);
        result = "unknown";
    }

    return env->NewStringUTF(result);
}

JNIEXPORT void JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_close
  (JNIEnv*, jobject, jlong handle)
{
    dlclose((void*)handle);
}
