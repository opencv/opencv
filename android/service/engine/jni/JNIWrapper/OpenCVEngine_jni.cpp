#include "OpenCVEngine_jni.h"
#include "EngineCommon.h"
#include "IOpenCVEngine.h"
#include "OpenCVEngine.h"
#include "IPackageManager.h"
#include "JavaBasedPackageManager.h"
#include <utils/Log.h>
#include <android_util_Binder.h>

#undef LOG_TAG
#define LOG_TAG "OpenCVEngine/JNI"

using namespace android;

sp<IBinder> OpenCVEngineBinder = NULL;
IPackageManager* PackageManager = NULL;

JNIEXPORT jobject JNICALL Java_org_opencv_engine_BinderConnector_Connect(JNIEnv* env, jobject)
{
    LOGI("Creating new component");
    if (NULL != OpenCVEngineBinder.get())
    {
        LOGI("New component created successfully");
    }
    else
    {
        LOGE("OpenCV Engine component was not created!");
    }

    return javaObjectForIBinder(env, OpenCVEngineBinder);
}

JNIEXPORT jboolean JNICALL Java_org_opencv_engine_BinderConnector_Init(JNIEnv* env, jobject , jobject market)
{
    LOGD("Java_org_opencv_engine_BinderConnector_Init");

    if (NULL == PackageManager)
    {
        JavaVM* jvm;
        env->GetJavaVM(&jvm);
        PackageManager = new JavaBasedPackageManager(jvm, env->NewGlobalRef(market));
    }
    if (PackageManager)
    {
        if (!OpenCVEngineBinder.get())
        {
            OpenCVEngineBinder = new OpenCVEngine(PackageManager);
            return (NULL != OpenCVEngineBinder.get());
        }
        else
        {
            return true;
        }
    }
    else
    {
        return false;
    }
}

JNIEXPORT void JNICALL Java_org_opencv_engine_BinderConnector_Final(JNIEnv *, jobject)
{
    LOGD("Java_org_opencv_engine_BinderConnector_Final");

    OpenCVEngineBinder = NULL;

    delete PackageManager;
    PackageManager = NULL;
}
