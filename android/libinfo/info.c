#include "opencv2/core/version.hpp"
#include <jni.h>

const char* GetPackageName(void);
const char* GetRevision(void);
const char* GetLibraryList(void);
JNIEXPORT jstring JNICALL Java_org_opencv_android_StaticHelper_getLibraryList(JNIEnv *, jclass);

#define PACKAGE_NAME "org.opencv.lib_v" CVAUX_STR(CV_VERSION_EPOCH) CVAUX_STR(CV_VERSION_MAJOR) "_" ANDROID_PACKAGE_PLATFORM
#define PACKAGE_REVISION CVAUX_STR(CV_VERSION_MINOR) "." CVAUX_STR(CV_VERSION_REVISION) "." CVAUX_STR(ANDROID_PACKAGE_RELEASE)

const char* GetPackageName(void)
{
    return PACKAGE_NAME;
}

const char* GetRevision(void)
{
    return PACKAGE_REVISION;
}

const char* GetLibraryList(void)
{
    return "";
}

JNIEXPORT jstring JNICALL Java_org_opencv_android_StaticHelper_getLibraryList(JNIEnv * env, jclass clazz)
{
    (void)clazz;
    return (*env)->NewStringUTF(env, GetLibraryList());
}
