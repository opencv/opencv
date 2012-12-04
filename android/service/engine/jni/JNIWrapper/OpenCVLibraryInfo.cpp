#include "OpenCVLibraryInfo.h"

JNIEXPORT jlong JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_open
  (JNIEnv * env, jobject, jstring)
{
    return 255;
}

JNIEXPORT jstring JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_getPackageName
  (JNIEnv* env, jobject, jlong)
{
    return env->NewStringUTF("org.opencv.lib_v24_tegra3");
}

JNIEXPORT jstring JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_getPublicName
(JNIEnv* env, jobject, jlong)
{
    return env->NewStringUTF("OpenCV library for Tegra3");
}

JNIEXPORT jstring JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_getLibraryList
  (JNIEnv* env, jobject, jlong)
{
    return env->NewStringUTF("");
}

JNIEXPORT jstring JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_getVersionName
  (JNIEnv* env, jobject, jlong)
{
    return env->NewStringUTF("9.9");
}

JNIEXPORT void JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_close
  (JNIEnv* env, jobject, jlong)
{

}
