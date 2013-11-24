#include "HardwareDetector_jni.h"
#include "HardwareDetector.h"
#include <jni.h>
#include <string>

JNIEXPORT jint JNICALL Java_org_opencv_engine_HardwareDetector_GetCpuID(JNIEnv* , jclass)
{
    return GetCpuID();
}

JNIEXPORT jstring JNICALL Java_org_opencv_engine_HardwareDetector_GetPlatformName(JNIEnv* env, jclass)
{
    std::string hardware_name = GetPlatformName();
    return env->NewStringUTF(hardware_name.c_str());
}

JNIEXPORT jint JNICALL Java_org_opencv_engine_HardwareDetector_GetProcessorCount(JNIEnv* , jclass)
{
    return GetProcessorCount();
}

JNIEXPORT jint JNICALL Java_org_opencv_engine_HardwareDetector_DetectKnownPlatforms(JNIEnv* , jclass)
{
    return DetectKnownPlatforms();
}
