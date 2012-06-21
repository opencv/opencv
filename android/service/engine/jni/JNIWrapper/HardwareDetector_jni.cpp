#include "HardwareDetector_jni.h"
#include "HardwareDetector.h"
#include <jni.h>
#include <string>

JNIEXPORT jint JNICALL Java_org_opencv_engine_manager_HardwareDetector_GetCpuID(JNIEnv* env, jclass)
{
    return GetCpuID();
}

JNIEXPORT jstring JNICALL Java_org_opencv_engine_manager_HardwareDetector_GetPlatformName(JNIEnv* env, jclass)
{
    std::string hardware_name = GetPlatformName();
    return env->NewStringUTF(hardware_name.c_str());
}

JNIEXPORT jint JNICALL Java_org_opencv_engine_manager_HardwareDetector_GetProcessorCount(JNIEnv* env, jclass)
{
    return GetProcessorCount();
}

JNIEXPORT jint JNICALL Java_org_opencv_engine_manager_HardwareDetector_DetectKnownPlatforms(JNIEnv* env, jclass)
{
    return DetectKnownPlatforms();
}