#include "info.h"
#include <jni.h>

#ifndef LIB_STRING
    #define LIB_STRING "libtbb.so;libopencv_java.so"
#endif

const char* GetLibraryList()
{
    return LIB_STRING;
}

JNIEXPORT jstring JNICALL Java_org_opencv_android_StaticHelper_getLibraryList(JNIEnv* jenv, jclass clazz)
{
    jstring result = (*jenv)->NewStringUTF(jenv, LIB_STRING);

    return result;
}