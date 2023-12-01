#include <jni.h>
#include "CLprocessor.hpp"

extern "C" {
JNIEXPORT jboolean JNICALL Java_org_opencv_samples_tutorial4_NativePart_builtWithOpenCL(JNIEnv * env, jclass cls);

JNIEXPORT jint JNICALL Java_org_opencv_samples_tutorial4_NativePart_initCL(JNIEnv * env, jclass cls);

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial4_NativePart_closeCL(JNIEnv * env, jclass cls);

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial4_NativePart_processFrame(JNIEnv * env, jclass cls, jint tex1, jint tex2, jint w, jint h, jint mode);

JNIEXPORT jboolean JNICALL Java_org_opencv_samples_tutorial4_NativePart_builtWithOpenCL(JNIEnv * env, jclass cls)
{
#ifdef OPENCL_FOUND
    return JNI_TRUE;
#else
    return JNI_FALSE;
#endif
}

JNIEXPORT jint JNICALL Java_org_opencv_samples_tutorial4_NativePart_initCL(JNIEnv * env, jclass cls)
{
    return initCL();
}

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial4_NativePart_closeCL(JNIEnv * env, jclass cls)
{
    closeCL();
}

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial4_NativePart_processFrame(JNIEnv * env, jclass cls, jint tex1, jint tex2, jint w, jint h, jint mode)
{
    processFrame(tex1, tex2, w, h, mode);
}
} // extern "C"
