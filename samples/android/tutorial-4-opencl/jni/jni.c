#include <jni.h>

int initCL();
void closeCL();
void processFrame(int tex1, int tex2, int w, int h, int mode);

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
