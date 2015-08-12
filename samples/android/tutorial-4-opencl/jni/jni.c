#include <jni.h>

int initGL();
void closeGL();
void changeSize(int width, int height);
void drawFrame();
void setProcessingMode(int mode);

JNIEXPORT jint JNICALL Java_org_opencv_samples_tutorial4_NativeGLRenderer_initGL(JNIEnv * env, jclass cls)
{
    return initGL();
}

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial4_NativeGLRenderer_closeGL(JNIEnv * env, jclass cls)
{
    closeGL();
}

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial4_NativeGLRenderer_changeSize(JNIEnv * env, jclass cls, jint width, jint height)
{
    changeSize(width, height);
}

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial4_NativeGLRenderer_drawFrame(JNIEnv * env, jclass cls)
{
    drawFrame();
}

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial4_NativeGLRenderer_setProcessingMode(JNIEnv * env, jclass cls, jint mode)
{
    setProcessingMode(mode);
}
