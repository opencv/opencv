// emulating the 'old' JNI names existed before the VideoCapture wrapping became automatic

#define LOG_TAG "org.opencv.highgui.VideoCapture"
#include "common.h"

#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_HIGHGUI

#include "opencv2/core/version.hpp"

#if (CV_VERSION_EPOCH == 2) && (CV_VERSION_MAJOR == 4)
extern "C" {

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__
  (JNIEnv* env, jclass c);

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_VideoCapture_10 (JNIEnv*, jclass);

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__
  (JNIEnv* env, jclass c)
{
    return Java_org_opencv_highgui_VideoCapture_VideoCapture_10(env, c);
}


JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__I
  (JNIEnv* env, jclass c, jint device);

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_VideoCapture_12 (JNIEnv*, jclass, jint);

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__I
  (JNIEnv* env, jclass c, jint device)
{
    return Java_org_opencv_highgui_VideoCapture_VideoCapture_12(env, c, device);
}


JNIEXPORT jdouble JNICALL Java_org_opencv_highgui_VideoCapture_n_1get
  (JNIEnv* env, jclass c, jlong self, jint propId);

JNIEXPORT jdouble JNICALL Java_org_opencv_highgui_VideoCapture_get_10 (JNIEnv*, jclass, jlong, jint);

JNIEXPORT jdouble JNICALL Java_org_opencv_highgui_VideoCapture_n_1get
  (JNIEnv* env, jclass c, jlong self, jint propId)
{
    return Java_org_opencv_highgui_VideoCapture_get_10(env, c, self, propId);
}


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1grab
  (JNIEnv* env, jclass c, jlong self);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_grab_10 (JNIEnv*, jclass, jlong);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1grab
  (JNIEnv* env, jclass c, jlong self)
{
    return Java_org_opencv_highgui_VideoCapture_grab_10(env, c, self);
}


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1isOpened
  (JNIEnv* env, jclass c, jlong self);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_isOpened_10 (JNIEnv*, jclass, jlong);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1isOpened
  (JNIEnv* env, jclass c, jlong self)
{
    return Java_org_opencv_highgui_VideoCapture_isOpened_10(env, c, self);
}


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1open__JI
  (JNIEnv* env, jclass c, jlong self, jint device);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_open_11 (JNIEnv*, jclass, jlong, jint);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1open__JI
  (JNIEnv* env, jclass c, jlong self, jint device)
{
    return Java_org_opencv_highgui_VideoCapture_open_11(env, c, self, device);
}


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1read
  (JNIEnv* env, jclass c, jlong self, jlong image_nativeObj);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_read_10 (JNIEnv*, jclass, jlong, jlong);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1read
  (JNIEnv* env, jclass c, jlong self, jlong image_nativeObj)
{
    return Java_org_opencv_highgui_VideoCapture_read_10(env, c, self, image_nativeObj);
}


JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_n_1release
  (JNIEnv* env, jclass c, jlong self);

JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_release_10 (JNIEnv*, jclass, jlong);

JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_n_1release
  (JNIEnv* env, jclass c, jlong self)
{
    Java_org_opencv_highgui_VideoCapture_release_10(env, c, self);
}


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJI
  (JNIEnv* env, jclass c, jlong self, jlong image_nativeObj, jint channel);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_retrieve_10 (JNIEnv*, jclass, jlong, jlong, jint);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJI
  (JNIEnv* env, jclass c, jlong self, jlong image_nativeObj, jint channel)
{
    return Java_org_opencv_highgui_VideoCapture_retrieve_10(env, c, self, image_nativeObj, channel);
}


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJ
  (JNIEnv* env, jclass c, jlong self, jlong image_nativeObj);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_retrieve_11 (JNIEnv*, jclass, jlong, jlong);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJ
  (JNIEnv* env, jclass c, jlong self, jlong image_nativeObj)
{
    return Java_org_opencv_highgui_VideoCapture_retrieve_11(env, c, self, image_nativeObj);
}


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1set
  (JNIEnv* env, jclass c, jlong self, jint propId, jdouble value);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_set_10 (JNIEnv*, jclass, jlong, jint, jdouble);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1set
  (JNIEnv* env, jclass c, jlong self, jint propId, jdouble value)
{
    return Java_org_opencv_highgui_VideoCapture_set_10(env, c, self, propId, value);
}


JNIEXPORT jstring JNICALL Java_org_opencv_highgui_VideoCapture_n_1getSupportedPreviewSizes
  (JNIEnv *env, jclass c, jlong self);

JNIEXPORT jstring JNICALL Java_org_opencv_highgui_VideoCapture_getSupportedPreviewSizes_10
  (JNIEnv *env, jclass, jlong self);

JNIEXPORT jstring JNICALL Java_org_opencv_highgui_VideoCapture_n_1getSupportedPreviewSizes
  (JNIEnv *env, jclass c, jlong self)
{
    return Java_org_opencv_highgui_VideoCapture_getSupportedPreviewSizes_10(env, c, self);
}


JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_n_1delete
  (JNIEnv *env, jclass c, jlong self);

JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_delete(JNIEnv*, jclass, jlong);

JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_n_1delete
  (JNIEnv *env, jclass c, jlong self)
{
    Java_org_opencv_highgui_VideoCapture_delete(env, c, self);
}


} // extern "C"
#endif // (CV_VERSION_EPOCH == 2) && (CV_VERSION_MAJOR == 4)
#endif // HAVE_OPENCV_HIGHGUI
