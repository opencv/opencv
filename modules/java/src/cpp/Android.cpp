#include <jni.h>

#include "opencv2/core/core.hpp"

#include <android/bitmap.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_opencv_Android
 * Method:    nBitmapToMat(Bitmap b)
 * Signature: (L)J
 */

JNIEXPORT jlong JNICALL Java_org_opencv_Android_nBitmapToMat
  (JNIEnv * env, jclass cls, jobject bitmap)
{
    AndroidBitmapInfo  info;
    void*              pixels;
    cv::Mat*           m = new cv::Mat();

	if ( AndroidBitmap_getInfo(env, bitmap, &info) < 0 )
        return (jlong)m; // can't get info

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return (jlong)m; // incompatible format

    if ( AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0 )
        return (jlong)m; // can't get pixels

    m->create(info.height, info.width, CV_8UC4);
    if(m->data && pixels)
        memcpy(m->data, pixels, info.height * info.width * 4);

    AndroidBitmap_unlockPixels(env, bitmap);

    return (jlong)m;
}

/*
 * Class:     org_opencv_Android
 * Method:    nBitmapToMat(long m, Bitmap b)
 * Signature: (JL)Z
 */
JNIEXPORT jboolean JNICALL Java_org_opencv_Android_nMatToBitmap
  (JNIEnv * env, jclass cls, jlong m, jobject bitmap)
{
    AndroidBitmapInfo  info;
    void*              pixels;
    cv::Mat* mat = (cv::Mat*) m;

    if ( mat == 0 || mat->data == 0)
        return false; // no native Mat behind

    if ( AndroidBitmap_getInfo(env, bitmap, &info) < 0 )
        return false; // can't get info

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return false; // incompatible format

    if ( AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0 )
        return false; // can't get pixels

    if(mat->data && pixels)
        memcpy(pixels, mat->data, info.height * info.width * 4);

    AndroidBitmap_unlockPixels(env, bitmap);

    return true;
}

#ifdef __cplusplus
}
#endif
