#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_opencv_utils
 * Method:    nBitmapToMat(Bitmap b)
 * Signature: (L)J
 */
JNIEXPORT jlong JNICALL Java_org_opencv_utils_nBitmapToMat
  (JNIEnv *, jclass, jobject);

/*
 * Class:     org_opencv_utils
 * Method:    nBitmapToMat(long m, Bitmap b)
 * Signature: (JL)Z
 */
JNIEXPORT jboolean JNICALL Java_org_opencv_utils_nMatToBitmap
  (JNIEnv *, jclass, jlong, jobject);



#ifdef __cplusplus
}
#endif

#include "opencv2/core/core.hpp"

#include <android/bitmap.h>

JNIEXPORT jlong JNICALL Java_org_opencv_utils_nBitmapToMat
  (JNIEnv * env, jclass cls, jobject bitmap)
{
    AndroidBitmapInfo  info;
    void*              pixels;
	cv::Mat* m = NULL;

	if ( AndroidBitmap_getInfo(env, bitmap, &info) < 0 )
        return 0; // can't get info

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return 0; // incompatible format

    if ( AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0 )
        return 0; // can't get pixels

	m = new cv::Mat(info.height, info.width, CV_8UC4);
	memcpy(m->data, pixels, info.height * info.width * 4);

    AndroidBitmap_unlockPixels(env, bitmap);

	return (jlong)m;
}

JNIEXPORT jboolean JNICALL Java_org_opencv_utils_nMatToBitmap
  (JNIEnv * env, jclass cls, jlong m, jobject bitmap)
{
    AndroidBitmapInfo  info;
    void*              pixels;
	cv::Mat* mat = (cv::Mat*) m;

	if ( m == 0 )
		return false; // no native Mat behind

	if ( AndroidBitmap_getInfo(env, bitmap, &info) < 0 )
        return false; // can't get info

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return false; // incompatible format

    if ( AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0 )
        return false; // can't get pixels

	memcpy(pixels, mat->data, info.height * info.width * 4);

    AndroidBitmap_unlockPixels(env, bitmap);

	return true;
}
