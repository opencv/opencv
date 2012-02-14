#include <jni.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <android/bitmap.h>

#include <android/log.h>
#define LOG_TAG "org.opencv.android.Utils"
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))
#ifdef DEBUG
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#else //!DEBUG
#define LOGD(...)
#endif //DEBUG

using namespace cv;


extern "C" {

/*
 * Class:     org_opencv_android_Utils
 * Method:    void nBitmapToMat(Bitmap b, long m_addr)
 */

JNIEXPORT void JNICALL Java_org_opencv_android_Utils_nBitmapToMat
  (JNIEnv * env, jclass cls, jobject bitmap, jlong m_addr)
{
    AndroidBitmapInfo  info;
    void*              pixels = 0;
    Mat&               dst = *((Mat*)m_addr);

    try {
            LOGD("nBitmapToMat");
            CV_Assert( AndroidBitmap_getInfo(env, bitmap, &info) >= 0 );
            CV_Assert( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                       info.format == ANDROID_BITMAP_FORMAT_RGB_565 );
            CV_Assert( AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0 );
            CV_Assert( pixels );
            dst.create(info.height, info.width, CV_8UC4);
            if( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 )
            {
                LOGD("nBitmapToMat: RGBA_8888 -> CV_8UC4");
                Mat tmp(info.height, info.width, CV_8UC4, pixels);
                tmp.copyTo(dst);
            } else {
                // info.format == ANDROID_BITMAP_FORMAT_RGB_565
                LOGD("nBitmapToMat: RGB_565 -> CV_8UC4");
                Mat tmp(info.height, info.width, CV_8UC2, pixels);
                cvtColor(tmp, dst, CV_BGR5652RGBA);
            }
            AndroidBitmap_unlockPixels(env, bitmap);
            return;
        } catch(cv::Exception e) {
        	AndroidBitmap_unlockPixels(env, bitmap);
            LOGE("nBitmapToMat catched cv::Exception: %s", e.what());
            jclass je = env->FindClass("org/opencv/core/CvException");
            if(!je) je = env->FindClass("java/lang/Exception");
            env->ThrowNew(je, e.what());
            return;
        } catch (...) {
        	AndroidBitmap_unlockPixels(env, bitmap);
            LOGE("nBitmapToMat catched unknown exception (...)");
            jclass je = env->FindClass("java/lang/Exception");
            env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
            return;
        }
}

/*
 * Class:     org_opencv_android_Utils
 * Method:    void nMatToBitmap(long m_addr, Bitmap b)
 */
JNIEXPORT void JNICALL Java_org_opencv_android_Utils_nMatToBitmap
  (JNIEnv * env, jclass cls, jlong m_addr, jobject bitmap)
{
    AndroidBitmapInfo  info;
    void*              pixels = 0;
    Mat&               dst = *((Mat*)m_addr);

    try {
            LOGD("nMatToBitmap");
            CV_Assert( AndroidBitmap_getInfo(env, bitmap, &info) >= 0 );
            CV_Assert( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                       info.format == ANDROID_BITMAP_FORMAT_RGB_565 );
            CV_Assert( dst.dims == 2 && info.height == (uint32_t)dst.rows && info.width == (uint32_t)dst.cols );
            CV_Assert( dst.type() == CV_8UC1 || dst.type() == CV_8UC3 || dst.type() == CV_8UC4 );
            CV_Assert( AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0 );
            CV_Assert( pixels );
            if( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 )
            {
                Mat tmp(info.height, info.width, CV_8UC4, pixels);
                if(dst.type() == CV_8UC1)
                {
                    LOGD("nMatToBitmap: CV_8UC1 -> RGBA_8888");
                    cvtColor(dst, tmp, CV_GRAY2RGBA);
                } else if(dst.type() == CV_8UC3){
                    LOGD("nMatToBitmap: CV_8UC3 -> RGBA_8888");
                    cvtColor(dst, tmp, CV_RGB2RGBA);
                } else if(dst.type() == CV_8UC4){
                    LOGD("nMatToBitmap: CV_8UC4 -> RGBA_8888");
                    dst.copyTo(tmp);
                }
            } else {
                // info.format == ANDROID_BITMAP_FORMAT_RGB_565
                Mat tmp(info.height, info.width, CV_8UC2, pixels);
                if(dst.type() == CV_8UC1)
                {
                    LOGD("nMatToBitmap: CV_8UC1 -> RGB_565");
                    cvtColor(dst, tmp, CV_GRAY2BGR565);
                } else if(dst.type() == CV_8UC3){
                    LOGD("nMatToBitmap: CV_8UC3 -> RGB_565");
                    cvtColor(dst, tmp, CV_RGB2BGR565);
                } else if(dst.type() == CV_8UC4){
                    LOGD("nMatToBitmap: CV_8UC4 -> RGB_565");
                    cvtColor(dst, tmp, CV_RGBA2BGR565);
                }
            }
            AndroidBitmap_unlockPixels(env, bitmap);
            return;
        } catch(cv::Exception e) {
        	AndroidBitmap_unlockPixels(env, bitmap);
            LOGE("nMatToBitmap catched cv::Exception: %s", e.what());
            jclass je = env->FindClass("org/opencv/core/CvException");
            if(!je) je = env->FindClass("java/lang/Exception");
            env->ThrowNew(je, e.what());
            return;
        } catch (...) {
        	AndroidBitmap_unlockPixels(env, bitmap);
            LOGE("nMatToBitmap catched unknown exception (...)");
            jclass je = env->FindClass("java/lang/Exception");
            env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
            return;
        }
}

} // extern "C"
