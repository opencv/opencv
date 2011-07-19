//
// This file is auto-generated, please don't edit!
//

#include <jni.h>

#ifdef DEBUG
#include <android/log.h>
#define MODULE_LOG_TAG "OpenCV.highgui"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, MODULE_LOG_TAG, __VA_ARGS__))
#endif // DEBUG

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;


extern "C" {

//
//   VideoCapture::VideoCapture()
//


JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__
  (JNIEnv* env, jclass cls)
{
    try {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1VideoCapture__()");
#endif // DEBUG
        
        VideoCapture* _retval_ = new VideoCapture(  );
        
        return (jlong) _retval_;
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1VideoCapture__() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1VideoCapture__() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1VideoCapture__()}");
        return 0;
    }
}


//
//   VideoCapture::VideoCapture(int device)
//


JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__I
  (JNIEnv* env, jclass cls, jint device)
{
    try {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1VideoCapture__I()");
#endif // DEBUG
        
        VideoCapture* _retval_ = new VideoCapture( device );
        
        return (jlong) _retval_;
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1VideoCapture__I() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1VideoCapture__I() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1VideoCapture__I()}");
        return 0;
    }
}



//
//  double VideoCapture::get(int propId)
//


JNIEXPORT jdouble JNICALL Java_org_opencv_highgui_VideoCapture_n_1get
  (JNIEnv* env, jclass cls, jlong self, jint propId)
{
    try {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1get()");
#endif // DEBUG
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        double _retval_ = me->get( propId );
        
        return _retval_;
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1get() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1get() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1get()}");
        return 0;
    }
}



//
//  bool VideoCapture::grab()
//


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1grab
  (JNIEnv* env, jclass cls, jlong self)
{
    try {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1grab()");
#endif // DEBUG
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        bool _retval_ = me->grab(  );
        
        return _retval_;
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1grab() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1grab() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1grab()}");
        return 0;
    }
}



//
//  bool VideoCapture::isOpened()
//


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1isOpened
  (JNIEnv* env, jclass cls, jlong self)
{
    try {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1isOpened()");
#endif // DEBUG
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        bool _retval_ = me->isOpened(  );
        
        return _retval_;
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1isOpened() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1isOpened() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1isOpened()}");
        return 0;
    }
}


//
//  bool VideoCapture::open(int device)
//


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1open__JI
  (JNIEnv* env, jclass cls, jlong self, jint device)
{
    try {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1open__JI()");
#endif // DEBUG
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        bool _retval_ = me->open( device );
        
        return _retval_;
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1open__JI() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1open__JI() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1open__JI()}");
        return 0;
    }
}



//
//  bool VideoCapture::read(Mat image)
//


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1read
  (JNIEnv* env, jclass cls, jlong self, jlong image_nativeObj)
{
    try {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1read()");
#endif // DEBUG
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        Mat& image = *((Mat*)image_nativeObj);
        bool _retval_ = me->read( image );
        
        return _retval_;
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1read() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1read() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1read()}");
        return 0;
    }
}



//
//  void VideoCapture::release()
//


JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_n_1release
  (JNIEnv* env, jclass cls, jlong self)
{
    try {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1release()");
#endif // DEBUG
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        me->release(  );
        
        return;
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1release() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1release() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1release()}");
        return;
    }
}



//
//  bool VideoCapture::retrieve(Mat image, int channel = 0)
//


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJI
  (JNIEnv* env, jclass cls, jlong self, jlong image_nativeObj, jint channel)
{
    try {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1retrieve__JJI()");
#endif // DEBUG
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        Mat& image = *((Mat*)image_nativeObj);
        bool _retval_ = me->retrieve( image, channel );
        
        return _retval_;
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1retrieve__JJI() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1retrieve__JJI() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1retrieve__JJI()}");
        return 0;
    }
}




JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJ
  (JNIEnv* env, jclass cls, jlong self, jlong image_nativeObj)
{
    try {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1retrieve__JJ()");
#endif // DEBUG
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        Mat& image = *((Mat*)image_nativeObj);
        bool _retval_ = me->retrieve( image );
        
        return _retval_;
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1retrieve__JJ() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1retrieve__JJ() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1retrieve__JJ()}");
        return 0;
    }
}



//
//  bool VideoCapture::set(int propId, double value)
//


JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1set
  (JNIEnv* env, jclass cls, jlong self, jint propId, jdouble value)
{
    try {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1set()");
#endif // DEBUG
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        bool _retval_ = me->set( propId, value );
        
        return _retval_;
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1set() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1set() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1set()}");
        return 0;
    }
}

JNIEXPORT jstring JNICALL Java_org_opencv_highgui_VideoCapture_n_1getSupportedPreviewSizes
  (JNIEnv *env, jclass cls, jlong self)
{
    try {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1set()");
#endif // DEBUG
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        double addr = me->get(CV_CAP_PROP_SUPPORTED_PREVIEW_SIZES_STRING);
        char* result = *((char**)&addr);
        return env->NewStringUTF(result);
    } catch(cv::Exception e) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1getSupportedPreviewSizes() catched cv::Exception: %s", e.what());
#endif // DEBUG
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return env->NewStringUTF("");
    } catch (...) {
#ifdef DEBUG
        LOGD("highgui::VideoCapture_n_1getSupportedPreviewSizes() catched unknown exception (...)");
#endif // DEBUG
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1getSupportedPreviewSizes()}");
        return env->NewStringUTF("");
    }
}



//
//  native support for java finalize()
//  static void VideoCapture::n_delete( __int64 self )
//

JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_n_1delete
  (JNIEnv* env, jclass cls, jlong self)
{
    delete (VideoCapture*) self;
}

} // extern "C"

