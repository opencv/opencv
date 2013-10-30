#define LOG_TAG "org.opencv.highgui.VideoCapture"
#include "common.h"

#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_HIGHGUI

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;


extern "C" {

//
//   VideoCapture::VideoCapture()
//

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__
  (JNIEnv* env, jclass);

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__
  (JNIEnv* env, jclass)
{
    try {
        LOGD("highgui::VideoCapture_n_1VideoCapture__()");

        VideoCapture* _retval_ = new VideoCapture(  );

        return (jlong) _retval_;
    } catch(cv::Exception e) {
        LOGD("highgui::VideoCapture_n_1VideoCapture__() catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("highgui::VideoCapture_n_1VideoCapture__() catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1VideoCapture__()}");
        return 0;
    }
}


//
//   VideoCapture::VideoCapture(int device)
//

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__I
  (JNIEnv* env, jclass, jint device);

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__I
  (JNIEnv* env, jclass, jint device)
{
    try {
        LOGD("highgui::VideoCapture_n_1VideoCapture__I()");

        VideoCapture* _retval_ = new VideoCapture( device );

        return (jlong) _retval_;
    } catch(cv::Exception e) {
        LOGD("highgui::VideoCapture_n_1VideoCapture__I() catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("highgui::VideoCapture_n_1VideoCapture__I() catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1VideoCapture__I()}");
        return 0;
    }
}



//
//  double VideoCapture::get(int propId)
//

JNIEXPORT jdouble JNICALL Java_org_opencv_highgui_VideoCapture_n_1get
  (JNIEnv* env, jclass, jlong self, jint propId);

JNIEXPORT jdouble JNICALL Java_org_opencv_highgui_VideoCapture_n_1get
  (JNIEnv* env, jclass, jlong self, jint propId)
{
    try {
        LOGD("highgui::VideoCapture_n_1get()");
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        double _retval_ = me->get( propId );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("highgui::VideoCapture_n_1get() catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("highgui::VideoCapture_n_1get() catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1get()}");
        return 0;
    }
}



//
//  bool VideoCapture::grab()
//

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1grab
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1grab
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("highgui::VideoCapture_n_1grab()");
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        bool _retval_ = me->grab(  );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("highgui::VideoCapture_n_1grab() catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("highgui::VideoCapture_n_1grab() catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1grab()}");
        return 0;
    }
}



//
//  bool VideoCapture::isOpened()
//

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1isOpened
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1isOpened
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("highgui::VideoCapture_n_1isOpened()");
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        bool _retval_ = me->isOpened(  );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("highgui::VideoCapture_n_1isOpened() catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("highgui::VideoCapture_n_1isOpened() catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1isOpened()}");
        return 0;
    }
}


//
//  bool VideoCapture::open(int device)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1open__JI
  (JNIEnv* env, jclass, jlong self, jint device);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1open__JI
  (JNIEnv* env, jclass, jlong self, jint device)
{
    try {
        LOGD("highgui::VideoCapture_n_1open__JI()");
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        bool _retval_ = me->open( device );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("highgui::VideoCapture_n_1open__JI() catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("highgui::VideoCapture_n_1open__JI() catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1open__JI()}");
        return 0;
    }
}



//
//  bool VideoCapture::read(Mat image)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1read
  (JNIEnv* env, jclass, jlong self, jlong image_nativeObj);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1read
  (JNIEnv* env, jclass, jlong self, jlong image_nativeObj)
{
    try {
        LOGD("highgui::VideoCapture_n_1read()");
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        Mat& image = *((Mat*)image_nativeObj);
        bool _retval_ = me->read( image );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("highgui::VideoCapture_n_1read() catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("highgui::VideoCapture_n_1read() catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1read()}");
        return 0;
    }
}



//
//  void VideoCapture::release()
//

JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_n_1release
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_n_1release
  (JNIEnv* env, jclass, jlong self)
{
    try {

        LOGD("highgui::VideoCapture_n_1release()");

        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        me->release(  );

        return;
    } catch(cv::Exception e) {

        LOGD("highgui::VideoCapture_n_1release() catched cv::Exception: %s", e.what());

        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {

        LOGD("highgui::VideoCapture_n_1release() catched unknown exception (...)");

        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1release()}");
        return;
    }
}



//
//  bool VideoCapture::retrieve(Mat image, int channel = 0)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJI
  (JNIEnv* env, jclass, jlong self, jlong image_nativeObj, jint channel);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJI
  (JNIEnv* env, jclass, jlong self, jlong image_nativeObj, jint channel)
{
    try {

        LOGD("highgui::VideoCapture_n_1retrieve__JJI()");

        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        Mat& image = *((Mat*)image_nativeObj);
        bool _retval_ = me->retrieve( image, channel );

        return _retval_;
    } catch(cv::Exception e) {

        LOGD("highgui::VideoCapture_n_1retrieve__JJI() catched cv::Exception: %s", e.what());

        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {

        LOGD("highgui::VideoCapture_n_1retrieve__JJI() catched unknown exception (...)");

        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1retrieve__JJI()}");
        return 0;
    }
}



JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJ
  (JNIEnv* env, jclass, jlong self, jlong image_nativeObj);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJ
  (JNIEnv* env, jclass, jlong self, jlong image_nativeObj)
{
    try {

        LOGD("highgui::VideoCapture_n_1retrieve__JJ()");

        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        Mat& image = *((Mat*)image_nativeObj);
        bool _retval_ = me->retrieve( image );

        return _retval_;
    } catch(cv::Exception e) {

        LOGD("highgui::VideoCapture_n_1retrieve__JJ() catched cv::Exception: %s", e.what());

        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {

        LOGD("highgui::VideoCapture_n_1retrieve__JJ() catched unknown exception (...)");

        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1retrieve__JJ()}");
        return 0;
    }
}



//
//  bool VideoCapture::set(int propId, double value)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1set
  (JNIEnv* env, jclass, jlong self, jint propId, jdouble value);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1set
  (JNIEnv* env, jclass, jlong self, jint propId, jdouble value)
{
    try {

        LOGD("highgui::VideoCapture_n_1set()");

        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        bool _retval_ = me->set( propId, value );

        return _retval_;
    } catch(cv::Exception e) {

        LOGD("highgui::VideoCapture_n_1set() catched cv::Exception: %s", e.what());

        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {

        LOGD("highgui::VideoCapture_n_1set() catched unknown exception (...)");

        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1set()}");
        return 0;
    }
}

JNIEXPORT jstring JNICALL Java_org_opencv_highgui_VideoCapture_n_1getSupportedPreviewSizes
  (JNIEnv *env, jclass, jlong self);

JNIEXPORT jstring JNICALL Java_org_opencv_highgui_VideoCapture_n_1getSupportedPreviewSizes
  (JNIEnv *env, jclass, jlong self)
{
    try {

        LOGD("highgui::VideoCapture_n_1set()");

        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        union {double prop; const char* name;} u;
        u.prop = me->get(CV_CAP_PROP_SUPPORTED_PREVIEW_SIZES_STRING);
        return env->NewStringUTF(u.name);
    } catch(cv::Exception e) {

        LOGD("highgui::VideoCapture_n_1getSupportedPreviewSizes() catched cv::Exception: %s", e.what());

        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return env->NewStringUTF("");
    } catch (...) {

        LOGD("highgui::VideoCapture_n_1getSupportedPreviewSizes() catched unknown exception (...)");

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
  (JNIEnv*, jclass, jlong self);

JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_n_1delete
  (JNIEnv*, jclass, jlong self)
{
    delete (VideoCapture*) self;
}

} // extern "C"

#endif // HAVE_OPENCV_HIGHGUI