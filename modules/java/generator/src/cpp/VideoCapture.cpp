#define LOG_TAG "org.opencv.highgui.VideoCapture"
#include "common.h"

#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_HIGHGUI

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;

/// throw java exception
static void throwJavaException(JNIEnv *env, const std::exception *e, const char *method) {
  std::string what = "unknown exception";
  jclass je = 0;

  if(e) {
    std::string exception_type = "std::exception";

    if(dynamic_cast<const cv::Exception*>(e)) {
      exception_type = "cv::Exception";
      je = env->FindClass("org/opencv/core/CvException");
    }

    what = exception_type + ": " + e->what();
  }

  if(!je) je = env->FindClass("java/lang/Exception");
  env->ThrowNew(je, what.c_str());

  LOGE("%s caught %s", method, what.c_str());
  (void)method;        // avoid "unused" warning
}

extern "C" {

//
//   VideoCapture::VideoCapture()
//

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__
  (JNIEnv* env, jclass);

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__
  (JNIEnv* env, jclass)
{
    static const char method_name[] = "highgui::VideoCapture::VideoCapture()";
    try {
        LOGD("%s", method_name);
        VideoCapture* _retval_ = new VideoCapture(  );
        return (jlong) _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}


//
//   VideoCapture::VideoCapture(int device)
//

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__I
  (JNIEnv* env, jclass, jint device);

JNIEXPORT jlong JNICALL Java_org_opencv_highgui_VideoCapture_n_1VideoCapture__I
  (JNIEnv* env, jclass, jint device)
{
    static const char method_name[] = "highgui::VideoCapture::VideoCapture(int device)";
    try {
        LOGD("%s", method_name);
        VideoCapture* _retval_ = new VideoCapture( device );
        return (jlong) _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  double VideoCapture::get(int propId)
//

JNIEXPORT jdouble JNICALL Java_org_opencv_highgui_VideoCapture_n_1get
  (JNIEnv* env, jclass, jlong self, jint propId);

JNIEXPORT jdouble JNICALL Java_org_opencv_highgui_VideoCapture_n_1get
  (JNIEnv* env, jclass, jlong self, jint propId)
{
    static const char method_name[] = "highgui::VideoCapture::get(int propId)";
    try {
        LOGD("%s", method_name);
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        double _retval_ = me->get( propId );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  bool VideoCapture::grab()
//

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1grab
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1grab
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "highgui::VideoCapture::grab()";
    try {
        LOGD("%s", method_name);
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        bool _retval_ = me->grab(  );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return false;
}



//
//  bool VideoCapture::isOpened()
//

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1isOpened
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1isOpened
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "highgui::VideoCapture::isOpened()";
    try {
        LOGD("%s", method_name);
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        bool _retval_ = me->isOpened(  );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return false;
}


//
//  bool VideoCapture::open(int device)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1open__JI
  (JNIEnv* env, jclass, jlong self, jint device);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1open__JI
  (JNIEnv* env, jclass, jlong self, jint device)
{
    static const char method_name[] = "highgui::VideoCapture::open(int device)";
    try {
        LOGD("%s", method_name);
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        bool _retval_ = me->open( device );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return false;
}



//
//  bool VideoCapture::read(Mat image)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1read
  (JNIEnv* env, jclass, jlong self, jlong image_nativeObj);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1read
  (JNIEnv* env, jclass, jlong self, jlong image_nativeObj)
{
    static const char method_name[] = "highgui::VideoCapture::read(Mat image)";
    try {
        LOGD("%s", method_name);
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        Mat& image = *((Mat*)image_nativeObj);
        bool _retval_ = me->read( image );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return false;
}



//
//  void VideoCapture::release()
//

JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_n_1release
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT void JNICALL Java_org_opencv_highgui_VideoCapture_n_1release
  (JNIEnv* env, jclass, jlong self)
{
    static const char method_name[] = "highgui::VideoCapture::release()";
    try {
        LOGD("%s", method_name);
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        me->release(  );
        return;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return;
}



//
//  bool VideoCapture::retrieve(Mat image, int channel = 0)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJI
  (JNIEnv* env, jclass, jlong self, jlong image_nativeObj, jint channel);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJI
  (JNIEnv* env, jclass, jlong self, jlong image_nativeObj, jint channel)
{
    static const char method_name[] = "highgui::VideoCapture::retrieve(Mat image, int channel)";
    try {
        LOGD("%s", method_name);
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        Mat& image = *((Mat*)image_nativeObj);
        bool _retval_ = me->retrieve( image, channel );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return false;
}



JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJ
  (JNIEnv* env, jclass, jlong self, jlong image_nativeObj);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1retrieve__JJ
  (JNIEnv* env, jclass, jlong self, jlong image_nativeObj)
{
    static const char method_name[] = "highgui::VideoCapture::retrieve(Mat image)";
    try {
        LOGD("%s", method_name);
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        Mat& image = *((Mat*)image_nativeObj);
        bool _retval_ = me->retrieve( image );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return false;
}



//
//  bool VideoCapture::set(int propId, double value)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1set
  (JNIEnv* env, jclass, jlong self, jint propId, jdouble value);

JNIEXPORT jboolean JNICALL Java_org_opencv_highgui_VideoCapture_n_1set
  (JNIEnv* env, jclass, jlong self, jint propId, jdouble value)
{
    static const char method_name[] = "highgui::VideoCapture::set(int propId, double value)";
    try {
        LOGD("%s", method_name);
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        bool _retval_ = me->set( propId, value );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return false;
}


//
//  string VideoCapture::getSupportedPreviewSizes(...)
//

JNIEXPORT jstring JNICALL Java_org_opencv_highgui_VideoCapture_n_1getSupportedPreviewSizes
  (JNIEnv *env, jclass, jlong self);

JNIEXPORT jstring JNICALL Java_org_opencv_highgui_VideoCapture_n_1getSupportedPreviewSizes
  (JNIEnv *env, jclass, jlong self)
{
    static const char method_name[] = "highgui::VideoCapture::getSupportedPreviewSizes(...)";
    try {
        LOGD("%s", method_name);
        VideoCapture* me = (VideoCapture*) self; //TODO: check for NULL
        union {double prop; const char* name;} u;
        u.prop = me->get(CV_CAP_PROP_SUPPORTED_PREVIEW_SIZES_STRING);
        return env->NewStringUTF(u.name);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return env->NewStringUTF("");
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