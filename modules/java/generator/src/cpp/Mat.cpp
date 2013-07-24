#define LOG_TAG "org.opencv.core.Mat"

#include "common.h"
#include "opencv2/core.hpp"

using namespace cv;

extern "C" {


//
//   MatXXX::MatXXX()
//


JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__
  (JNIEnv*, jclass);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__
  (JNIEnv*, jclass)
{
    LOGD("Mat::n_1Mat__()");
    return (jlong) new cv::Mat();
}



//
//   Mat::Mat(int rows, int cols, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type)
{
    try {
        LOGD("Mat::n_1Mat__III()");

        Mat* _retval_ = new Mat( rows, cols, type );

        return (jlong) _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1Mat__III() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1Mat__III() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1Mat__III()}");
        return 0;
    }
}



//
//   Mat::Mat(Size size, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type)
{
    try {
        LOGD("Mat::n_1Mat__DDI()");
        Size size((int)size_width, (int)size_height);
        Mat* _retval_ = new Mat( size, type );

        return (jlong) _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1Mat__DDI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1Mat__DDI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1Mat__DDI()}");
        return 0;
    }
}



//
//   Mat::Mat(int rows, int cols, int type, Scalar s)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__IIIDDDD
  (JNIEnv* env, jclass, jint rows, jint cols, jint type, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3);


JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__IIIDDDD
  (JNIEnv* env, jclass, jint rows, jint cols, jint type, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3)
{
    try {
        LOGD("Mat::n_1Mat__IIIDDDD()");
        Scalar s(s_val0, s_val1, s_val2, s_val3);
        Mat* _retval_ = new Mat( rows, cols, type, s );

        return (jlong) _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1Mat__IIIDDDD() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1Mat__IIIDDDD() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1Mat__IIIDDDD()}");
        return 0;
    }
}



//
//   Mat::Mat(Size size, int type, Scalar s)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__DDIDDDD
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__DDIDDDD
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3)
{
    try {
        LOGD("Mat::n_1Mat__DDIDDDD()");
        Size size((int)size_width, (int)size_height);
        Scalar s(s_val0, s_val1, s_val2, s_val3);
        Mat* _retval_ = new Mat( size, type, s );

        return (jlong) _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1Mat__DDIDDDD() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1Mat__DDIDDDD() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1Mat__DDIDDDD()}");
        return 0;
    }
}



//
//   Mat::Mat(Mat m, Range rowRange, Range colRange = Range::all())
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__JIIII
  (JNIEnv* env, jclass, jlong m_nativeObj, jint rowRange_start, jint rowRange_end, jint colRange_start, jint colRange_end);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__JIIII
  (JNIEnv* env, jclass, jlong m_nativeObj, jint rowRange_start, jint rowRange_end, jint colRange_start, jint colRange_end)
{
    try {
        LOGD("Mat::n_1Mat__JIIII()");
        Range rowRange(rowRange_start, rowRange_end);
        Range colRange(colRange_start, colRange_end);
        Mat* _retval_ = new Mat( (*(Mat*)m_nativeObj), rowRange, colRange );

        return (jlong) _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1Mat__JIIII() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1Mat__JIIII() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1Mat__JIIII()}");
        return 0;
    }
}


JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__JII
  (JNIEnv* env, jclass, jlong m_nativeObj, jint rowRange_start, jint rowRange_end);


JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1Mat__JII
  (JNIEnv* env, jclass, jlong m_nativeObj, jint rowRange_start, jint rowRange_end)
{
    try {
        LOGD("Mat::n_1Mat__JII()");
        Range rowRange(rowRange_start, rowRange_end);
        Mat* _retval_ = new Mat( (*(Mat*)m_nativeObj), rowRange );

        return (jlong) _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1Mat__JII() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1Mat__JII() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1Mat__JII()}");
        return 0;
    }
}


//
//  Mat Mat::adjustROI(int dtop, int dbottom, int dleft, int dright)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1adjustROI
  (JNIEnv* env, jclass, jlong self, jint dtop, jint dbottom, jint dleft, jint dright);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1adjustROI
  (JNIEnv* env, jclass, jlong self, jint dtop, jint dbottom, jint dleft, jint dright)
{
    try {
        LOGD("Mat::n_1adjustROI()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->adjustROI( dtop, dbottom, dleft, dright );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1adjustROI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1adjustROI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1adjustROI()}");
        return 0;
    }
}



//
//  void Mat::assignTo(Mat m, int type = -1)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1assignTo__JJI
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint type);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1assignTo__JJI
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint type)
{
    try {
        LOGD("Mat::n_1assignTo__JJI()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        me->assignTo( (*(Mat*)m_nativeObj), type );

        return;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1assignTo__JJI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        LOGD("Mat::n_1assignTo__JJI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1assignTo__JJI()}");
        return;
    }
}


JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1assignTo__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1assignTo__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    try {
        LOGD("Mat::n_1assignTo__JJ()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        me->assignTo( (*(Mat*)m_nativeObj) );

        return;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1assignTo__JJ() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        LOGD("Mat::n_1assignTo__JJ() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1assignTo__JJ()}");
        return;
    }
}



//
//  int Mat::channels()
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1channels
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1channels
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1channels()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        int _retval_ = me->channels(  );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1channels() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1channels() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1channels()}");
        return 0;
    }
}



//
//  int Mat::checkVector(int elemChannels, int depth = -1, bool requireContinuous = true)
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1checkVector__JIIZ
  (JNIEnv* env, jclass, jlong self, jint elemChannels, jint depth, jboolean requireContinuous);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1checkVector__JIIZ
  (JNIEnv* env, jclass, jlong self, jint elemChannels, jint depth, jboolean requireContinuous)
{
    try {
        LOGD("Mat::n_1checkVector__JIIZ()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        int _retval_ = me->checkVector( elemChannels, depth, requireContinuous );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1checkVector__JIIZ() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1checkVector__JIIZ() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1checkVector__JIIZ()}");
        return 0;
    }
}



JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1checkVector__JII
  (JNIEnv* env, jclass, jlong self, jint elemChannels, jint depth);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1checkVector__JII
  (JNIEnv* env, jclass, jlong self, jint elemChannels, jint depth)
{
    try {
        LOGD("Mat::n_1checkVector__JII()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        int _retval_ = me->checkVector( elemChannels, depth );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1checkVector__JII() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1checkVector__JII() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1checkVector__JII()}");
        return 0;
    }
}


JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1checkVector__JI
  (JNIEnv* env, jclass, jlong self, jint elemChannels);


JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1checkVector__JI
  (JNIEnv* env, jclass, jlong self, jint elemChannels)
{
    try {
        LOGD("Mat::n_1checkVector__JI()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        int _retval_ = me->checkVector( elemChannels );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1checkVector__JI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1checkVector__JI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1checkVector__JI()}");
        return 0;
    }
}



//
//  Mat Mat::clone()
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1clone
  (JNIEnv* env, jclass, jlong self);


JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1clone
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1clone()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->clone(  );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1clone() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1clone() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1clone()}");
        return 0;
    }
}



//
//  Mat Mat::col(int x)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1col
  (JNIEnv* env, jclass, jlong self, jint x);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1col
  (JNIEnv* env, jclass, jlong self, jint x)
{
    try {
        LOGD("Mat::n_1col()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->col( x );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1col() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1col() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1col()}");
        return 0;
    }
}



//
//  Mat Mat::colRange(int startcol, int endcol)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1colRange
  (JNIEnv* env, jclass, jlong self, jint startcol, jint endcol);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1colRange
  (JNIEnv* env, jclass, jlong self, jint startcol, jint endcol)
{
    try {
        LOGD("Mat::n_1colRange()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->colRange( startcol, endcol );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1colRange() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1colRange() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1colRange()}");
        return 0;
    }
}



//
//  int Mat::dims()
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1dims
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1dims
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1dims()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        int _retval_ = me->dims;

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1dims() catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1dims() catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1dims()}");
        return 0;
    }
}


//
//  int Mat::cols()
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1cols
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1cols
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1cols()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        int _retval_ = me->cols;

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1cols() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1cols() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1cols()}");
        return 0;
    }
}



//
//  void Mat::convertTo(Mat& m, int rtype, double alpha = 1, double beta = 0)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1convertTo__JJIDD
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint rtype, jdouble alpha, jdouble beta);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1convertTo__JJIDD
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint rtype, jdouble alpha, jdouble beta)
{
    try {
        LOGD("Mat::n_1convertTo__JJIDD()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        me->convertTo( m, rtype, alpha, beta );

        return;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1convertTo__JJIDD() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        LOGD("Mat::n_1convertTo__JJIDD() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1convertTo__JJIDD()}");
        return;
    }
}


JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1convertTo__JJID
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint rtype, jdouble alpha);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1convertTo__JJID
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint rtype, jdouble alpha)
{
    try {
        LOGD("Mat::n_1convertTo__JJID()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        me->convertTo( m, rtype, alpha );

        return;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1convertTo__JJID() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        LOGD("Mat::n_1convertTo__JJID() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1convertTo__JJID()}");
        return;
    }
}


JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1convertTo__JJI
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint rtype);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1convertTo__JJI
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jint rtype)
{
    try {
        LOGD("Mat::n_1convertTo__JJI()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        me->convertTo( m, rtype );

        return;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1convertTo__JJI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        LOGD("Mat::n_1convertTo__JJI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1convertTo__JJI()}");
        return;
    }
}



//
//  void Mat::copyTo(Mat& m)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1copyTo__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1copyTo__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    try {
        LOGD("Mat::n_1copyTo__JJ()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        me->copyTo( m );

        return;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1copyTo__JJ() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        LOGD("Mat::n_1copyTo__JJ() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1copyTo__JJ()}");
        return;
    }
}



//
//  void Mat::copyTo(Mat& m, Mat mask)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1copyTo__JJJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jlong mask_nativeObj);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1copyTo__JJJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jlong mask_nativeObj)
{
    try {
        LOGD("Mat::n_1copyTo__JJJ()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        Mat& mask = *((Mat*)mask_nativeObj);
        me->copyTo( m, mask );

        return;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1copyTo__JJJ() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        LOGD("Mat::n_1copyTo__JJJ() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1copyTo__JJJ()}");
        return;
    }
}



//
//  void Mat::create(int rows, int cols, int type)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1create__JIII
  (JNIEnv* env, jclass, jlong self, jint rows, jint cols, jint type);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1create__JIII
  (JNIEnv* env, jclass, jlong self, jint rows, jint cols, jint type)
{
    try {
        LOGD("Mat::n_1create__JIII()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        me->create( rows, cols, type );

        return;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1create__JIII() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        LOGD("Mat::n_1create__JIII() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1create__JIII()}");
        return;
    }
}



//
//  void Mat::create(Size size, int type)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1create__JDDI
  (JNIEnv* env, jclass, jlong self, jdouble size_width, jdouble size_height, jint type);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1create__JDDI
  (JNIEnv* env, jclass, jlong self, jdouble size_width, jdouble size_height, jint type)
{
    try {
        LOGD("Mat::n_1create__JDDI()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Size size((int)size_width, (int)size_height);
        me->create( size, type );

        return;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1create__JDDI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        LOGD("Mat::n_1create__JDDI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1create__JDDI()}");
        return;
    }
}



//
//  Mat Mat::cross(Mat m)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1cross
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1cross
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    try {
        LOGD("Mat::n_1cross()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        Mat _retval_ = me->cross( m );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1cross() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1cross() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1cross()}");
        return 0;
    }
}



//
//  long Mat::dataAddr()
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1dataAddr
  (JNIEnv*, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1dataAddr
  (JNIEnv*, jclass, jlong self)
{
    LOGD("Mat::n_1dataAddr()");
    Mat* me = (Mat*) self; //TODO: check for NULL
    return (jlong) me->data;
}



//
//  int Mat::depth()
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1depth
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1depth
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1depth()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        int _retval_ = me->depth(  );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1depth() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1depth() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1depth()}");
        return 0;
    }
}



//
//  Mat Mat::diag(int d = 0)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1diag__JI
  (JNIEnv* env, jclass, jlong self, jint d);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1diag__JI
  (JNIEnv* env, jclass, jlong self, jint d)
{
    try {
        LOGD("Mat::n_1diag__JI()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->diag( d );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1diag__JI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1diag__JI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1diag__JI()}");
        return 0;
    }
}




//
// static Mat Mat::diag(Mat d)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1diag__J
  (JNIEnv* env, jclass, jlong d_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1diag__J
  (JNIEnv* env, jclass, jlong d_nativeObj)
{
    try {
        LOGD("Mat::n_1diag__J()");

        Mat _retval_ = Mat::diag( (*(Mat*)d_nativeObj) );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1diag__J() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1diag__J() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1diag__J()}");
        return 0;
    }
}



//
//  double Mat::dot(Mat m)
//

JNIEXPORT jdouble JNICALL Java_org_opencv_core_Mat_n_1dot
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT jdouble JNICALL Java_org_opencv_core_Mat_n_1dot
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    try {
        LOGD("Mat::n_1dot()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        double _retval_ = me->dot( m );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1dot() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1dot() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1dot()}");
        return 0;
    }
}



//
//  size_t Mat::elemSize()
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1elemSize
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1elemSize
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1elemSize()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        size_t _retval_ = me->elemSize(  );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1elemSize() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1elemSize() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1elemSize()}");
        return 0;
    }
}



//
//  size_t Mat::elemSize1()
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1elemSize1
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1elemSize1
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1elemSize1()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        size_t _retval_ = me->elemSize1(  );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1elemSize1() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1elemSize1() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1elemSize1()}");
        return 0;
    }
}



//
//  bool Mat::empty()
//

JNIEXPORT jboolean JNICALL Java_org_opencv_core_Mat_n_1empty
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jboolean JNICALL Java_org_opencv_core_Mat_n_1empty
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1empty()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        bool _retval_ = me->empty(  );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1empty() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1empty() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1empty()}");
        return 0;
    }
}



//
// static Mat Mat::eye(int rows, int cols, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1eye__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1eye__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type)
{
    try {
        LOGD("Mat::n_1eye__III()");

        Mat _retval_ = Mat::eye( rows, cols, type );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1eye__III() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1eye__III() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1eye__III()}");
        return 0;
    }
}



//
// static Mat Mat::eye(Size size, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1eye__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1eye__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type)
{
    try {
        LOGD("Mat::n_1eye__DDI()");
        Size size((int)size_width, (int)size_height);
        Mat _retval_ = Mat::eye( size, type );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1eye__DDI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1eye__DDI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1eye__DDI()}");
        return 0;
    }
}



//
//  Mat Mat::inv(int method = DECOMP_LU)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1inv__JI
  (JNIEnv* env, jclass, jlong self, jint method);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1inv__JI
  (JNIEnv* env, jclass, jlong self, jint method)
{
    try {
        LOGD("Mat::n_1inv__JI()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->inv( method );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1inv__JI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1inv__JI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1inv__JI()}");
        return 0;
    }
}


JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1inv__J
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1inv__J
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1inv__J()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->inv(  );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1inv__J() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1inv__J() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1inv__J()}");
        return 0;
    }
}



//
//  bool Mat::isContinuous()
//

JNIEXPORT jboolean JNICALL Java_org_opencv_core_Mat_n_1isContinuous
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jboolean JNICALL Java_org_opencv_core_Mat_n_1isContinuous
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1isContinuous()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        bool _retval_ = me->isContinuous(  );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1isContinuous() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1isContinuous() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1isContinuous()}");
        return 0;
    }
}



//
//  bool Mat::isSubmatrix()
//

JNIEXPORT jboolean JNICALL Java_org_opencv_core_Mat_n_1isSubmatrix
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jboolean JNICALL Java_org_opencv_core_Mat_n_1isSubmatrix
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1isSubmatrix()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        bool _retval_ = me->isSubmatrix(  );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1isSubmatrix() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1isSubmatrix() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1isSubmatrix()}");
        return 0;
    }
}



//
//  void Mat::locateROI(Size wholeSize, Point ofs)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_locateROI_10
  (JNIEnv* env, jclass, jlong self, jdoubleArray wholeSize_out, jdoubleArray ofs_out);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_locateROI_10
  (JNIEnv* env, jclass, jlong self, jdoubleArray wholeSize_out, jdoubleArray ofs_out)
{
    try {
        LOGD("core::locateROI_10()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Size wholeSize;
        Point ofs;
        me->locateROI( wholeSize, ofs );
        jdouble tmp_wholeSize[2] = {wholeSize.width, wholeSize.height}; env->SetDoubleArrayRegion(wholeSize_out, 0, 2, tmp_wholeSize);  jdouble tmp_ofs[2] = {ofs.x, ofs.y}; env->SetDoubleArrayRegion(ofs_out, 0, 2, tmp_ofs);
        return;
    } catch(cv::Exception e) {
        LOGD("Mat::locateROI_10() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        LOGD("Mat::locateROI_10() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::locateROI_10()}");
        return;
    }
}



//
//  Mat Mat::mul(Mat m, double scale = 1)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1mul__JJD
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jdouble scale);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1mul__JJD
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj, jdouble scale)
{
    try {
        LOGD("Mat::n_1mul__JJD()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        Mat _retval_ = me->mul( m, scale );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1mul__JJD() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1mul__JJD() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1mul__JJD()}");
        return 0;
    }
}



JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1mul__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1mul__JJ
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    try {
        LOGD("Mat::n_1mul__JJ()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& m = *((Mat*)m_nativeObj);
        Mat _retval_ = me->mul( m );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1mul__JJ() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1mul__JJ() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1mul__JJ()}");
        return 0;
    }
}



//
// static Mat Mat::ones(int rows, int cols, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1ones__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1ones__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type)
{
    try {
        LOGD("Mat::n_1ones__III()");

        Mat _retval_ = Mat::ones( rows, cols, type );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1ones__III() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1ones__III() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1ones__III()}");
        return 0;
    }
}



//
// static Mat Mat::ones(Size size, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1ones__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1ones__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type)
{
    try {
        LOGD("Mat::n_1ones__DDI()");
        Size size((int)size_width, (int)size_height);
        Mat _retval_ = Mat::ones( size, type );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1ones__DDI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1ones__DDI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1ones__DDI()}");
        return 0;
    }
}



//
//  void Mat::push_back(Mat m)
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1push_1back
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1push_1back
  (JNIEnv* env, jclass, jlong self, jlong m_nativeObj)
{
    try {
        LOGD("Mat::n_1push_1back()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        me->push_back( (*(Mat*)m_nativeObj) );

        return;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1push_1back() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        LOGD("Mat::n_1push_1back() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1push_1back()}");
        return;
    }
}



//
//  void Mat::release()
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1release
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1release
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1release()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        me->release(  );

        return;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1release() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        LOGD("Mat::n_1release() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1release()}");
        return;
    }
}



//
//  Mat Mat::reshape(int cn, int rows = 0)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1reshape__JII
  (JNIEnv* env, jclass, jlong self, jint cn, jint rows);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1reshape__JII
  (JNIEnv* env, jclass, jlong self, jint cn, jint rows)
{
    try {
        LOGD("Mat::n_1reshape__JII()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->reshape( cn, rows );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1reshape__JII() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1reshape__JII() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1reshape__JII()}");
        return 0;
    }
}



JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1reshape__JI
  (JNIEnv* env, jclass, jlong self, jint cn);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1reshape__JI
  (JNIEnv* env, jclass, jlong self, jint cn)
{
    try {
        LOGD("Mat::n_1reshape__JI()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->reshape( cn );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1reshape__JI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1reshape__JI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1reshape__JI()}");
        return 0;
    }
}



//
//  Mat Mat::row(int y)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1row
  (JNIEnv* env, jclass, jlong self, jint y);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1row
  (JNIEnv* env, jclass, jlong self, jint y)
{
    try {
        LOGD("Mat::n_1row()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->row( y );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1row() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1row() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1row()}");
        return 0;
    }
}



//
//  Mat Mat::rowRange(int startrow, int endrow)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1rowRange
  (JNIEnv* env, jclass, jlong self, jint startrow, jint endrow);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1rowRange
  (JNIEnv* env, jclass, jlong self, jint startrow, jint endrow)
{
    try {
        LOGD("Mat::n_1rowRange()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->rowRange( startrow, endrow );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1rowRange() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1rowRange() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1rowRange()}");
        return 0;
    }
}



//
//  int Mat::rows()
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1rows
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1rows
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1rows()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        int _retval_ = me->rows;

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1rows() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1rows() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1rows()}");
        return 0;
    }
}



//
//  Mat Mat::operator =(Scalar s)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JDDDD
  (JNIEnv* env, jclass, jlong self, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JDDDD
  (JNIEnv* env, jclass, jlong self, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3)
{
    try {
        LOGD("Mat::n_1setTo__JDDDD()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Scalar s(s_val0, s_val1, s_val2, s_val3);
        Mat _retval_ = me->operator =( s );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1setTo__JDDDD() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1setTo__JDDDD() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1setTo__JDDDD()}");
        return 0;
    }
}



//
//  Mat Mat::setTo(Scalar value, Mat mask = Mat())
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JDDDDJ
  (JNIEnv* env, jclass, jlong self, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3, jlong mask_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JDDDDJ
  (JNIEnv* env, jclass, jlong self, jdouble s_val0, jdouble s_val1, jdouble s_val2, jdouble s_val3, jlong mask_nativeObj)
{
    try {
        LOGD("Mat::n_1setTo__JDDDDJ()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Scalar s(s_val0, s_val1, s_val2, s_val3);
        Mat& mask = *((Mat*)mask_nativeObj);
        Mat _retval_ = me->setTo( s, mask );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1setTo__JDDDDJ() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1setTo__JDDDDJ() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1setTo__JDDDDJ()}");
        return 0;
    }
}



//
//  Mat Mat::setTo(Mat value, Mat mask = Mat())
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JJJ
  (JNIEnv* env, jclass, jlong self, jlong value_nativeObj, jlong mask_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JJJ
  (JNIEnv* env, jclass, jlong self, jlong value_nativeObj, jlong mask_nativeObj)
{
    try {
        LOGD("Mat::n_1setTo__JJJ()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& value = *((Mat*)value_nativeObj);
        Mat& mask = *((Mat*)mask_nativeObj);
        Mat _retval_ = me->setTo( value, mask );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1setTo__JJJ() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1setTo__JJJ() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1setTo__JJJ()}");
        return 0;
    }
}



JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JJ
  (JNIEnv* env, jclass, jlong self, jlong value_nativeObj);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1setTo__JJ
  (JNIEnv* env, jclass, jlong self, jlong value_nativeObj)
{
    try {
        LOGD("Mat::n_1setTo__JJ()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat& value = *((Mat*)value_nativeObj);
        Mat _retval_ = me->setTo( value );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1setTo__JJ() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1setTo__JJ() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1setTo__JJ()}");
        return 0;
    }
}



//
//  Size Mat::size()
//

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Mat_n_1size
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Mat_n_1size
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1size()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Size _retval_ = me->size(  );
        jdoubleArray _da_retval_ = env->NewDoubleArray(2);  jdouble _tmp_retval_[2] = {_retval_.width, _retval_.height}; env->SetDoubleArrayRegion(_da_retval_, 0, 2, _tmp_retval_);
        return _da_retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1size() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1size() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1size()}");
        return 0;
    }
}



//
//  size_t Mat::step1(int i = 0)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1step1__JI
  (JNIEnv* env, jclass, jlong self, jint i);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1step1__JI
  (JNIEnv* env, jclass, jlong self, jint i)
{
    try {
        LOGD("Mat::n_1step1__JI()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        size_t _retval_ = me->step1( i );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1step1__JI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1step1__JI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1step1__JI()}");
        return 0;
    }
}



JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1step1__J
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1step1__J
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1step1__J()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        size_t _retval_ = me->step1(  );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1step1__J() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1step1__J() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1step1__J()}");
        return 0;
    }
}

//
//  Mat Mat::operator()(Range rowRange, Range colRange)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1submat_1rr
  (JNIEnv* env, jclass, jlong self, jint rowRange_start, jint rowRange_end, jint colRange_start, jint colRange_end);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1submat_1rr
  (JNIEnv* env, jclass, jlong self, jint rowRange_start, jint rowRange_end, jint colRange_start, jint colRange_end)
{
    try {
        LOGD("Mat::n_1submat_1rr()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Range rowRange(rowRange_start, rowRange_end);
        Range colRange(colRange_start, colRange_end);
        Mat _retval_ = me->operator()( rowRange, colRange );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1submat_1rr() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1submat_1rr() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1submat_1rr()}");
        return 0;
    }
}



//
//  Mat Mat::operator()(Rect roi)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1submat
  (JNIEnv* env, jclass, jlong self, jint roi_x, jint roi_y, jint roi_width, jint roi_height);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1submat
  (JNIEnv* env, jclass, jlong self, jint roi_x, jint roi_y, jint roi_width, jint roi_height)
{
    try {
        LOGD("Mat::n_1submat()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Rect roi(roi_x, roi_y, roi_width, roi_height);
        Mat _retval_ = me->operator()( roi );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1submat() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1submat() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1submat()}");
        return 0;
    }
}



//
//  Mat Mat::t()
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1t
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1t
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1t()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        Mat _retval_ = me->t(  );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1t() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1t() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1t()}");
        return 0;
    }
}



//
//  size_t Mat::total()
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1total
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1total
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1total()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        size_t _retval_ = me->total(  );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1total() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1total() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1total()}");
        return 0;
    }
}



//
//  int Mat::type()
//

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1type
  (JNIEnv* env, jclass, jlong self);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_n_1type
  (JNIEnv* env, jclass, jlong self)
{
    try {
        LOGD("Mat::n_1type()");
        Mat* me = (Mat*) self; //TODO: check for NULL
        int _retval_ = me->type(  );

        return _retval_;
    } catch(cv::Exception e) {
        LOGD("Mat::n_1type() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1type() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1type()}");
        return 0;
    }
}



//
// static Mat Mat::zeros(int rows, int cols, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1zeros__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1zeros__III
  (JNIEnv* env, jclass, jint rows, jint cols, jint type)
{
    try {
        LOGD("Mat::n_1zeros__III()");

        Mat _retval_ = Mat::zeros( rows, cols, type );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1zeros__III() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1zeros__III() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1zeros__III()}");
        return 0;
    }
}



//
// static Mat Mat::zeros(Size size, int type)
//

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1zeros__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_n_1zeros__DDI
  (JNIEnv* env, jclass, jdouble size_width, jdouble size_height, jint type)
{
    try {
        LOGD("Mat::n_1zeros__DDI()");
        Size size((int)size_width, (int)size_height);
        Mat _retval_ = Mat::zeros( size, type );

        return (jlong) new Mat(_retval_);
    } catch(cv::Exception e) {
        LOGD("Mat::n_1zeros__DDI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::n_1zeros__DDI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::n_1zeros__DDI()}");
        return 0;
    }
}



//
//  native support for java finalize()
//  static void Mat::n_delete( __int64 self )
//

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1delete
  (JNIEnv*, jclass, jlong self);

JNIEXPORT void JNICALL Java_org_opencv_core_Mat_n_1delete
  (JNIEnv*, jclass, jlong self)
{
    delete (Mat*) self;
}

// unlike other nPut()-s this one (with double[]) should convert input values to correct type
#define PUT_ITEM(T, R, C) { T*dst = (T*)me->ptr(R, C); for(int ch=0; ch<me->channels() && count>0; count--,ch++,src++,dst++) *dst = cv::saturate_cast<T>(*src); }

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutD
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jdoubleArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutD
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jdoubleArray vals)
{
    try {
        LOGD("Mat::nPutD()");
        cv::Mat* me = (cv::Mat*) self;
        if(!me || !me->data) return 0;  // no native object behind
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        int rest = ((me->rows - row) * me->cols - col) * me->channels();
        if(count>rest) count = rest;
        int res = count;
        double* values = (double*)env->GetPrimitiveArrayCritical(vals, 0);
        double* src = values;
        int r, c;
        for(c=col; c<me->cols && count>0; c++)
        {
            switch(me->depth()) {
                case CV_8U:  PUT_ITEM(uchar,  row, c); break;
                case CV_8S:  PUT_ITEM(schar,  row, c); break;
                case CV_16U: PUT_ITEM(ushort, row, c); break;
                case CV_16S: PUT_ITEM(short,  row, c); break;
                case CV_32S: PUT_ITEM(int,    row, c); break;
                case CV_32F: PUT_ITEM(float,  row, c); break;
                case CV_64F: PUT_ITEM(double, row, c); break;
            }
        }

        for(r=row+1; r<me->rows && count>0; r++)
            for(c=0; c<me->cols && count>0; c++)
            {
                switch(me->depth()) {
                    case CV_8U:  PUT_ITEM(uchar,  r, c); break;
                    case CV_8S:  PUT_ITEM(schar,  r, c); break;
                    case CV_16U: PUT_ITEM(ushort, r, c); break;
                    case CV_16S: PUT_ITEM(short,  r, c); break;
                    case CV_32S: PUT_ITEM(int,    r, c); break;
                    case CV_32F: PUT_ITEM(float,  r, c); break;
                    case CV_64F: PUT_ITEM(double, r, c); break;
                }
            }

        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(cv::Exception e) {
        LOGD("Mat::nPutD() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::nPutD() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::nPutD()}");
        return 0;
    }
}


} // extern "C"

template<typename T> static int mat_put(cv::Mat* m, int row, int col, int count, char* buff)
{
    if(! m) return 0;
    if(! buff) return 0;

    count *= sizeof(T);
    int rest = ((m->rows - row) * m->cols - col) * (int)m->elemSize();
    if(count>rest) count = rest;
    int res = count;

    if( m->isContinuous() )
    {
        memcpy(m->ptr(row, col), buff, count);
    } else {
        // row by row
        int num = (m->cols - col) * (int)m->elemSize(); // 1st partial row
        if(count<num) num = count;
        uchar* data = m->ptr(row++, col);
        while(count>0){
            memcpy(data, buff, num);
            count -= num;
            buff += num;
            num = m->cols * (int)m->elemSize();
            if(count<num) num = count;
            data = m->ptr(row++, 0);
        }
    }
    return res;
}


extern "C" {

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutB
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jbyteArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutB
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jbyteArray vals)
{
    try {
        LOGD("Mat::nPutB()");
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != CV_8U && me->depth() != CV_8S) return 0; // incompatible type
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_put<char>(me, row, col, count, values);
        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(cv::Exception e) {
        LOGD("Mat::nPutB() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::nPutB() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::nPutB()}");
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutS
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jshortArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutS
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jshortArray vals)
{
    try {
        LOGD("Mat::nPutS()");
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != CV_16U && me->depth() != CV_16S) return 0; // incompatible type
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_put<short>(me, row, col, count, values);
        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(cv::Exception e) {
        LOGD("Mat::nPutS() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::nPutS() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::nPutS()}");
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutI
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jintArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutI
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jintArray vals)
{
    try {
        LOGD("Mat::nPutI()");
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != CV_32S) return 0; // incompatible type
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_put<int>(me, row, col, count, values);
        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(cv::Exception e) {
        LOGD("Mat::nPutI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::nPutI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::nPutI()}");
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutF
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jfloatArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nPutF
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jfloatArray vals)
{
    try {
        LOGD("Mat::nPutF()");
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != CV_32F) return 0; // incompatible type
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_put<float>(me, row, col, count, values);
        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(cv::Exception e) {
        LOGD("Mat::nPutF() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::nPutF() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::nPutF()}");
        return 0;
    }
}


} // extern "C"

template<typename T> int mat_get(cv::Mat* m, int row, int col, int count, char* buff)
{
    if(! m) return 0;
    if(! buff) return 0;

    int bytesToCopy = count * sizeof(T);
    int bytesRestInMat = ((m->rows - row) * m->cols - col) * (int)m->elemSize();
    if(bytesToCopy > bytesRestInMat) bytesToCopy = bytesRestInMat;
    int res = bytesToCopy;

    if( m->isContinuous() )
    {
        memcpy(buff, m->ptr(row, col), bytesToCopy);
    } else {
        // row by row
        int bytesInRow = (m->cols - col) * (int)m->elemSize(); // 1st partial row
        while(bytesToCopy > 0)
        {
            int len = std::min(bytesToCopy, bytesInRow);
            memcpy(buff, m->ptr(row, col), len);
            bytesToCopy -= len;
            buff += len;
            row++;
            col = 0;
            bytesInRow = m->cols * (int)m->elemSize();
        }
    }
    return res;
}

extern "C" {

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetB
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jbyteArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetB
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jbyteArray vals)
{
    try {
        LOGD("Mat::nGetB()");
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != CV_8U && me->depth() != CV_8S) return 0; // incompatible type
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_get<char>(me, row, col, count, values);
        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(cv::Exception e) {
        LOGD("Mat::nGetB() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::nGetB() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::nGetB()}");
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetS
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jshortArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetS
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jshortArray vals)
{
    try {
        LOGD("Mat::nGetS()");
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != CV_16U && me->depth() != CV_16S) return 0; // incompatible type
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_get<short>(me, row, col, count, values);
        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(cv::Exception e) {
        LOGD("Mat::nGetS() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::nGetS() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::nGetS()}");
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetI
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jintArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetI
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jintArray vals)
{
    try {
        LOGD("Mat::nGetI()");
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != CV_32S) return 0; // incompatible type
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_get<int>(me, row, col, count, values);
        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(cv::Exception e) {
        LOGD("Mat::nGetI() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::nGetI() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::nGetI()}");
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetF
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jfloatArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetF
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jfloatArray vals)
{
    try {
        LOGD("Mat::nGetF()");
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != CV_32F) return 0; // incompatible type
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_get<float>(me, row, col, count, values);
        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(cv::Exception e) {
        LOGD("Mat::nGetF() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::nGetF() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::nGetF()}");
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetD
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jdoubleArray vals);

JNIEXPORT jint JNICALL Java_org_opencv_core_Mat_nGetD
    (JNIEnv* env, jclass, jlong self, jint row, jint col, jint count, jdoubleArray vals)
{
    try {
        LOGD("Mat::nGetD()");
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->depth() != CV_64F) return 0; // incompatible type
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
        int res = mat_get<double>(me, row, col, count, values);
        env->ReleasePrimitiveArrayCritical(vals, values, 0);
        return res;
    } catch(cv::Exception e) {
        LOGD("Mat::nGetD() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::nGetD() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::nGetD()}");
        return 0;
    }
}

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Mat_nGet
    (JNIEnv* env, jclass, jlong self, jint row, jint col);

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Mat_nGet
    (JNIEnv* env, jclass, jlong self, jint row, jint col)
{
    try {
        LOGD("Mat::nGet()");
        cv::Mat* me = (cv::Mat*) self;
        if(! self) return 0; // no native object behind
        if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

        jdoubleArray res = env->NewDoubleArray(me->channels());
        if(res){
            jdouble buff[CV_CN_MAX];//me->channels()
            int i;
            switch(me->depth()){
                case CV_8U:  for(i=0; i<me->channels(); i++) buff[i] = *((unsigned char*) me->ptr(row, col) + i); break;
                case CV_8S:  for(i=0; i<me->channels(); i++) buff[i] = *((signed char*)   me->ptr(row, col) + i); break;
                case CV_16U: for(i=0; i<me->channels(); i++) buff[i] = *((unsigned short*)me->ptr(row, col) + i); break;
                case CV_16S: for(i=0; i<me->channels(); i++) buff[i] = *((signed short*)  me->ptr(row, col) + i); break;
                case CV_32S: for(i=0; i<me->channels(); i++) buff[i] = *((int*)           me->ptr(row, col) + i); break;
                case CV_32F: for(i=0; i<me->channels(); i++) buff[i] = *((float*)         me->ptr(row, col) + i); break;
                case CV_64F: for(i=0; i<me->channels(); i++) buff[i] = *((double*)        me->ptr(row, col) + i); break;
            }
            env->SetDoubleArrayRegion(res, 0, me->channels(), buff);
        }
        return res;
    } catch(cv::Exception e) {
        LOGD("Mat::nGet() caught cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return 0;
    } catch (...) {
        LOGD("Mat::nGet() caught unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {Mat::nGet()}");
        return 0;
    }
}

JNIEXPORT jstring JNICALL Java_org_opencv_core_Mat_nDump
  (JNIEnv *env, jclass, jlong self);

JNIEXPORT jstring JNICALL Java_org_opencv_core_Mat_nDump
  (JNIEnv *env, jclass, jlong self)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    try {
            LOGD("Mat::nDump()");
            String s;
            Ptr<Formatted> fmtd = Formatter::get()->format(*me);
            for(const char* str = fmtd->next(); str; str = fmtd->next())
            {
                s = s + String(str);
            }
            return env->NewStringUTF(s.c_str());
        } catch(cv::Exception e) {
            LOGE("Mat::nDump() caught cv::Exception: %s", e.what());
            jclass je = env->FindClass("org/opencv/core/CvException");
            if(!je) je = env->FindClass("java/lang/Exception");
            env->ThrowNew(je, e.what());
            return env->NewStringUTF("ERROR");
        } catch (...) {
            LOGE("Mat::nDump() caught unknown exception (...)");
            jclass je = env->FindClass("java/lang/Exception");
            env->ThrowNew(je, "Unknown exception in JNI code {Mat::nDump()}");
            return env->NewStringUTF("ERROR");
        }
}


} // extern "C"
