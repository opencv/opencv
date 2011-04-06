#include "image_pool.h"

#include "yuv420sp2rgb.h"

#include "android_logger.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <cstdlib>
#include <jni.h>
#ifdef __cplusplus
extern "C"
{
#endif

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved);
//
//JNIEXPORT jobject JNICALL Java_com_opencv_jni_opencvJNI_getBitmapBuffer(
//		JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);


JNIEXPORT void JNICALL Java_com_opencv_jni_opencvJNI_addYUVtoPool(JNIEnv *, jclass, jlong, jobject, jbyteArray, jint,
    jint, jint, jboolean);

#ifdef __cplusplus
}
#endif

using namespace cv;

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
  JNIEnv *env;
  LOGI("JNI_OnLoad called for opencv");
  return JNI_VERSION_1_4;
}

JNIEXPORT void JNICALL Java_com_opencv_jni_opencvJNI_addYUVtoPool(JNIEnv * env, jclass thiz, jlong ppool,
    jobject _jpool, jbyteArray jbuffer, jint jidx,
    jint jwidth, jint jheight, jboolean jgrey)
{
  int buff_height = jheight + (jheight / 2);
  Size buff_size(jwidth, buff_height);
  image_pool *pool = (image_pool *)ppool;

  Mat mat = pool->getYUV(jidx);
  //create is smart and only copies if the buffer size is different
  mat.create(buff_size, CV_8UC1);
  {
    uchar* buff = mat.ptr<uchar> (0);
    jsize sz = env->GetArrayLength(jbuffer);
    //http://elliotth.blogspot.com/2007/03/optimizing-jni-array-access.html
    env->GetByteArrayRegion(jbuffer, 0, sz, (jbyte*)buff);
  }
  pool->addYUVMat(jidx, mat);

  Mat color;
  if (jgrey)
  {
    Mat grey = pool->getGrey(jidx);
    color = grey;
  }
  else
  {
    color = pool->getImage(jidx);
    pool->convertYUVtoColor(jidx, color);
  }
  pool->addImage(jidx, color);
}

image_pool::image_pool()
{

}

image_pool::~image_pool()
{

}

Mat image_pool::getImage(int i)
{
  return imagesmap[i];
}
Mat image_pool::getGrey(int i)
{
  Mat tm = yuvImagesMap[i];
  if (tm.empty())
    return tm;
  return tm(Range(0, tm.rows * (2.0f / 3)), Range::all());
}
Mat image_pool::getYUV(int i)
{
  return yuvImagesMap[i];
}
void image_pool::addYUVMat(int i, Mat mat)
{
  yuvImagesMap[i] = mat;
}
void image_pool::addImage(int i, Mat mat)
{
  imagesmap[i] = mat;
}

void image_pool::convertYUVtoColor(int i, cv::Mat& out)
{
  Mat yuv = getYUV(i);
  if (yuv.empty())
    return;
  int width = yuv.cols;
  int height = yuv.rows * (2.0f / 3);
  out.create(height, width, CV_8UC3);
  const unsigned char* buff = yuv.ptr<unsigned char> (0);
  unsigned char* out_buff = out.ptr<unsigned char> (0);
  color_convert_common(buff, buff + width * height, width, height, out_buff, false);
}

void copyMatToBuffer(char* buffer, const cv::Mat& mat)
{
  memcpy(buffer, mat.data, mat.rows * mat.cols * mat.step1());
}
void copyBufferToMat(cv::Mat& mat, const char* buffer)
{
  memcpy(mat.data, buffer, mat.rows * mat.cols * mat.step1());
}

void RGB2BGR(const Mat& in, Mat& out)
{
  cvtColor(in, out, CV_RGB2BGR);
}
