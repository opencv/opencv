#include "image_pool.h"

#include "yuv420sp2rgb.h"

#include <android/log.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <jni.h>
using namespace cv;

#define  LOG_TAG    "libandroid-opencv"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

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

  if (mat.empty() || mat.size() != buff_size)
  {
    mat.create(buff_size, CV_8UC1);
  }

  jsize sz = env->GetArrayLength(jbuffer);
  uchar* buff = mat.ptr<uchar> (0);

  env->GetByteArrayRegion(jbuffer, 0, sz, (jbyte*)buff);

  pool->addYUVMat(jidx, mat);

  Mat color = pool->getImage(jidx);

  if (!jgrey)
  {

    if (color.cols != jwidth || color.rows != jheight || color.channels() != 3)
    {
      color.create(jheight, jwidth, CV_8UC3);
    }
    //doesn't work unfortunately..
    //TODO cvtColor(mat,color, CV_YCrCb2RGB);
    color_convert_common(buff, buff + jwidth * jheight, jwidth, jheight, color.ptr<uchar> (0), false);
  }

  if (jgrey)
  {
    Mat grey = pool->getGrey(jidx);
    color = grey;
  }

  pool->addImage(jidx, color);

}

image_pool::image_pool()
{

}

image_pool::~image_pool()
{
  __android_log_print(ANDROID_LOG_INFO, "image_pool", "destructor called");
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
  //doesn't work unfortunately..
  //TODO cvtColor(mat,color, CV_YCrCb2RGB);
  color_convert_common(buff, buff + width * height, width, height, out_buff, false);
}
