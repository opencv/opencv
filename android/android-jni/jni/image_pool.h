#ifndef IMAGE_POOL_H_ANDROID_KDJFKJ
#define IMAGE_POOL_H_ANDROID_KDJFKJ
#include <opencv2/core/core.hpp>
#include <map>

#if ANDROID
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
#endif
class image_pool
{

public:
  image_pool();
  ~image_pool();
  cv::Mat getImage(int i);
  cv::Mat getGrey(int i);
  cv::Mat getYUV(int i);

  int getCount()
  {
    return imagesmap.size();
  }

  /** Adds a mat at the given index - will not do a deep copy, just images[i] = mat
   *
   */
  void addImage(int i, cv::Mat mat);

  /** this function stores the given matrix in the the yuvImagesMap. Also,
   * after this call getGrey will work, as the grey image is just the top
   * half of the YUV mat.
   *
   * \param i index to store yuv image at
   * \param mat the yuv matrix to store
   */
  void addYUVMat(int i, cv::Mat mat);

  //	int addYUV(uchar* buffer, int size, int width, int height, bool grey,int idx);
  //
  //	void getBitmap(int * outintarray, int size, int idx);
private:
  std::map<int, cv::Mat> imagesmap;
  std::map<int, cv::Mat> yuvImagesMap;

};
#endif
