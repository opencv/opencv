#ifndef IMAGE_POOL_H
#define IMAGE_POOL_H
#include <opencv2/core/core.hpp>
#include <jni.h>
#include <map>
using namespace cv;

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved);
//
//JNIEXPORT jobject JNICALL Java_com_opencv_jni_opencvJNI_getBitmapBuffer(
//		JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);


JNIEXPORT void JNICALL Java_com_opencv_jni_opencvJNI_addYUVtoPool
  (JNIEnv *, jclass, jlong, jobject, jbyteArray, jint, jint, jint, jboolean);

#ifdef __cplusplus
}
#endif

//bool yuv2mat2(char *data, int size, int width, int height, bool grey, Mat& mat);


class image_pool {
	std::map<int, Ptr< Mat> > imagesmap;
	std::map<int, Ptr< Mat> > yuvImagesMap;
	//uchar * mbuffer;
	//int length;
public:
	image_pool();
	~image_pool();
	cv::Ptr<Mat> getImage(int i);

	void getGrey(int i, Mat & grey);
	cv::Ptr<Mat> getYUV(int i);

	int getCount(){
		return imagesmap.size();
	}

	void addImage(int i, Ptr< Mat> mat);
	/** this function stores the given matrix in the the yuvImagesMap. Also,
	 * after this call getGrey will work, as the grey image is just the top
	 * half of the YUV mat.
	 *
	 * \param i index to store yuv image at
	 * \param mat the yuv matrix to store
	 */
	void addYUVMat(int i, Ptr< Mat> mat);


	int addYUV(uchar* buffer, int size, int width, int height, bool grey,int idx);

	void getBitmap(int * outintarray, int size, int idx);


};
#endif
