#include "image_pool.h"

#include "yuv420sp2rgb.h"

#include <android/log.h>
#include <opencv2/imgproc/imgproc.hpp>


JNIEXPORT void JNICALL Java_com_opencv_jni_opencvJNI_addYUVtoPool(JNIEnv * env,
		jclass thiz, jlong ppool, jobject _jpool, jbyteArray jbuffer,
		jint jidx, jint jwidth, jint jheight, jboolean jgrey) {
	image_pool *pool = (image_pool *) ppool;

	Ptr<Mat> mat = pool->getYUV(jidx);

	if (mat.empty() || mat->cols != jwidth || mat->rows != jheight * 2) {
		//pool->deleteGrey(jidx);
		mat = new Mat(jheight * 2, jwidth, CV_8UC1);
	}

	jsize sz = env->GetArrayLength(jbuffer);
	uchar* buff = mat->ptr<uchar> (0);

	env->GetByteArrayRegion(jbuffer, 0, sz, (jbyte*) buff);

	pool->addYUVMat(jidx, mat);
	Ptr<Mat> color = pool->getImage(jidx);
	if (color.empty() || color->cols != jwidth || color->rows != jheight) {
		//pool->deleteImage(jidx);
		color = new Mat(jheight, jwidth, CV_8UC3);
	}
	if (!jgrey) {

		//doesn't work unfortunately..
		//cvtColor(*mat,*color, CV_YCrCb2RGB);
		color_convert_common(buff, buff + jwidth * jheight, jwidth, jheight,
				color->ptr<uchar> (0), false);

	}

	if (jgrey) {
		Mat grey;
		pool->getGrey(jidx, grey);

		cvtColor(grey, *color, CV_GRAY2RGB);

	}

	pool->addImage(jidx, color);

}

image_pool::image_pool() {

}

image_pool::~image_pool() {
	__android_log_print(ANDROID_LOG_INFO, "image_pool", "destructor called");
}

cv::Ptr<Mat> image_pool::getImage(int i) {
	return imagesmap[i];
}
void image_pool::getGrey(int i, Mat & grey) {

	cv::Ptr<Mat> tm = yuvImagesMap[i];
	if (tm.empty())
		return;
	grey = (*tm)(Range(0, tm->rows / 2), Range::all());
}
cv::Ptr<Mat> image_pool::getYUV(int i) {

	return yuvImagesMap[i];

}
void image_pool::addYUVMat(int i, cv::Ptr<Mat> mat) {

	yuvImagesMap[i] = mat;

}
void image_pool::addImage(int i, cv::Ptr<Mat> mat) {

	imagesmap[i] = mat;

}

