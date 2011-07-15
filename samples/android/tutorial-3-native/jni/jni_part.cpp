#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

using namespace std;
using namespace cv;

extern "C" {
JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial3_Sample3View_FindFeatures(JNIEnv* env, jobject thiz, jint width, jint height, jbyteArray yuv, jintArray rgba)
{
    jbyte* _yuv  = env->GetByteArrayElements(yuv, 0);
    jint*  _rgba = env->GetIntArrayElements(rgba, 0);

    Mat myuv(height + height/2, width, CV_8UC1, (unsigned char *)_yuv);
    Mat mrgba(height, width, CV_8UC4, (unsigned char *)_rgba);
    Mat mgray(height, width, CV_8UC1, (unsigned char *)_yuv);

    cvtColor(myuv, mrgba, CV_YUV420i2BGR, 4);

    vector<KeyPoint> v;

    FastFeatureDetector detector(50);
    detector.detect(mgray, v);
    for( size_t i = 0; i < v.size(); i++ )
        circle(mrgba, Point(v[i].pt.x, v[i].pt.y), 10, Scalar(0,0,255,255));

    env->ReleaseIntArrayElements(rgba, _rgba, 0);
    env->ReleaseByteArrayElements(yuv, _yuv, 0);
}

}
