#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::gpu;

#include <android/log.h>

#define LOG_TAG "Cuda"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

extern "C" {
JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial4_Tutorial4Activity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba);

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial4_Tutorial4Activity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    Mat& mGr  = *(Mat*)addrGray;
    Mat& mRgb = *(Mat*)addrRgba;
    vector<KeyPoint> keypoints;
    GpuMat grGpu(mGr);

    FAST_GPU fast(50);
    fast(grGpu, GpuMat(), keypoints);
    for( unsigned int i = 0; i < keypoints.size(); i++ )
    {
        const KeyPoint& kp = keypoints[i];
        circle(mRgb, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
    }
}
}
