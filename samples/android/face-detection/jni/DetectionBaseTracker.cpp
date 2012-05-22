#include <DetectionBaseTracker.h>
#include <opencv2/core/core.hpp> 
#include <opencv2/contrib/detection_based_tracker.hpp>

#include <string>
#include <vector>

using namespace std;
using namespace cv;

vector<Rect> RectFaces;

inline void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
    mat = Mat(v_rect, true);
}

JNIEXPORT jlong JNICALL Java_org_opencv_samples_fd_DetectionBaseTracker_nativeCreateObject
(JNIEnv * jenv, jclass jobj, jstring jFileName, jint faceSize)
{
    const char* jnamestr = jenv->GetStringUTFChars(jFileName, NULL);
    string stdFileName(jnamestr);
    DetectionBasedTracker::Parameters DetectorParams;
    if (faceSize > 0)
	DetectorParams.minObjectSize = faceSize;
    return (jlong)new DetectionBasedTracker(stdFileName, DetectorParams);
}

JNIEXPORT void JNICALL Java_org_opencv_samples_fd_DetectionBaseTracker_nativeDestroyObject
(JNIEnv * jenv, jclass jobj, jlong thiz)
{
    delete (DetectionBasedTracker*)thiz;
}

JNIEXPORT void JNICALL Java_org_opencv_samples_fd_DetectionBaseTracker_nativeStart
(JNIEnv * jenv, jclass jobj, jlong thiz)
{
    ((DetectionBasedTracker*)thiz)->run();
}

JNIEXPORT void JNICALL Java_org_opencv_samples_fd_DetectionBaseTracker_nativeStop
(JNIEnv * jenv, jclass jobj, jlong thiz)
{
    ((DetectionBasedTracker*)thiz)->stop();
}

JNIEXPORT void JNICALL Java_org_opencv_samples_fd_DetectionBaseTracker_nativeDetect
(JNIEnv * jenv, jclass jobj, jlong thiz, jlong imageGray, jlong faces)
{
    ((DetectionBasedTracker*)thiz)->process(*((Mat*)imageGray));
    ((DetectionBasedTracker*)thiz)->getObjects(RectFaces);
    vector_Rect_to_Mat(RectFaces, *((Mat*)faces));
}