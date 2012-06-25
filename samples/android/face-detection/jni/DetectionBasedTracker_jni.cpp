#include <DetectionBasedTracker_jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/detection_based_tracker.hpp>

#include <string>
#include <vector>

#include <android/log.h>

#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace std;
using namespace cv;

inline void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
    mat = Mat(v_rect, true);
}

class CascadeDetectorAdapter: public DetectionBasedTracker::IDetector
{
public:
    CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector):
    IDetector(),
    Detector(detector)
    {
	CV_Assert(!detector.empty());
    }
    
    void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
    {
	Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
    }
    
    virtual ~CascadeDetectorAdapter()
    {}
    
private:
    CascadeDetectorAdapter();
    cv::Ptr<cv::CascadeClassifier> Detector;
};

JNIEXPORT jlong JNICALL Java_org_opencv_samples_fd_DetectionBasedTracker_nativeCreateObject
(JNIEnv * jenv, jclass, jstring jFileName, jint faceSize)
{
    const char* jnamestr = jenv->GetStringUTFChars(jFileName, NULL);
    string stdFileName(jnamestr);
    jlong result = 0;

    try
    {
	// TODO: Reimplement using adapter
//	DetectionBasedTracker::Parameters DetectorParams;
//	if (faceSize > 0)
//	    DetectorParams.minObjectSize = faceSize;
//	result = (jlong)new DetectionBasedTracker(stdFileName, DetectorParams);
    }
    catch(cv::Exception e)
    {
	LOGD("nativeCreateObject catched cv::Exception: %s", e.what());
	jclass je = jenv->FindClass("org/opencv/core/CvException");
	if(!je)
	    je = jenv->FindClass("java/lang/Exception");
	jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
	LOGD("nativeCreateObject catched unknown exception");
	jclass je = jenv->FindClass("java/lang/Exception");
	jenv->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1VideoCapture__()}");
	return 0;
    }

    return result;
}

JNIEXPORT void JNICALL Java_org_opencv_samples_fd_DetectionBasedTracker_nativeDestroyObject
(JNIEnv * jenv, jclass, jlong thiz)
{
    try
    {
	((DetectionBasedTracker*)thiz)->stop();
	delete (DetectionBasedTracker*)thiz;
    }
    catch(cv::Exception e)
    {
	LOGD("nativeestroyObject catched cv::Exception: %s", e.what());
	jclass je = jenv->FindClass("org/opencv/core/CvException");
	if(!je)
	    je = jenv->FindClass("java/lang/Exception");
	jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
	LOGD("nativeDestroyObject catched unknown exception");
	jclass je = jenv->FindClass("java/lang/Exception");
	jenv->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1VideoCapture__()}");
    }
}

JNIEXPORT void JNICALL Java_org_opencv_samples_fd_DetectionBasedTracker_nativeStart
(JNIEnv * jenv, jclass, jlong thiz)
{
    try
    {
	((DetectionBasedTracker*)thiz)->run();
    }
    catch(cv::Exception e)
    {
	LOGD("nativeStart catched cv::Exception: %s", e.what());
	jclass je = jenv->FindClass("org/opencv/core/CvException");
	if(!je)
	    je = jenv->FindClass("java/lang/Exception");
	jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
	LOGD("nativeStart catched unknown exception");
	jclass je = jenv->FindClass("java/lang/Exception");
	jenv->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1VideoCapture__()}");
    }
}

JNIEXPORT void JNICALL Java_org_opencv_samples_fd_DetectionBasedTracker_nativeStop
(JNIEnv * jenv, jclass, jlong thiz)
{
    try
    {
	((DetectionBasedTracker*)thiz)->stop();
    }
    catch(cv::Exception e)
    {
	LOGD("nativeStop catched cv::Exception: %s", e.what());
	jclass je = jenv->FindClass("org/opencv/core/CvException");
	if(!je)
	    je = jenv->FindClass("java/lang/Exception");
	jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
	LOGD("nativeStop catched unknown exception");
	jclass je = jenv->FindClass("java/lang/Exception");
	jenv->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1VideoCapture__()}");
    }
}

JNIEXPORT void JNICALL Java_org_opencv_samples_fd_DetectionBasedTracker_nativeSetFaceSize
(JNIEnv * jenv, jclass, jlong thiz, jint faceSize)
{
    try
    {
	if (faceSize > 0)
	{
	// TODO: Reimplement using adapter
//        DetectionBasedTracker::Parameters DetectorParams = ((DetectionBasedTracker*)thiz)->getParameters();
//        DetectorParams.minObjectSize = faceSize;
//        ((DetectionBasedTracker*)thiz)->setParameters(DetectorParams);
	}
    }
    catch(cv::Exception e)
    {
	LOGD("nativeStop catched cv::Exception: %s", e.what());
	jclass je = jenv->FindClass("org/opencv/core/CvException");
	if(!je)
	    je = jenv->FindClass("java/lang/Exception");
	jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
	LOGD("nativeSetFaceSize catched unknown exception");
	jclass je = jenv->FindClass("java/lang/Exception");
	jenv->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1VideoCapture__()}");
    }
}


JNIEXPORT void JNICALL Java_org_opencv_samples_fd_DetectionBasedTracker_nativeDetect
(JNIEnv * jenv, jclass, jlong thiz, jlong imageGray, jlong faces)
{
    try
    {
	vector<Rect> RectFaces;
	((DetectionBasedTracker*)thiz)->process(*((Mat*)imageGray));
	((DetectionBasedTracker*)thiz)->getObjects(RectFaces);
	*((Mat*)faces) = Mat(RectFaces, true);
    }
    catch(cv::Exception e)
    {
	LOGD("nativeCreateObject catched cv::Exception: %s", e.what());
	jclass je = jenv->FindClass("org/opencv/core/CvException");
	if(!je)
	    je = jenv->FindClass("java/lang/Exception");
	jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
	LOGD("nativeDetect catched unknown exception");
	jclass je = jenv->FindClass("java/lang/Exception");
	jenv->ThrowNew(je, "Unknown exception in JNI code {highgui::VideoCapture_n_1VideoCapture__()}");
    }
}
