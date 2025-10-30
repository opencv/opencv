#ifndef VIDEOIO_CONVERTERS_HPP
#define VIDEOIO_CONVERTERS_HPP

#include <jni.h>
#include "opencv_java.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"

class JavaStreamReader : public cv::IStreamReader
{
public:
    JavaStreamReader(JNIEnv* env, jobject obj);
    ~JavaStreamReader();
    long long read(char* buffer, long long size) CV_OVERRIDE;
    long long seek(long long offset, int way) CV_OVERRIDE;

private:
    JavaVM* vm;
    jobject obj;
    jmethodID m_read, m_seek;
};

jobject vector_VideoCaptureAPIs_to_List(JNIEnv* env, std::vector<cv::VideoCaptureAPIs>& vs);

#endif
