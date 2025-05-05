#ifndef VIDEOIO_CONVERTERS_HPP
#define VIDEOIO_CONVERTERS_HPP

#include <jni.h>
#include "opencv_java.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio/videoio.hpp"

class JavaStreamReader : public cv::IStreamReader
{
public:
    JavaStreamReader(JNIEnv* _env, jclass _jobject);
    long long read(char* buffer, long long size) CV_OVERRIDE;
    long long seek(long long offset, int way) CV_OVERRIDE;

private:
    JNIEnv* env;
    jclass jobject;
};

#endif
