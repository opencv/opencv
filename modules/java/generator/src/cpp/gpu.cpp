#define LOG_TAG "org.opencv.gpu"

#include "common.h"

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/gpumat.hpp"

using namespace cv;
using namespace cv::gpu;

/// throw java exception
static void throwJavaException(JNIEnv *env, const std::exception *e, const char *method) {
  std::string what = "unknown exception";
  jclass je = 0;

  if(e) {
    std::string exception_type = "std::exception";

    if(dynamic_cast<const cv::Exception*>(e)) {
      exception_type = "cv::Exception";
      je = env->FindClass("org/opencv/core/CvException");
    }

    what = exception_type + ": " + e->what();
  }

  if(!je) je = env->FindClass("java/lang/Exception");
  env->ThrowNew(je, what.c_str());

  LOGE("%s caught %s", method, what.c_str());
  (void)method;        // avoid "unused" warning
}


extern "C" {


//
//  bool deviceSupports(cv::gpu::FeatureSet feature_set)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_Gpu_deviceSupports_10 (JNIEnv*, jclass, jint);

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_Gpu_deviceSupports_10
  (JNIEnv* env, jclass , jint feature_set)
{
    static const char method_name[] = "gpu::deviceSupports_10()";
    try {
        LOGD("%s", method_name);

        bool _retval_ = deviceSupports( (cv::gpu::FeatureSet)feature_set );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  int getCudaEnabledDeviceCount()
//

JNIEXPORT jint JNICALL Java_org_opencv_gpu_Gpu_getCudaEnabledDeviceCount_10 (JNIEnv*, jclass);

JNIEXPORT jint JNICALL Java_org_opencv_gpu_Gpu_getCudaEnabledDeviceCount_10
  (JNIEnv* env, jclass )
{
    static const char method_name[] = "gpu::getCudaEnabledDeviceCount_10()";
    try {
        LOGD("%s", method_name);

        int _retval_ = getCudaEnabledDeviceCount(  );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  int getDevice()
//

JNIEXPORT jint JNICALL Java_org_opencv_gpu_Gpu_getDevice_10 (JNIEnv*, jclass);

JNIEXPORT jint JNICALL Java_org_opencv_gpu_Gpu_getDevice_10
  (JNIEnv* env, jclass )
{
    static const char method_name[] = "gpu::getDevice_10()";
    try {
        LOGD("%s", method_name);

        int _retval_ = getDevice(  );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  void printCudaDeviceInfo(int device)
//

JNIEXPORT void JNICALL Java_org_opencv_gpu_Gpu_printCudaDeviceInfo_10 (JNIEnv*, jclass, jint);

JNIEXPORT void JNICALL Java_org_opencv_gpu_Gpu_printCudaDeviceInfo_10
  (JNIEnv* env, jclass , jint device)
{
    static const char method_name[] = "gpu::printCudaDeviceInfo_10()";
    try {
        LOGD("%s", method_name);

        printCudaDeviceInfo( (int)device );
        return;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return;
}



//
//  void printShortCudaDeviceInfo(int device)
//

JNIEXPORT void JNICALL Java_org_opencv_gpu_Gpu_printShortCudaDeviceInfo_10 (JNIEnv*, jclass, jint);

JNIEXPORT void JNICALL Java_org_opencv_gpu_Gpu_printShortCudaDeviceInfo_10
  (JNIEnv* env, jclass , jint device)
{
    static const char method_name[] = "gpu::printShortCudaDeviceInfo_10()";
    try {
        LOGD("%s", method_name);

        printShortCudaDeviceInfo( (int)device );
        return;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return;
}



//
//  void resetDevice()
//

JNIEXPORT void JNICALL Java_org_opencv_gpu_Gpu_resetDevice_10 (JNIEnv*, jclass);

JNIEXPORT void JNICALL Java_org_opencv_gpu_Gpu_resetDevice_10
  (JNIEnv* env, jclass )
{
    static const char method_name[] = "gpu::resetDevice_10()";
    try {
        LOGD("%s", method_name);

        resetDevice();
        return;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return;
}



//
//  void setDevice(int device)
//

JNIEXPORT void JNICALL Java_org_opencv_gpu_Gpu_setDevice_10 (JNIEnv*, jclass, jint);

JNIEXPORT void JNICALL Java_org_opencv_gpu_Gpu_setDevice_10
  (JNIEnv* env, jclass , jint device)
{
    static const char method_name[] = "gpu::setDevice_10()";
    try {
        LOGD("%s", method_name);

        setDevice( (int)device );
        return;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return;
}



//
//   DeviceInfo::DeviceInfo()
//

JNIEXPORT jlong JNICALL Java_org_opencv_gpu_DeviceInfo_DeviceInfo_10 (JNIEnv*, jclass);

JNIEXPORT jlong JNICALL Java_org_opencv_gpu_DeviceInfo_DeviceInfo_10
  (JNIEnv* env, jclass )
{
    static const char method_name[] = "gpu::DeviceInfo_10()";
    try {
        LOGD("%s", method_name);

        DeviceInfo* _retval_ = new DeviceInfo(  );
        return (jlong) _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//   DeviceInfo::DeviceInfo(int device_id)
//

JNIEXPORT jlong JNICALL Java_org_opencv_gpu_DeviceInfo_DeviceInfo_11 (JNIEnv*, jclass, jint);

JNIEXPORT jlong JNICALL Java_org_opencv_gpu_DeviceInfo_DeviceInfo_11
  (JNIEnv* env, jclass , jint device_id)
{
    static const char method_name[] = "gpu::DeviceInfo_11()";
    try {
        LOGD("%s", method_name);

        DeviceInfo* _retval_ = new DeviceInfo( (int)device_id );
        return (jlong) _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  int DeviceInfo::deviceID()
//

JNIEXPORT jint JNICALL Java_org_opencv_gpu_DeviceInfo_deviceID_10 (JNIEnv*, jclass, jlong);

JNIEXPORT jint JNICALL Java_org_opencv_gpu_DeviceInfo_deviceID_10
  (JNIEnv* env, jclass , jlong self)
{
    static const char method_name[] = "gpu::deviceID_10()";
    try {
        LOGD("%s", method_name);
        DeviceInfo* me = (DeviceInfo*) self; //TODO: check for NULL
        int _retval_ = me->deviceID(  );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  size_t DeviceInfo::freeMemory()
//

JNIEXPORT jlong JNICALL Java_org_opencv_gpu_DeviceInfo_freeMemory_10 (JNIEnv*, jclass, jlong);

JNIEXPORT jlong JNICALL Java_org_opencv_gpu_DeviceInfo_freeMemory_10
  (JNIEnv* env, jclass , jlong self)
{
    static const char method_name[] = "gpu::freeMemory_10()";
    try {
        LOGD("%s", method_name);
        DeviceInfo* me = (DeviceInfo*) self; //TODO: check for NULL
        size_t _retval_ = me->freeMemory(  );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  bool DeviceInfo::isCompatible()
//

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_DeviceInfo_isCompatible_10 (JNIEnv*, jclass, jlong);

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_DeviceInfo_isCompatible_10
  (JNIEnv* env, jclass , jlong self)
{
    static const char method_name[] = "gpu::isCompatible_10()";
    try {
        LOGD("%s", method_name);
        DeviceInfo* me = (DeviceInfo*) self; //TODO: check for NULL
        bool _retval_ = me->isCompatible(  );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  int DeviceInfo::majorVersion()
//

JNIEXPORT jint JNICALL Java_org_opencv_gpu_DeviceInfo_majorVersion_10 (JNIEnv*, jclass, jlong);

JNIEXPORT jint JNICALL Java_org_opencv_gpu_DeviceInfo_majorVersion_10
  (JNIEnv* env, jclass , jlong self)
{
    static const char method_name[] = "gpu::majorVersion_10()";
    try {
        LOGD("%s", method_name);
        DeviceInfo* me = (DeviceInfo*) self; //TODO: check for NULL
        int _retval_ = me->majorVersion(  );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  int DeviceInfo::minorVersion()
//

JNIEXPORT jint JNICALL Java_org_opencv_gpu_DeviceInfo_minorVersion_10 (JNIEnv*, jclass, jlong);

JNIEXPORT jint JNICALL Java_org_opencv_gpu_DeviceInfo_minorVersion_10
  (JNIEnv* env, jclass , jlong self)
{
    static const char method_name[] = "gpu::minorVersion_10()";
    try {
        LOGD("%s", method_name);
        DeviceInfo* me = (DeviceInfo*) self; //TODO: check for NULL
        int _retval_ = me->minorVersion(  );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  int DeviceInfo::multiProcessorCount()
//

JNIEXPORT jint JNICALL Java_org_opencv_gpu_DeviceInfo_multiProcessorCount_10 (JNIEnv*, jclass, jlong);

JNIEXPORT jint JNICALL Java_org_opencv_gpu_DeviceInfo_multiProcessorCount_10
  (JNIEnv* env, jclass , jlong self)
{
    static const char method_name[] = "gpu::multiProcessorCount_10()";
    try {
        LOGD("%s", method_name);
        DeviceInfo* me = (DeviceInfo*) self; //TODO: check for NULL
        int _retval_ = me->multiProcessorCount(  );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  string DeviceInfo::name()
//

JNIEXPORT jstring JNICALL Java_org_opencv_gpu_DeviceInfo_name_10 (JNIEnv*, jclass, jlong);

JNIEXPORT jstring JNICALL Java_org_opencv_gpu_DeviceInfo_name_10
  (JNIEnv* env, jclass , jlong self)
{
    static const char method_name[] = "gpu::name_10()";
    try {
        LOGD("%s", method_name);
        DeviceInfo* me = (DeviceInfo*) self; //TODO: check for NULL
        string _retval_ = me->name(  );
        return env->NewStringUTF(_retval_.c_str());
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return env->NewStringUTF("");
}



//
//  void DeviceInfo::queryMemory(size_t& totalMemory, size_t& freeMemory)
//

JNIEXPORT void JNICALL Java_org_opencv_gpu_DeviceInfo_queryMemory_10 (JNIEnv*, jclass, jlong, jdoubleArray, jdoubleArray);

JNIEXPORT void JNICALL Java_org_opencv_gpu_DeviceInfo_queryMemory_10
(JNIEnv* env, jclass , jlong self, jdoubleArray totalMemory_out, jdoubleArray freeMemory_out)
{
    static const char method_name[] = "gpu::queryMemory_10()";
    try {
        LOGD("%s", method_name);
        DeviceInfo* me = (DeviceInfo*) self; //TODO: check for NULL
        size_t totalMemory;
        size_t freeMemory;
        me->queryMemory( totalMemory, freeMemory );
        jdouble tmp_totalMemory[1] = {(jdouble)totalMemory};
        env->SetDoubleArrayRegion(totalMemory_out, 0, 1, tmp_totalMemory);
        jdouble tmp_freeMemory[1] = {(jdouble)freeMemory};
        env->SetDoubleArrayRegion(freeMemory_out, 0, 1, tmp_freeMemory);
        return;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return;
}



//
//  size_t DeviceInfo::sharedMemPerBlock()
//

JNIEXPORT jlong JNICALL Java_org_opencv_gpu_DeviceInfo_sharedMemPerBlock_10 (JNIEnv*, jclass, jlong);

JNIEXPORT jlong JNICALL Java_org_opencv_gpu_DeviceInfo_sharedMemPerBlock_10
  (JNIEnv* env, jclass , jlong self)
{
    static const char method_name[] = "gpu::sharedMemPerBlock_10()";
    try {
        LOGD("%s", method_name);
        DeviceInfo* me = (DeviceInfo*) self; //TODO: check for NULL
        size_t _retval_ = me->sharedMemPerBlock(  );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  bool DeviceInfo::supports(cv::gpu::FeatureSet feature_set)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_DeviceInfo_supports_10 (JNIEnv*, jclass, jlong, jint);

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_DeviceInfo_supports_10
  (JNIEnv* env, jclass , jlong self, jint feature_set)
{
    static const char method_name[] = "gpu::supports_10()";
    try {
        LOGD("%s", method_name);
        DeviceInfo* me = (DeviceInfo*) self; //TODO: check for NULL
        bool _retval_ = me->supports( (cv::gpu::FeatureSet)feature_set );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  size_t DeviceInfo::totalMemory()
//

JNIEXPORT jlong JNICALL Java_org_opencv_gpu_DeviceInfo_totalMemory_10 (JNIEnv*, jclass, jlong);

JNIEXPORT jlong JNICALL Java_org_opencv_gpu_DeviceInfo_totalMemory_10
  (JNIEnv* env, jclass , jlong self)
{
    static const char method_name[] = "gpu::totalMemory_10()";
    try {
        LOGD("%s", method_name);
        DeviceInfo* me = (DeviceInfo*) self; //TODO: check for NULL
        size_t _retval_ = me->totalMemory(  );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  native support for java finalize()
//  static void DeviceInfo::delete( __int64 self )
//
JNIEXPORT void JNICALL Java_org_opencv_gpu_DeviceInfo_delete(JNIEnv*, jclass, jlong);

JNIEXPORT void JNICALL Java_org_opencv_gpu_DeviceInfo_delete
  (JNIEnv*, jclass, jlong self)
{
    delete (DeviceInfo*) self;
}


//
// static bool TargetArchs::builtWith(cv::gpu::FeatureSet feature_set)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_builtWith_10 (JNIEnv*, jclass, jint);

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_builtWith_10
  (JNIEnv* env, jclass , jint feature_set)
{
    static const char method_name[] = "gpu::builtWith_10()";
    try {
        LOGD("%s", method_name);

        bool _retval_ = TargetArchs::builtWith( (cv::gpu::FeatureSet)feature_set );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
// static bool TargetArchs::has(int major, int minor)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_has_10 (JNIEnv*, jclass, jint, jint);

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_has_10
  (JNIEnv* env, jclass , jint major, jint minor)
{
    static const char method_name[] = "gpu::has_10()";
    try {
        LOGD("%s", method_name);

        bool _retval_ = TargetArchs::has( (int)major, (int)minor );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
// static bool TargetArchs::hasBin(int major, int minor)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_hasBin_10 (JNIEnv*, jclass, jint, jint);

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_hasBin_10
  (JNIEnv* env, jclass , jint major, jint minor)
{
    static const char method_name[] = "gpu::hasBin_10()";
    try {
        LOGD("%s", method_name);

        bool _retval_ = TargetArchs::hasBin( (int)major, (int)minor );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
// static bool TargetArchs::hasEqualOrGreater(int major, int minor)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_hasEqualOrGreater_10 (JNIEnv*, jclass, jint, jint);

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_hasEqualOrGreater_10
  (JNIEnv* env, jclass , jint major, jint minor)
{
    static const char method_name[] = "gpu::hasEqualOrGreater_10()";
    try {
        LOGD("%s", method_name);

        bool _retval_ = TargetArchs::hasEqualOrGreater( (int)major, (int)minor );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
// static bool TargetArchs::hasEqualOrGreaterBin(int major, int minor)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_hasEqualOrGreaterBin_10 (JNIEnv*, jclass, jint, jint);

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_hasEqualOrGreaterBin_10
  (JNIEnv* env, jclass , jint major, jint minor)
{
    static const char method_name[] = "gpu::hasEqualOrGreaterBin_10()";
    try {
        LOGD("%s", method_name);

        bool _retval_ = TargetArchs::hasEqualOrGreaterBin( (int)major, (int)minor );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
// static bool TargetArchs::hasEqualOrGreaterPtx(int major, int minor)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_hasEqualOrGreaterPtx_10 (JNIEnv*, jclass, jint, jint);

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_hasEqualOrGreaterPtx_10
  (JNIEnv* env, jclass , jint major, jint minor)
{
    static const char method_name[] = "gpu::hasEqualOrGreaterPtx_10()";
    try {
        LOGD("%s", method_name);

        bool _retval_ = TargetArchs::hasEqualOrGreaterPtx( (int)major, (int)minor );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
// static bool TargetArchs::hasEqualOrLessPtx(int major, int minor)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_hasEqualOrLessPtx_10 (JNIEnv*, jclass, jint, jint);

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_hasEqualOrLessPtx_10
  (JNIEnv* env, jclass , jint major, jint minor)
{
    static const char method_name[] = "gpu::hasEqualOrLessPtx_10()";
    try {
        LOGD("%s", method_name);

        bool _retval_ = TargetArchs::hasEqualOrLessPtx( (int)major, (int)minor );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
// static bool TargetArchs::hasPtx(int major, int minor)
//

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_hasPtx_10 (JNIEnv*, jclass, jint, jint);

JNIEXPORT jboolean JNICALL Java_org_opencv_gpu_TargetArchs_hasPtx_10
  (JNIEnv* env, jclass , jint major, jint minor)
{
    static const char method_name[] = "gpu::hasPtx_10()";
    try {
        LOGD("%s", method_name);

        bool _retval_ = TargetArchs::hasPtx( (int)major, (int)minor );
        return _retval_;
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    return 0;
}



//
//  native support for java finalize()
//  static void TargetArchs::delete( __int64 self )
//
JNIEXPORT void JNICALL Java_org_opencv_gpu_TargetArchs_delete(JNIEnv*, jclass, jlong);

JNIEXPORT void JNICALL Java_org_opencv_gpu_TargetArchs_delete
  (JNIEnv*, jclass, jlong self)
{
    delete (TargetArchs*) self;
}


} // extern "C"
