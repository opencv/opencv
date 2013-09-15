#include <jni.h>

#ifndef _Included_org_opencv_engine_OpenCVLibraryInfo
#define _Included_org_opencv_engine_OpenCVLibraryInfo
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_open
  (JNIEnv *, jobject, jstring);

JNIEXPORT jstring JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_getPackageName
  (JNIEnv *, jobject, jlong);

JNIEXPORT jstring JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_getLibraryList
  (JNIEnv *, jobject, jlong);

JNIEXPORT jstring JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_getVersionName
  (JNIEnv *, jobject, jlong);

JNIEXPORT void JNICALL Java_org_opencv_engine_OpenCVLibraryInfo_close
  (JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif
