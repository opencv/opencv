// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "common.h"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_FEATURES
#  include "opencv2/features.hpp"
#endif

#ifdef HAVE_OPENCV_VIDEO
#  include "opencv2/video.hpp"
#endif

#ifdef HAVE_OPENCV_CONTRIB
#  include "opencv2/contrib.hpp"
#endif

extern "C" {

JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* )
{
    JNIEnv* env;
    if (vm->GetEnv((void**) &env, JNI_VERSION_1_6) != JNI_OK)
        return -1;

    /* get class with (*env)->FindClass */
    /* register methods with (*env)->RegisterNatives */

    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM*, void*)
{
  //do nothing
}

} // extern "C"
