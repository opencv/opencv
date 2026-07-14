// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/core.hpp"

#define LOG_TAG "org.opencv.core.CleanableMat"
#include "common.h"
#include <iostream>

using namespace cv;

extern "C" {
    //
    //  native support for java finalize() or cleaners
    //  static void CleanableMat::n_delete( __int64 self )
    //

    JNIEXPORT void JNICALL Java_org_opencv_core_CleanableMat_n_1delete
    (JNIEnv*, jclass, jlong self);

    JNIEXPORT void JNICALL Java_org_opencv_core_CleanableMat_n_1delete
    (JNIEnv*, jclass, jlong self)
    {
        // LOGD("CleanableMat.n_delete() called\n");
        delete (Mat*) self;
    }
}
