// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OBJDETECT_CONVERTERS_HPP
#define	OBJDETECT_CONVERTERS_HPP

#include <jni.h>
#include "opencv_java.hpp"
#include "opencv2/core.hpp"

void Copy_vector_NativeByteArray_to_List(JNIEnv* env, std::vector<std::string>& vs, jobject list);

#endif	/* DNN_CONVERTERS_HPP */
