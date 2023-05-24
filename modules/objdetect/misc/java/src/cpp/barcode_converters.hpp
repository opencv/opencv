// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Author: darkliang

#ifndef BARCODE_CONVERTERS_HPP
#define BARCODE_CONVERTERS_HPP

#include <jni.h>
#include "opencv_java.hpp"
#include "opencv2/core.hpp"
#include "opencv2/objdetect/barcode.hpp"


using namespace cv::barcode;

void Copy_vector_BarcodeType_to_List(JNIEnv* env, std::vector<cv::barcode::BarcodeType>& vs, jobject list);

#endif    /* BARCODE_CONVERTERS_HPP */
