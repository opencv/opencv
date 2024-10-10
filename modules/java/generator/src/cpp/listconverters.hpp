// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Author: abratchik

#ifndef LISTCONVERTERS_HPP
#define	LISTCONVERTERS_HPP

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"

jobject vector_String_to_List(JNIEnv* env, std::vector<cv::String>& vs);

std::vector<cv::String> List_to_vector_String(JNIEnv* env, jobject list);

void Copy_vector_String_to_List(JNIEnv* env, std::vector<cv::String>& vs, jobject list);


jobject vector_string_to_List(JNIEnv* env, std::vector<std::string>& vs);

std::vector<std::string> List_to_vector_string(JNIEnv* env, jobject list);

void Copy_vector_string_to_List(JNIEnv* env, std::vector<std::string>& vs, jobject list);

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"

void Copy_vector_MatShape_to_List(JNIEnv* env, std::vector<cv::MatShape>& vs, jobject list);

#endif // HAVE_OPENCV_DNN

#endif /* LISTCONVERTERS_HPP */
