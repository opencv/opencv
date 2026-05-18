// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OBJDETECT_CONVERTERS_HPP
#define	OBJDETECT_CONVERTERS_HPP

#include <jni.h>
#include "opencv_java.hpp"
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"

void Copy_vector_NativeByteArray_to_List(JNIEnv* env, std::vector<std::string>& vs, jobject list);

jobject vector_aruco2_FiducialMarker_to_List(JNIEnv* env, std::vector<cv::aruco2::FiducialMarker>& vs);
std::vector<cv::aruco2::FiducialMarker> List_to_vector_aruco2_FiducialMarker(JNIEnv* env, jobject list);

jobject vector_aruco2_Diamond_to_List(JNIEnv* env, std::vector<cv::aruco2::Diamond>& vs);
std::vector<cv::aruco2::Diamond> List_to_vector_aruco2_Diamond(JNIEnv* env, jobject list);

jobject vector_aruco2_FractalMarker_to_List(JNIEnv* env, std::vector<cv::aruco2::FractalMarker>& vs);
std::vector<cv::aruco2::FractalMarker> List_to_vector_aruco2_FractalMarker(JNIEnv* env, jobject list);

void Mat_to_vector_aruco2_DictionaryType(cv::Mat& mat, std::vector<cv::aruco2::DictionaryType>& v);
void vector_aruco2_DictionaryType_to_Mat(std::vector<cv::aruco2::DictionaryType>& v, cv::Mat& mat);

#endif	/* OBJDETECT_CONVERTERS_HPP */
