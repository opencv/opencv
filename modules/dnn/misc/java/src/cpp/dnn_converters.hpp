// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Author: abratchik

#ifndef DNN_CONVERTERS_HPP
#define	DNN_CONVERTERS_HPP

#include <jni.h>
#include "opencv_java.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn/dnn.hpp"

#define LAYER(ENV) static_cast<jclass>(ENV->NewGlobalRef(ENV->FindClass("org/opencv/dnn/Layer")))
#define LAYER_CONSTRUCTOR(ENV, CLS) ENV->GetMethodID(CLS, "<init>", "(J)V")


using namespace cv::dnn;

void Mat_to_MatShape(cv::Mat& mat, MatShape& matshape);

void MatShape_to_Mat(MatShape& matshape, cv::Mat& mat);

std::vector<MatShape> List_to_vector_MatShape(JNIEnv* env, jobject list);

jobject vector_Ptr_Layer_to_List(JNIEnv* env, std::vector<cv::Ptr<cv::dnn::Layer> >& vs);

std::vector<cv::Ptr<cv::dnn::Layer> > List_to_vector_Ptr_Layer(JNIEnv* env, jobject list);

jobject vector_Target_to_List(JNIEnv* env, std::vector<cv::dnn::Target>& vs);

#endif	/* DNN_CONVERTERS_HPP */
