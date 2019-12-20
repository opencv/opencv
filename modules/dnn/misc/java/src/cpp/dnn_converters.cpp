// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Author: abratchik

#include "dnn_converters.hpp"

#define LOG_TAG "org.opencv.dnn"

void Mat_to_MatShape(cv::Mat& mat, MatShape& matshape)
{
    matshape.clear();
    CHECK_MAT(mat.type()==CV_32SC1 && mat.cols==1);
    matshape = (MatShape) mat;
}

void MatShape_to_Mat(MatShape& matshape, cv::Mat& mat)
{
    mat = cv::Mat(matshape, true);
}

std::vector<MatShape> List_to_vector_MatShape(JNIEnv* env, jobject list)
{
    static jclass juArrayList       = ARRAYLIST(env);
    jmethodID m_size       = LIST_SIZE(env, juArrayList);
    jmethodID m_get        = LIST_GET(env, juArrayList);

    static jclass jMatOfInt = MATOFINT(env);

    jint len = env->CallIntMethod(list, m_size);
    std::vector<MatShape> result;
    result.reserve(len);
    for (jint i=0; i<len; i++)
    {
        jobject element = static_cast<jobject>(env->CallObjectMethod(list, m_get, i));
        cv::Mat& mat = *((cv::Mat*) GETNATIVEOBJ(env, jMatOfInt, element) );
        MatShape matshape = (MatShape) mat;
        result.push_back(matshape);
        env->DeleteLocalRef(element);
    }
    return result;
}

jobject vector_Ptr_Layer_to_List(JNIEnv* env, std::vector<cv::Ptr<cv::dnn::Layer> >& vs)
{
    static jclass juArrayList   = ARRAYLIST(env);
    static jmethodID m_create   = CONSTRUCTOR(env, juArrayList);
    jmethodID m_add       = LIST_ADD(env, juArrayList);

    static jclass jLayerClass = LAYER(env);
    static jmethodID m_create_layer = LAYER_CONSTRUCTOR(env, jLayerClass);

    jobject result = env->NewObject(juArrayList, m_create, vs.size());
    for (std::vector< cv::Ptr<cv::dnn::Layer> >::iterator it = vs.begin(); it != vs.end(); ++it) {
        jobject element = env->NewObject(jLayerClass, m_create_layer, (*it).get());
        env->CallBooleanMethod(result, m_add, element);
        env->DeleteLocalRef(element);
    }
    return result;
}

std::vector<cv::Ptr<cv::dnn::Layer> > List_to_vector_Ptr_Layer(JNIEnv* env, jobject list)
{
    static jclass juArrayList       = ARRAYLIST(env);
    jmethodID m_size       = LIST_SIZE(env, juArrayList);
    jmethodID m_get        = LIST_GET(env, juArrayList);

    static jclass jLayerClass = LAYER(env);

    jint len = env->CallIntMethod(list, m_size);
    std::vector< cv::Ptr<cv::dnn::Layer> > result;
    result.reserve(len);
    for (jint i=0; i<len; i++)
    {
        jobject element = static_cast<jobject>(env->CallObjectMethod(list, m_get, i));
        cv::Ptr<cv::dnn::Layer>* layer_ptr = (cv::Ptr<cv::dnn::Layer>*) GETNATIVEOBJ(env, jLayerClass, element) ;
        cv::Ptr<cv::dnn::Layer> layer = *(layer_ptr);
        result.push_back(layer);
        env->DeleteLocalRef(element);
    }
    return result;
}
