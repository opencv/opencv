// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Author: abratchik

#include "dnn_converters.hpp"
#include "converters.h"

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

void Mat_to_vector_MatShape(cv::Mat& mat, std::vector<MatShape>& v_matshape)
{
    v_matshape.clear();
    if(mat.type() == CV_32SC2 && mat.cols == 1)
    {
        v_matshape.reserve(mat.rows);
        for(int i=0; i<mat.rows; i++)
        {
            cv::Vec<int, 2> a = mat.at< cv::Vec<int, 2> >(i, 0);
            long long addr = (((long long)a[0])<<32) | (a[1]&0xffffffff);
            cv::Mat& m = *( (cv::Mat*) addr );
            MatShape matshape = (MatShape) m;
            v_matshape.push_back(matshape);
        }
    } else {
        LOGD("Mat_to_vector_MatShape() FAILED: mat.type() == CV_32SC2 && mat.cols == 1");
    }
}

void vector_MatShape_to_Mat(std::vector<MatShape>& v_matshape, cv::Mat& mat)
{
    int count = (int)v_matshape.size();
    mat.create(count, 1, CV_32SC2);
    for(int i=0; i<count; i++)
    {
        cv::Mat temp_mat = cv::Mat(v_matshape[i], true);
        long long addr = (long long) new cv::Mat(temp_mat);
        mat.at< cv::Vec<int, 2> >(i, 0) = cv::Vec<int, 2>(addr>>32, addr&0xffffffff);
    }
}

void Mat_to_vector_vector_MatShape(cv::Mat& mat, std::vector< std::vector< MatShape > >& vv_matshape)
{
    std::vector<cv::Mat> vm;
    vm.reserve( mat.rows );
    Mat_to_vector_Mat(mat, vm);
    for(size_t i=0; i<vm.size(); i++)
    {
        std::vector<MatShape> vmatshape;
        Mat_to_vector_MatShape(vm[i], vmatshape);
        vv_matshape.push_back(vmatshape);
    }
}

void vector_vector_MatShape_to_Mat(std::vector< std::vector< MatShape > >& vv_matshape, cv::Mat& mat)
{
    std::vector<cv::Mat> vm;
    vm.reserve( vv_matshape.size() );
    for(size_t i=0; i<vv_matshape.size(); i++)
    {
        cv::Mat m;
        vector_MatShape_to_Mat(vv_matshape[i], m);
        vm.push_back(m);
    }
    vector_Mat_to_Mat(vm, mat);
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

jobject vector_Target_to_List(JNIEnv* env, std::vector<cv::dnn::Target>& vs)
{
    static jclass juArrayList   = ARRAYLIST(env);
    static jmethodID m_create   = CONSTRUCTOR(env, juArrayList);
    jmethodID m_add       = LIST_ADD(env, juArrayList);

    static jclass jInteger = env->FindClass("java/lang/Integer");
    static jmethodID m_create_Integer = env->GetMethodID(jInteger, "<init>", "(I)V");

    jobject result = env->NewObject(juArrayList, m_create, vs.size());
    for (size_t i = 0; i < vs.size(); ++i)
    {
        jobject element = env->NewObject(jInteger, m_create_Integer, vs[i]);
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
