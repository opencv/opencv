// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Author: abratchik

#define LOG_TAG "org.opencv.utils.Converters"
#include "common.h"


jobject vector_String_to_List(JNIEnv* env, std::vector<cv::String>& vs) {

    static jclass juArrayList   = ARRAYLIST(env);
    static jmethodID m_create   = CONSTRUCTOR(env, juArrayList);
    jmethodID m_add       = LIST_ADD(env, juArrayList);

    jobject result = env->NewObject(juArrayList, m_create, vs.size());
    for (std::vector<cv::String>::iterator it = vs.begin(); it != vs.end(); ++it) {
        jstring element = env->NewStringUTF((*it).c_str());
        env->CallBooleanMethod(result, m_add, element);
        env->DeleteLocalRef(element);
    }
    return result;
}

std::vector<cv::String> List_to_vector_String(JNIEnv* env, jobject list)
{
    static jclass juArrayList       = ARRAYLIST(env);
    jmethodID m_size       = LIST_SIZE(env,juArrayList);
    jmethodID m_get        = LIST_GET(env, juArrayList);

    jint len = env->CallIntMethod(list, m_size);
    std::vector<cv::String> result;
    result.reserve(len);
    for (jint i=0; i<len; i++)
    {
        jstring element = static_cast<jstring>(env->CallObjectMethod(list, m_get, i));
        const char* pchars = env->GetStringUTFChars(element, NULL);
        result.push_back(pchars);
        env->ReleaseStringUTFChars(element, pchars);
        env->DeleteLocalRef(element);
    }
    return result;
}

void Copy_vector_String_to_List(JNIEnv* env, std::vector<cv::String>& vs, jobject list)
{
    static jclass juArrayList       = ARRAYLIST(env);
    jmethodID m_clear     = LIST_CLEAR(env, juArrayList);
    jmethodID m_add       = LIST_ADD(env, juArrayList);

    env->CallVoidMethod(list, m_clear);
    for (std::vector<cv::String>::iterator it = vs.begin(); it != vs.end(); ++it)
    {
        jstring element = env->NewStringUTF((*it).c_str());
        env->CallBooleanMethod(list, m_add, element);
        env->DeleteLocalRef(element);
    }
}


jobject vector_string_to_List(JNIEnv* env, std::vector<std::string>& vs) {

    static jclass juArrayList   = ARRAYLIST(env);
    static jmethodID m_create   = CONSTRUCTOR(env, juArrayList);
    jmethodID m_add       = LIST_ADD(env, juArrayList);

    jobject result = env->NewObject(juArrayList, m_create, vs.size());
    for (std::vector<std::string>::iterator it = vs.begin(); it != vs.end(); ++it) {
        jstring element = env->NewStringUTF((*it).c_str());
        env->CallBooleanMethod(result, m_add, element);
        env->DeleteLocalRef(element);
    }
    return result;
}

std::vector<std::string> List_to_vector_string(JNIEnv* env, jobject list)
{
    static jclass juArrayList       = ARRAYLIST(env);
    jmethodID m_size       = LIST_SIZE(env,juArrayList);
    jmethodID m_get        = LIST_GET(env, juArrayList);

    jint len = env->CallIntMethod(list, m_size);
    std::vector<std::string> result;
    result.reserve(len);
    for (jint i=0; i<len; i++)
    {
        jstring element = static_cast<jstring>(env->CallObjectMethod(list, m_get, i));
        const char* pchars = env->GetStringUTFChars(element, NULL);
        result.push_back(pchars);
        env->ReleaseStringUTFChars(element, pchars);
        env->DeleteLocalRef(element);
    }
    return result;
}

void Copy_vector_string_to_List(JNIEnv* env, std::vector<std::string>& vs, jobject list)
{
    static jclass juArrayList       = ARRAYLIST(env);
    jmethodID m_clear     = LIST_CLEAR(env, juArrayList);
    jmethodID m_add       = LIST_ADD(env, juArrayList);

    env->CallVoidMethod(list, m_clear);
    for (std::vector<std::string>::iterator it = vs.begin(); it != vs.end(); ++it)
    {
        jstring element = env->NewStringUTF((*it).c_str());
        env->CallBooleanMethod(list, m_add, element);
        env->DeleteLocalRef(element);
    }
}

#ifdef HAVE_OPENCV_DNN
void Copy_vector_MatShape_to_List(JNIEnv* env, std::vector<cv::dnn::MatShape>& vs, jobject list)
{
    static jclass juArrayList       = ARRAYLIST(env);
    jmethodID m_clear     = LIST_CLEAR(env, juArrayList);
    jmethodID m_add       = LIST_ADD(env, juArrayList);

    env->CallVoidMethod(list, m_clear);
    for (std::vector<cv::dnn::MatShape>::iterator it = vs.begin(); it != vs.end(); ++it)
    {
        jintArray element = env->NewIntArray((int)it->size());
        env->SetIntArrayRegion(element, 0, (int)it->size(), &(*it)[0]);
        env->CallBooleanMethod(list, m_add, element);
        env->DeleteLocalRef(element);
    }
}
#endif // HAVE_OPENCV_DNN
