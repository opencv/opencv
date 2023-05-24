// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Author: darkliang

#include "barcode_converters.hpp"

#define LOG_TAG "org.opencv.barcode"


void Copy_vector_BarcodeType_to_List(JNIEnv* env, std::vector<cv::barcode::BarcodeType>& vs, jobject list)
{
    static jclass juArrayList = ARRAYLIST(env);
    jmethodID m_add = LIST_ADD(env, juArrayList);
    jmethodID m_clear = LIST_CLEAR(env, juArrayList);
    env->CallVoidMethod(list, m_clear);

    jclass jInteger = env->FindClass("java/lang/Integer");
    jmethodID m_create_Integer = CONSTRUCTOR(env, jInteger);

    for (size_t i = 0; i < vs.size(); ++i)
    {
        jobject element = env->NewObject(jInteger, m_create_Integer, vs[i]);
        env->CallBooleanMethod(list, m_add, element);
        env->DeleteLocalRef(element);
    }
}
