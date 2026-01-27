// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Author: abratchik

#define LOG_TAG "org.opencv.utils.Converters"
#include "common.h"

// Helper function to safely create a Java string from raw bytes.
// NewStringUTF expects valid Modified UTF-8, but QR codes may contain
// non-UTF-8 data (e.g., GB2312/GBK Chinese encoding). This function
// uses a byte array and lets Java handle the charset conversion.
static jstring safeNewString(JNIEnv* env, const char* str, size_t len) {
    // First, try NewStringUTF for valid UTF-8 strings (most common case)
    // Check if the string is valid UTF-8
    bool isValidUtf8 = true;
    for (size_t i = 0; i < len && isValidUtf8; ) {
        unsigned char c = (unsigned char)str[i];
        if (c < 0x80) {
            // ASCII
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence
            if (i + 1 >= len || (str[i + 1] & 0xC0) != 0x80) {
                isValidUtf8 = false;
            }
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence
            if (i + 2 >= len || (str[i + 1] & 0xC0) != 0x80 || (str[i + 2] & 0xC0) != 0x80) {
                isValidUtf8 = false;
            }
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence
            if (i + 3 >= len || (str[i + 1] & 0xC0) != 0x80 || (str[i + 2] & 0xC0) != 0x80 || (str[i + 3] & 0xC0) != 0x80) {
                isValidUtf8 = false;
            }
            i += 4;
        } else {
            isValidUtf8 = false;
        }
    }

    if (isValidUtf8) {
        return env->NewStringUTF(str);
    }

    // For non-UTF-8 data, create a byte array and use String(byte[], charset)
    // Using ISO-8859-1 to preserve all byte values
    jbyteArray byteArray = env->NewByteArray(len);
    if (byteArray == NULL) {
        return env->NewStringUTF(""); // Out of memory
    }
    env->SetByteArrayRegion(byteArray, 0, len, (const jbyte*)str);

    static jclass stringClass = NULL;
    static jmethodID stringConstructor = NULL;

    if (stringClass == NULL) {
        jclass localClass = env->FindClass("java/lang/String");
        stringClass = (jclass)env->NewGlobalRef(localClass);
        env->DeleteLocalRef(localClass);
        stringConstructor = env->GetMethodID(stringClass, "<init>", "([BLjava/lang/String;)V");
    }

    // Use ISO-8859-1 charset to preserve all byte values
    jstring charset = env->NewStringUTF("ISO-8859-1");
    jstring result = (jstring)env->NewObject(stringClass, stringConstructor, byteArray, charset);

    env->DeleteLocalRef(byteArray);
    env->DeleteLocalRef(charset);

    return result;
}


jobject vector_String_to_List(JNIEnv* env, std::vector<cv::String>& vs) {

    static jclass juArrayList   = ARRAYLIST(env);
    static jmethodID m_create   = CONSTRUCTOR(env, juArrayList);
    jmethodID m_add       = LIST_ADD(env, juArrayList);

    jobject result = env->NewObject(juArrayList, m_create, vs.size());
    for (std::vector<cv::String>::iterator it = vs.begin(); it != vs.end(); ++it) {
        jstring element = safeNewString(env, (*it).c_str(), (*it).size());
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
        jstring element = safeNewString(env, (*it).c_str(), (*it).size());
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
        jstring element = safeNewString(env, (*it).c_str(), (*it).size());
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
        jstring element = safeNewString(env, (*it).c_str(), (*it).size());
        env->CallBooleanMethod(list, m_add, element);
        env->DeleteLocalRef(element);
    }
}
