#include "objdetect_converters.hpp"

#define LOG_TAG "org.opencv.objdetect"

void Copy_vector_NativeByteArray_to_List(JNIEnv* env, std::vector<std::string>& vs, jobject list)
{
    static jclass juArrayList       = ARRAYLIST(env);
    jmethodID m_clear     = LIST_CLEAR(env, juArrayList);
    jmethodID m_add       = LIST_ADD(env, juArrayList);

    env->CallVoidMethod(list, m_clear);
    for (std::vector<std::string>::iterator it = vs.begin(); it != vs.end(); ++it)
    {
        jsize sz = static_cast<jsize>((*it).size());
        jbyteArray element = env->NewByteArray(sz);
        env->SetByteArrayRegion(element, 0, sz, reinterpret_cast<const jbyte*>((*it).c_str()));
        env->CallBooleanMethod(list, m_add, element);
        env->DeleteLocalRef(element);
    }
}
