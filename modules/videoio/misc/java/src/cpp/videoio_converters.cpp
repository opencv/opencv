#include "videoio_converters.hpp"

JavaStreamReader::JavaStreamReader(JNIEnv* env, jobject _obj)
{
    obj = env->NewGlobalRef(_obj);
    jclass cls = env->GetObjectClass(obj);
    m_read = env->GetMethodID(cls, "read", "([BJ)J");
    m_seek = env->GetMethodID(cls, "seek", "(JI)J");
    env->GetJavaVM(&vm);
}

JavaStreamReader::~JavaStreamReader()
{
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK)
        return;
    env->DeleteGlobalRef(obj);
}

long long JavaStreamReader::read(char* buffer, long long size)
{
    if (!m_read)
        return 0;
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK)
        return 0;
    jbyteArray jBuffer = env->NewByteArray(static_cast<jsize>(size));
    if (!jBuffer)
        return 0;
    jlong res = env->CallLongMethod(obj, m_read, jBuffer, size);
    env->GetByteArrayRegion(jBuffer, 0, static_cast<jsize>(size), reinterpret_cast<jbyte*>(buffer));
    env->DeleteLocalRef(jBuffer);
    return res;
}

long long JavaStreamReader::seek(long long offset, int way)
{
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK)
        return 0;
    if (!m_seek)
        return 0;
    return env->CallLongMethod(obj, m_seek, offset, way);
}

// Same as dnn::vector_Target_to_List
jobject vector_VideoCaptureAPIs_to_List(JNIEnv* env, std::vector<cv::VideoCaptureAPIs>& vs)
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
