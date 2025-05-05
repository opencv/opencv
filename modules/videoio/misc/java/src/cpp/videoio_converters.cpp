#include "videoio_converters.hpp"

JavaStreamReader::JavaStreamReader(JNIEnv* _env, jclass _jobject) : env(_env), jobject(_jobject) {}

long long JavaStreamReader::read(char* buffer, long long size)
{
    jmethodID m_read = env->GetMethodID(env->GetObjectClass(jobject), "read", "([BJ)J");
    if (!m_read)
        return 0;
    jbyteArray jBuffer = env->NewByteArray(size);
    if (!jBuffer)
        return 0;
    jlong res = env->CallIntMethod(jobject, m_read, jBuffer, size);
    env->GetByteArrayRegion(jBuffer, 0, size, reinterpret_cast<jbyte*>(buffer));
    env->DeleteLocalRef(jBuffer);
    return res;
}

long long JavaStreamReader::seek(long long offset, int way)
{
    jmethodID m_seek = env->GetMethodID(env->GetObjectClass(jobject), "seek", "(JJ)J");
    if (!m_seek)
        return 0;
    jlong res = env->CallIntMethod(jobject, m_seek, offset, way);
    return res;
}
