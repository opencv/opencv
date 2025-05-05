#include "videoio_converters.hpp"

JavaStreamReader::JavaStreamReader(JNIEnv* _env, jclass _obj) : env(_env), obj(_obj) {}

long long JavaStreamReader::read(char* buffer, long long size)
{
    jmethodID m_read = env->GetMethodID(env->GetObjectClass(obj), "read", "([B)J");
    if (!m_read)
        return 0;
    jbyteArray jBuffer = env->NewByteArray(size);
    if (!jBuffer)
        return 0;
    jlong res = env->CallIntMethod(obj, m_read, jBuffer);
    env->GetByteArrayRegion(jBuffer, 0, size, reinterpret_cast<jbyte*>(buffer));
    env->DeleteLocalRef(jBuffer);
    return res;
}

long long JavaStreamReader::seek(long long offset, int way)
{
    jmethodID m_seek = env->GetMethodID(env->GetObjectClass(obj), "seek", "(JJ)J");
    if (!m_seek)
        return 0;
    jlong res = env->CallIntMethod(obj, m_seek, offset, way);
    return res;
}
