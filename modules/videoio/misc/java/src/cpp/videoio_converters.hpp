#pragma once

class JavaStreamReader : public cv::IStreamReader
{
public:
    JavaStreamReader(JNIEnv* _env, jclass _jobject) : env(_env), jobject(_jobject) {}

    long long read(char* buffer, long long size) CV_OVERRIDE
    {
        jmethodID m_read = env->GetMethodID(env->GetObjectClass(jobject), "read", "([BJ)J");
        if (!m_read)
            return 0;
        jbyteArray jBuffer = env->NewByteArray(size);
        if (!jBuffer)
            return 0;
        jlong res = env->CallIntMethod(jobject, m_read, jBuffer, size);
        env->GetByteArrayRegion(jBuffer, 0, size, reinterpret_cast<jbyte*>(buffer));
        return res;
    }

    long long seek(long long offset, int way) CV_OVERRIDE
    {
        jmethodID m_seek = env->GetMethodID(env->GetObjectClass(jobject), "seek", "(JJ)J");
        if (!m_seek)
            return 0;
        jlong res = env->CallIntMethod(jobject, m_seek, offset, way);
        return res;
    }

private:
    JNIEnv* env;
    jclass jobject;
};
