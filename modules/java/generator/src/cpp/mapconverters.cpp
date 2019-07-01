// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Author: janstarzy / Planet artificial intelligence GmbH

#include "common.h"

namespace
{
    template<class T = jobject>
    class LocalRef
    {
    public:
        LocalRef(JNIEnv *env, jobject obj) : m_env(env), m_obj(static_cast<T>(obj)) {}
        ~LocalRef()
        {
            if (m_env && m_obj) m_env->DeleteLocalRef(m_obj);
        }

        T get() const { return m_obj; }
        operator T() const { return get(); }

    private:
        JNIEnv *m_env;
        T m_obj;

        LocalRef(const LocalRef &);
        LocalRef &operator=(const LocalRef &);
    };

    class JNIMethods
    {
    public:
        jmethodID toString;
        jmethodID keySet;
        jmethodID iterator;
        jmethodID get;
        jmethodID put;
        jmethodID hasNext;
        jmethodID next;
        jmethodID intValue;

        static const JNIMethods &getInstance(JNIEnv *env)
        {
            static JNIMethods jniMethods(env);
            return jniMethods;
        }

    private:
        JNIMethods(JNIEnv *env)
        {
#define getMethodID(clazz, name, sig) name = env->GetMethodID(clazz, #name, sig)
            {
                LocalRef<jclass> Object(env, env->FindClass("java/lang/Object"));
                getMethodID(Object, toString, "()Ljava/lang/String;");
            }

            {
                LocalRef<jclass> Iterable(env, env->FindClass("java/lang/Iterable"));
                getMethodID(Iterable, iterator, "()Ljava/util/Iterator;");
            }

            {
                LocalRef<jclass> Iterator(env, env->FindClass("java/util/Iterator"));
                getMethodID(Iterator, hasNext, "()Z");
                getMethodID(Iterator, next, "()Ljava/lang/Object;");
            }

            {
                LocalRef<jclass> Number(env, env->FindClass("java/lang/Number"));
                getMethodID(Number, intValue, "()I");
            }

            {
                LocalRef<jclass> Map(env, env->FindClass("java/util/Map"));
                getMethodID(Map, get, "(Ljava/lang/Object;)Ljava/lang/Object;");
                getMethodID(Map, put, "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
                getMethodID(Map, keySet, "()Ljava/util/Set;");
            }
#undef getMethodID
        }
    };
}

template<class T>
static T Object_to(JNIEnv *env, jobject obj, const JNIMethods &jni);

template<>
int Object_to<int>(JNIEnv *env, jobject obj, const JNIMethods &jni)
{
    if (!obj) return 0;
    return env->CallIntMethod(obj, jni.intValue);
}

template<>
cv::String Object_to<cv::String>(JNIEnv *env, jobject obj, const JNIMethods &jni)
{
    if (!obj) return "";
    LocalRef<jstring> str(env, env->CallObjectMethod(obj, jni.toString));
    const char *chars = env->GetStringUTFChars(str, 0);
    jsize size = env->GetStringUTFLength(str);
    cv::String ret(chars, size);
    env->ReleaseStringUTFChars(str, chars);
    return ret;
}

template<class K, class V>
static std::map<K, V> Map_to_map(JNIEnv *env, jobject map)
{
    std::map<K, V> ret;
    if (!map) return ret;
    const JNIMethods &jni = JNIMethods::getInstance(env);
    LocalRef<jobject> key_set(env, env->CallObjectMethod(map, jni.keySet));
    LocalRef<jobject> iterator(env, env->CallObjectMethod(key_set, jni.iterator));
    while (!env->ExceptionCheck() && env->CallBooleanMethod(iterator, jni.hasNext)) {
        LocalRef<jobject> key(env, env->CallObjectMethod(iterator, jni.next));
        LocalRef<jobject> val(env, env->CallObjectMethod(map, jni.get, key.get()));
        ret.insert(std::pair<K, V>(Object_to<K>(env, key, jni), Object_to<V>(env, val, jni)));
    }
    return ret;
}

std::map<int, int> Map_to_map_int_and_int(JNIEnv *env, jobject map)
{
    return Map_to_map<int, int>(env, map);
}

std::map<int, cv::String> Map_to_map_int_and_String(JNIEnv *env, jobject map)
{
    return Map_to_map<int, cv::String>(env, map);
}

template<class T>
static jobject toObject(JNIEnv *env, const T &val);

template<>
jobject toObject(JNIEnv *env, const cv::String &val)
{
    return env->NewStringUTF(val.c_str());
}

template<class K, class V>
static void Copy_map_to_Map(JNIEnv *env, const std::map<K, V> &src, jobject dst)
{
    if (!dst) return;
    const JNIMethods &jni = JNIMethods::getInstance(env);
    for (typename std::map<K, V>::const_iterator it = src.begin(); it != src.end(); ++it) {
        LocalRef<jobject> key(env, toObject(env, it->first));
        LocalRef<jobject> val(env, toObject(env, it->second));
        env->CallObjectMethod(dst, jni.put, key.get(), val.get());
    }
}

void Copy_map_String_and_String_to_Map(JNIEnv *env, const std::map<std::string, std::string> &src, jobject dst)
{
    Copy_map_to_Map(env, src, dst);
}
