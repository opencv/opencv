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

template<typename T>
jobject vector_to_List(JNIEnv* env, std::vector<T>& vs, const char* classPath) {
    static jclass juArrayList = ARRAYLIST(env);
    static jmethodID m_ctor = env->GetMethodID(juArrayList, "<init>", "()V");
    static jmethodID m_add = LIST_ADD(env, juArrayList);

    jclass jItemClass = env->FindClass(classPath);
    std::string signature = "(J)L" + std::string(classPath) + ";";
    jmethodID m_fromPtr = env->GetStaticMethodID(jItemClass, "__fromPtr__", signature.c_str());

    jobject list = env->NewObject(juArrayList, m_ctor);
    for (auto it = vs.begin(); it != vs.end(); ++it) {
        T* m = new T(*it);
        jobject jm = env->CallStaticObjectMethod(jItemClass, m_fromPtr, (jlong)m);
        env->CallBooleanMethod(list, m_add, jm);
        env->DeleteLocalRef(jm);
    }
    return list;
}

template<typename T>
std::vector<T> List_to_vector(JNIEnv* env, jobject list, const char* classPath) {
    static jclass juList = env->FindClass("java/util/List");
    static jmethodID m_size = env->GetMethodID(juList, "size", "()I");
    static jmethodID m_get = env->GetMethodID(juList, "get", "(I)Ljava/lang/Object;");

    jclass jItemClass = env->FindClass(classPath);
    jmethodID m_getNativeObjAddr = env->GetMethodID(jItemClass, "getNativeObjAddr", "()J");

    int size = env->CallIntMethod(list, m_size);
    std::vector<T> vs;
    vs.reserve(size);
    for (int i = 0; i < size; ++i) {
        jobject jm = env->CallObjectMethod(list, m_get, i);
        jlong addr = env->CallLongMethod(jm, m_getNativeObjAddr);
        vs.push_back(*((T*)addr));
        env->DeleteLocalRef(jm);
    }
    return vs;
}

jobject vector_aruco2_Marker_to_List(JNIEnv* env, std::vector<cv::aruco2::Marker>& vs) {
    return vector_to_List(env, vs, "org/opencv/objdetect/Marker");
}

std::vector<cv::aruco2::Marker> List_to_vector_aruco2_Marker(JNIEnv* env, jobject list) {
    return List_to_vector<cv::aruco2::Marker>(env, list, "org/opencv/objdetect/Marker");
}

jobject vector_aruco2_Diamond_to_List(JNIEnv* env, std::vector<cv::aruco2::Diamond>& vs) {
    return vector_to_List(env, vs, "org/opencv/objdetect/Diamond");
}

std::vector<cv::aruco2::Diamond> List_to_vector_aruco2_Diamond(JNIEnv* env, jobject list) {
    return List_to_vector<cv::aruco2::Diamond>(env, list, "org/opencv/objdetect/Diamond");
}

jobject vector_aruco2_FractalMarker_to_List(JNIEnv* env, std::vector<cv::aruco2::FractalMarker>& vs) {
    return vector_to_List(env, vs, "org/opencv/objdetect/FractalMarker");
}

std::vector<cv::aruco2::FractalMarker> List_to_vector_aruco2_FractalMarker(JNIEnv* env, jobject list) {
    return List_to_vector<cv::aruco2::FractalMarker>(env, list, "org/opencv/objdetect/FractalMarker");
}

void Mat_to_vector_aruco2_DictionaryType(cv::Mat& mat, std::vector<cv::aruco2::DictionaryType>& v) {
    std::vector<int> v_int;
    cv::Mat(mat.reshape(1, 1)).copyTo(v_int);
    v.clear();
    v.reserve(v_int.size());
    for(int i : v_int) v.push_back((cv::aruco2::DictionaryType)i);
}

void vector_aruco2_DictionaryType_to_Mat(std::vector<cv::aruco2::DictionaryType>& v, cv::Mat& mat) {
    std::vector<int> v_int;
    v_int.reserve(v.size());
    for(auto i : v) v_int.push_back((int)i);
    mat = cv::Mat(v_int, true).clone();
}
