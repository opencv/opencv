#include "JavaBasedPackageManager.h"
#include <utils/Log.h>
#include <assert.h>

#undef LOG_TAG
#define LOG_TAG "JavaBasedPackageManager"

using namespace std;

JavaBasedPackageManager::JavaBasedPackageManager(JavaVM* JavaMashine, jobject MarketConnector):
JavaContext(JavaMashine),
JavaPackageManager(MarketConnector)
{
    assert(JavaContext);
    assert(JavaPackageManager);
}

bool JavaBasedPackageManager::InstallPackage(const PackageInfo& package)
{
    JNIEnv* jenv;
    bool self_attached;
    LOGD("JavaBasedPackageManager::InstallPackage() begin\n");

    self_attached = (JNI_EDETACHED == JavaContext->GetEnv((void**)&jenv, JNI_VERSION_1_6));
    if (self_attached)
    {
        JavaContext->AttachCurrentThread(&jenv, NULL);
    }

    LOGD("GetObjectClass call\n");
    jclass jclazz = jenv->GetObjectClass(JavaPackageManager);
    if (!jclazz)
    {
        LOGE("MarketConnector class was not found!");
        return false;
    }

    LOGD("GetMethodID call\n");
    jmethodID jmethod = jenv->GetMethodID(jclazz, "InstallAppFromMarket", "(Ljava/lang/String;)Z");
    if (!jmethod)
    {
        LOGE("MarketConnector::GetAppFormMarket method was not found!");
        jenv->DeleteLocalRef(jclazz);
        return false;
    }

    LOGD("Calling java package manager with package name %s\n", package.GetFullName().c_str());
    jobject jpkgname = jenv->NewStringUTF(package.GetFullName().c_str());
    bool result = jenv->CallNonvirtualBooleanMethod(JavaPackageManager, jclazz, jmethod, jpkgname);

    jenv->DeleteLocalRef(jpkgname);
    jenv->DeleteLocalRef(jclazz);

    if (self_attached)
    {
        JavaContext->DetachCurrentThread();
    }

    LOGD("JavaBasedPackageManager::InstallPackage() end\n");

    return result;
}

vector<PackageInfo> JavaBasedPackageManager::GetInstalledPackages()
{
    vector<PackageInfo> result;
    JNIEnv* jenv;
    bool self_attached;

    LOGD("JavaBasedPackageManager::GetInstalledPackages() begin");

    self_attached = (JNI_EDETACHED == JavaContext->GetEnv((void**)&jenv, JNI_VERSION_1_6));
    if (self_attached)
    {
        JavaContext->AttachCurrentThread(&jenv, NULL);
    }

    jclass jclazz = jenv->GetObjectClass(JavaPackageManager);
    if (!jclazz)
    {
        LOGE("MarketConnector class was not found!");
        return result;
    }

    jmethodID jmethod = jenv->GetMethodID(jclazz, "GetInstalledOpenCVPackages", "()[Landroid/content/pm/PackageInfo;");
    if (!jmethod)
    {
        LOGE("MarketConnector::GetInstalledOpenCVPackages method was not found!");
        jenv->DeleteLocalRef(jclazz);
        return result;
    }

    jobjectArray jpkgs = static_cast<jobjectArray>(jenv->CallNonvirtualObjectMethod(JavaPackageManager, jclazz, jmethod));
    jsize size = jenv->GetArrayLength(jpkgs);

    LOGD("Package info conversion");

    result.reserve(size);

    for (jsize i = 0; i < size; i++)
    {
        jobject jtmp = jenv->GetObjectArrayElement(jpkgs, i);
        PackageInfo tmp = ConvertPackageFromJava(jtmp, jenv);

        if (tmp.IsValid())
            result.push_back(tmp);

        jenv->DeleteLocalRef(jtmp);
    }

    jenv->DeleteLocalRef(jpkgs);
    jenv->DeleteLocalRef(jclazz);

    if (self_attached)
    {
        JavaContext->DetachCurrentThread();
    }

    LOGD("JavaBasedPackageManager::GetInstalledPackages() end");

    return result;
}

static jint GetAndroidVersion(JNIEnv* jenv)
{
    jclass jclazz = jenv->FindClass("android/os/Build$VERSION");
    jfieldID jfield = jenv->GetStaticFieldID(jclazz, "SDK_INT", "I");
    jint api_level = jenv->GetStaticIntField(jclazz, jfield);
    jenv->DeleteLocalRef(jclazz);

    return api_level;
}

// IMPORTANT: This method can be called only if thread is attached to Dalvik
PackageInfo JavaBasedPackageManager::ConvertPackageFromJava(jobject package, JNIEnv* jenv)
{
    jclass jclazz = jenv->GetObjectClass(package);

    jfieldID jfield = jenv->GetFieldID(jclazz, "packageName", "Ljava/lang/String;");
    jstring jnameobj = static_cast<jstring>(jenv->GetObjectField(package, jfield));
    const char* jnamestr = jenv->GetStringUTFChars(jnameobj, NULL);
    string name(jnamestr);
    jenv->DeleteLocalRef(jnameobj);

    jfield = jenv->GetFieldID(jclazz, "versionName", "Ljava/lang/String;");
    jstring jversionobj = static_cast<jstring>(jenv->GetObjectField(package, jfield));
    const char* jversionstr = jenv->GetStringUTFChars(jversionobj, NULL);
    string verison(jversionstr);
    jenv->DeleteLocalRef(jversionobj);

    jenv->DeleteLocalRef(jclazz);

    static const jint api_level = GetAndroidVersion(jenv);
    string path;
    if (api_level > 8)
    {
        jclazz = jenv->GetObjectClass(package);
        jfield = jenv->GetFieldID(jclazz, "applicationInfo", "Landroid/content/pm/ApplicationInfo;");
        jobject japp_info = jenv->GetObjectField(package, jfield);
        jenv->DeleteLocalRef(jclazz);

        jclazz = jenv->GetObjectClass(japp_info);
        jfield = jenv->GetFieldID(jclazz, "nativeLibraryDir", "Ljava/lang/String;");
        jstring jpathobj = static_cast<jstring>(jenv->GetObjectField(japp_info, jfield));
        const char* jpathstr = jenv->GetStringUTFChars(jpathobj, NULL);
        path = string(jpathstr);
        jenv->ReleaseStringUTFChars(jpathobj, jpathstr);

        jenv->DeleteLocalRef(japp_info);
        jenv->DeleteLocalRef(jpathobj);
        jenv->DeleteLocalRef(jclazz);
    }
    else
    {
        path = "/data/data/" + name + "/lib";
    }

    return PackageInfo(name, path, verison);
}

JavaBasedPackageManager::~JavaBasedPackageManager()
{
    JNIEnv* jenv;
    bool self_attached;

    LOGD("JavaBasedPackageManager::~JavaBasedPackageManager() begin");

    JavaContext->GetEnv((void**)&jenv, JNI_VERSION_1_6);
    self_attached = (JNI_EDETACHED == JavaContext->GetEnv((void**)&jenv, JNI_VERSION_1_6));
    if (self_attached)
    {
        JavaContext->AttachCurrentThread(&jenv, NULL);
    }

    jenv->DeleteGlobalRef(JavaPackageManager);

    if (self_attached)
    {
        JavaContext->DetachCurrentThread();
    }
    LOGD("JavaBasedPackageManager::~JavaBasedPackageManager() end");
}
