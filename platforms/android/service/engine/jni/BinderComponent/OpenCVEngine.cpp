#include "EngineCommon.h"
#include "OpenCVEngine.h"
#include "HardwareDetector.h"
#include "StringUtils.h"
#include <utils/Log.h>
#include <assert.h>
#include <string>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <dlfcn.h>

using namespace android;

const int OpenCVEngine::Platform = DetectKnownPlatforms();
const int OpenCVEngine::CpuID = GetCpuID();
const int OpenCVEngine::KnownVersions[] = {2040000, 2040100, 2040200, 2040300, 2040301, 2040302, 2040400, 2040500, 2040600, 2040700, 2040701, 2040800, 2040900};

bool OpenCVEngine::ValidateVersion(int version)
{
    for (size_t i = 0; i < sizeof(KnownVersions)/sizeof(int); i++)
        if (KnownVersions[i] == version)
            return true;

    return false;
}

int OpenCVEngine::NormalizeVersionString(std::string version)
{
    int result = 0;

    if (version.empty())
    {
        return result;
    }

    std::vector<std::string> parts = SplitStringVector(version, '.');

    // Use only 4 digits of the version, i.e. 1.2.3.4.
    // Other digits will be ignored.
    if (parts.size() > 4)
        parts.erase(parts.begin()+4, parts.end());

    int multiplyer = 1000000;
    for (std::vector<std::string>::const_iterator it = parts.begin(); it != parts.end(); ++it)
    {
        int digit = atoi(it->c_str());
        result += multiplyer*digit;
        multiplyer /= 100;
    }

    if (!ValidateVersion(result))
        result  = 0;

    return result;
}

OpenCVEngine::OpenCVEngine(IPackageManager* PkgManager):
    PackageManager(PkgManager)
{
    assert(PkgManager);
}

int32_t OpenCVEngine::GetVersion()
{
    return OPEN_CV_ENGINE_VERSION;
}

String16 OpenCVEngine::GetLibPathByVersion(android::String16 version)
{
    std::string std_version(String8(version).string());
    int norm_version;
    std::string path;

    LOGD("OpenCVEngine::GetLibPathByVersion(%s) impl", String8(version).string());

    norm_version = NormalizeVersionString(std_version);

    if (0 != norm_version)
    {
        path = PackageManager->GetPackagePathByVersion(norm_version, Platform, CpuID);
        if (path.empty())
        {
            LOGI("Package OpenCV of version \"%s\" (%d) is not installed. Try to install it :)", String8(version).string(), norm_version);
        }
        else
        {
            FixPermissions(path);
        }
    }
    else
    {
        LOGE("OpenCV version \"%s\" (%d) is not supported", String8(version).string(), norm_version);
    }

    return String16(path.c_str());
}

android::String16 OpenCVEngine::GetLibraryList(android::String16 version)
{
    std::string std_version = String8(version).string();
    int norm_version;
    String16 result;
    norm_version = NormalizeVersionString(std_version);

    if (0 != norm_version)
    {
        std::string tmp = PackageManager->GetPackagePathByVersion(norm_version, Platform, CpuID);
        if (!tmp.empty())
        {
            tmp += (std::string("/") + LIB_OPENCV_INFO_NAME);

            LOGD("Trying to load info library \"%s\"", tmp.c_str());

            void* handle;
            InfoFunctionType info_func;

            handle = dlopen(tmp.c_str(), RTLD_LAZY);
            if (handle)
            {
                const char* error;

                dlerror();
                info_func = (InfoFunctionType)dlsym(handle, "GetLibraryList");
                if ((error = dlerror()) == NULL)
                {
                    result = String16((*info_func)());
                    dlclose(handle);
                }
                else
                {
                    LOGE("Library loading error: \"%s\"", error);
                }
            }
            else
            {
                LOGI("Info library not found in package");
            }
        }
        else
        {
            LOGI("Package OpenCV of version \"%s\" (%d) is not installed. Try to install it :)", std_version.c_str(), norm_version);
        }
    }
    else
    {
        LOGE("OpenCV version \"%s\" is not supported", std_version.c_str());
    }

    return result;
}

bool OpenCVEngine::InstallVersion(android::String16 version)
{
    std::string std_version = String8(version).string();
    int norm_version;
    bool result = false;

    LOGD("OpenCVEngine::InstallVersion() begin");

    norm_version = NormalizeVersionString(std_version);

    if (0 != norm_version)
    {
        LOGD("PackageManager->InstallVersion call");
        result = PackageManager->InstallVersion(norm_version, Platform, CpuID);
    }
    else
    {
        LOGE("OpenCV version \"%s\" (%d) is not supported", std_version.c_str(), norm_version);
    }

    LOGD("OpenCVEngine::InstallVersion() end");

    return result;
}

bool OpenCVEngine::FixPermissions(const std::string& path)
{
    LOGD("Fixing permissions for folder: \"%s\"", path.c_str());
    chmod(path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);

    DIR* dir = opendir(path.c_str());
    if (!dir)
    {
        LOGD("Fixing permissions error");
        return false;
    }

    dirent* files = readdir(dir);
    while (files)
    {
        LOGD("Fix permissions for \"%s\"", files->d_name);
        chmod((path + std::string("/") + std::string(files->d_name)).c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
        files = readdir(dir);
    }

    closedir(dir);

    return true;
}
