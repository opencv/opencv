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

std::set<std::string> OpenCVEngine::InitKnownOpenCVersions()
{
    std::set<std::string> result;

    result.insert("240");
    result.insert("241");
    result.insert("242");

    return result;
}

const std::set<std::string> OpenCVEngine::KnownVersions = InitKnownOpenCVersions();

bool OpenCVEngine::ValidateVersionString(const std::string& version)
{
    return (KnownVersions.find(version) != KnownVersions.end());
}

std::string OpenCVEngine::NormalizeVersionString(std::string version)
{
    std::string result = "";
    std::string suffix = "";

    if (version.empty())
    {
	return result;
    }

    if (('a' == version[version.size()-1]) || ('b' == version[version.size()-1]))
    {
	suffix = version[version.size()-1];
	version.erase(version.size()-1);
    }

    std::vector<std::string> parts = SplitStringVector(version, '.');

    if (parts.size() >= 2)
    {
	if (parts.size() >= 3)
	{
	    result = parts[0] + parts[1] + parts[2] + suffix;
	    if (!ValidateVersionString(result))
		result = "";
	}
	else
	{
	    result = parts[0] + parts[1] + "0" + suffix;
	    if (!ValidateVersionString(result))
		result = "";
	}
    }

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
    std::string norm_version;
    std::string path;

    LOGD("OpenCVEngine::GetLibPathByVersion(%s) impl", String8(version).string());

    norm_version = NormalizeVersionString(std_version);

    if (!norm_version.empty())
    {
	path = PackageManager->GetPackagePathByVersion(norm_version, Platform, CpuID);
	if (path.empty())
	{
	    LOGI("Package OpenCV of version %s is not installed. Try to install it :)", norm_version.c_str());
	}
	else
	{
	    FixPermissions(path);
	}
    }
    else
    {
	LOGE("OpenCV version \"%s\" (%s) is not supported", String8(version).string(), norm_version.c_str());
    }

    return String16(path.c_str());
}

android::String16 OpenCVEngine::GetLibraryList(android::String16 version)
{
    std::string std_version = String8(version).string();
    std::string norm_version;
    String16 result;
    norm_version = NormalizeVersionString(std_version);

    if (!norm_version.empty())
    {
	std::string tmp = PackageManager->GetPackagePathByVersion(norm_version, Platform, CpuID);
	if (!tmp.empty())
	{
	    tmp += (std::string("/") + LIB_OPENCV_INFO_NAME);

	    LOGD("Trying to load info library \"%s\"", tmp.c_str());

	    void* handle;
	    char* (*info_func)();

	    handle = dlopen(tmp.c_str(), RTLD_LAZY);
	    if (handle)
	    {
		const char* error;

		dlerror();
		*(void **) (&info_func) = dlsym(handle, "GetLibraryList");
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
	    LOGI("Package OpenCV of version %s is not installed. Try to install it :)", norm_version.c_str());
	}
    }
    else
    {
	LOGE("OpenCV version \"%s\" is not supported", norm_version.c_str());
    }

    return result;
}

bool OpenCVEngine::InstallVersion(android::String16 version)
{
    std::string std_version = String8(version).string();
    std::string norm_version;
    bool result = false;

    norm_version = NormalizeVersionString(std_version);

    if (!norm_version.empty())
    {
	LOGD("OpenCVEngine::InstallVersion() begin");
	
	if (!PackageManager->CheckVersionInstalled(norm_version, Platform, CpuID))
	{
	    LOGD("PackageManager->InstallVersion call");
	    result = PackageManager->InstallVersion(norm_version, Platform, CpuID);
	}
	else
	{
	    LOGI("Package OpenCV of version %s is already installed. Skiped.", norm_version.c_str());
	    result = true;
	}
    }
    else
    {
	LOGE("OpenCV version \"%s\" is not supported", norm_version.c_str());
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
