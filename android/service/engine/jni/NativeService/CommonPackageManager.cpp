#include "IOpenCVEngine.h"
#include "CommonPackageManager.h"
#include "HardwareDetector.h"
#include <utils/Log.h>
#include <algorithm>
#include <stdio.h>
#include <assert.h>

#undef LOG_TAG
#define LOG_TAG "CommonPackageManager"

using namespace std;

set<string> CommonPackageManager::GetInstalledVersions()
{
    set<string> result;
    vector<PackageInfo> installed_packages = GetInstalledPackages();

    for (vector<PackageInfo>::const_iterator it = installed_packages.begin(); it != installed_packages.end(); ++it)
    {
    string version = it->GetVersion();
    assert(!version.empty());
    result.insert(version);
    }

    return result;
}

bool CommonPackageManager::CheckVersionInstalled(const std::string& version, int platform, int cpu_id)
{
    bool result = false;
    LOGD("CommonPackageManager::CheckVersionInstalled() begin");
    PackageInfo target_package(version, platform, cpu_id);
    LOGD("GetInstalledPackages() call");
    vector<PackageInfo> packages = GetInstalledPackages();

    for (vector<PackageInfo>::const_iterator it = packages.begin(); it != packages.end(); ++it)
    {
    LOGD("Found package: \"%s\"", it->GetFullName().c_str());
    }

    if (!packages.empty())
    {
    result = (packages.end() != find(packages.begin(), packages.end(), target_package));
    }
    LOGD("CommonPackageManager::CheckVersionInstalled() end");
    return result;
}

bool CommonPackageManager::InstallVersion(const std::string& version, int platform, int cpu_id)
{
    LOGD("CommonPackageManager::InstallVersion() begin");
    PackageInfo package(version, platform, cpu_id);
    return InstallPackage(package);
}

string CommonPackageManager::GetPackagePathByVersion(const std::string& version, int platform, int cpu_id)
{
    string result;
    PackageInfo target_package(version, platform, cpu_id);
    vector<PackageInfo> all_packages = GetInstalledPackages();
    vector<PackageInfo> packages;

    for (vector<PackageInfo>::iterator it = all_packages.begin(); it != all_packages.end(); ++it)
    {
    LOGD("Check version \"%s\" compatibility with \"%s\"\n", version.c_str(), it->GetVersion().c_str());
    if (IsVersionCompatible(version, it->GetVersion()))
    {
        LOGD("Compatible");
        packages.push_back(*it);
    }
    else
    {
        LOGD("NOT Compatible");
    }
    }

    if (!packages.empty())
    {
    vector<PackageInfo>::iterator found = find(packages.begin(), packages.end(), target_package);
    if (packages.end() != found)
    {
        result = found->GetInstalationPath();
    }
    else
    {
        int OptRating = -1;
        std::vector<std::pair<int, int> >& group = CommonPackageManager::ArmRating;

        if ((cpu_id & ARCH_X86) || (cpu_id & ARCH_X64))
        group = CommonPackageManager::IntelRating;

        int HardwareRating = GetHardwareRating(platform, cpu_id, group);
        LOGD("Current hardware platform %d, %d", platform, cpu_id);

        if (-1 == HardwareRating)
        {
        LOGE("Cannot calculate rating for current hardware platform!");
        }
        else
        {
        for (vector<PackageInfo>::iterator it = packages.begin(); it != packages.end(); ++it)
        {
            int PackageRating = GetHardwareRating(it->GetPlatform(), it->GetCpuID(), group);
            if (PackageRating >= 0)
            {
            if ((PackageRating <= HardwareRating) && (PackageRating > OptRating))
            {
                OptRating = PackageRating;
                found = it;
            }
            }
        }

        if ((-1 != OptRating) && (packages.end() != found))
        {
            result = found->GetInstalationPath();
        }
        else
        {
            LOGI("Found package is incompatible with current hardware platform");
        }
        }
    }
    }

    return result;
}

bool CommonPackageManager::IsVersionCompatible(const std::string& target_version, const std::string& package_version)
{
    assert (target_version.size() == 3);
    assert (package_version.size() == 3);

    bool result = false;

    // major version is the same and minor package version is above or the same as target.
    if ((package_version[0] == target_version[0]) && (package_version[1] == target_version[1]) && (package_version[2] >= target_version[2]))
    {
    result = true;
    }

    return result;
}

int CommonPackageManager::GetHardwareRating(int platform, int cpu_id, const std::vector<std::pair<int, int> >& group)
{
    int result = -1;

    for (size_t i = 0; i < group.size(); i++)
    {
    if (group[i] == std::pair<int, int>(platform, cpu_id))
    {
        result = i;
        break;
    }
    }

    return result;
}

std::vector<std::pair<int, int> > CommonPackageManager::InitArmRating()
{
    std::vector<std::pair<int, int> > result;

    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv5));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv6));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv6 | FEATURES_HAS_VFPv3d16));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv6 | FEATURES_HAS_VFPv3));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv6 | FEATURES_HAS_VFPv3 | FEATURES_HAS_VFPv3d16));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3d16));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3d16 | FEATURES_HAS_VFPv3));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3d16 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_VFPv3d16 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_TEGRA2, ARCH_ARMv7 | FEATURES_HAS_VFPv3d16));
    result.push_back(std::pair<int, int>(PLATFORM_TEGRA3, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON));

    return result;
}

std::vector<std::pair<int, int> > CommonPackageManager::InitIntelRating()
{
    std::vector<std::pair<int, int> > result;

    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_X64));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_X86 | FEATURES_HAS_SSSE3));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_X86 | FEATURES_HAS_SSE2));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_X86 | FEATURES_HAS_SSE));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_X86));

    return result;
}

std::vector<std::pair<int, int> > CommonPackageManager::IntelRating = CommonPackageManager::InitIntelRating();
std::vector<std::pair<int, int> > CommonPackageManager::ArmRating = InitArmRating();

CommonPackageManager::~CommonPackageManager()
{
}
