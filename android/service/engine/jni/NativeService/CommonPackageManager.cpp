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

vector<int> CommonPackageManager::GetInstalledVersions()
{
    vector<int> result;
    vector<PackageInfo> installed_packages = GetInstalledPackages();

    result.resize(installed_packages.size());

    for (size_t i = 0; i < installed_packages.size(); i++)
    {
        int version = installed_packages[i].GetVersion();
        assert(version);
        result[i] = version;
    }

    return result;
}

bool CommonPackageManager::CheckVersionInstalled(int version, int platform, int cpu_id)
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
        vector<PackageInfo>::const_iterator it = find(packages.begin(), packages.end(), target_package);
        result = (it != packages.end());
    }
    LOGD("CommonPackageManager::CheckVersionInstalled() end");
    return result;
}

bool CommonPackageManager::InstallVersion(int version, int platform, int cpu_id)
{
    LOGD("CommonPackageManager::InstallVersion() begin");
    PackageInfo package(version, platform, cpu_id);
    return InstallPackage(package);
}

string CommonPackageManager::GetPackagePathByVersion(int version, int platform, int cpu_id)
{
    string result;
    PackageInfo target_package(version, platform, cpu_id);
    vector<PackageInfo> all_packages = GetInstalledPackages();
    vector<PackageInfo> packages;

    for (vector<PackageInfo>::iterator it = all_packages.begin(); it != all_packages.end(); ++it)
    {
        LOGD("Check version \"%d\" compatibility with \"%d\"\n", version, it->GetVersion());
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
        int OptRating = -1;
        int OptVersion = 0;
        std::vector<std::pair<int, int> >& group = CommonPackageManager::ArmRating;

        if ((cpu_id & ARCH_X86) || (cpu_id & ARCH_X64))
            group = CommonPackageManager::IntelRating;

        int HardwareRating = GetHardwareRating(platform, cpu_id, group);
        LOGD("Current hardware platform rating %d for (%d,%d)", HardwareRating, platform, cpu_id);

        if (-1 == HardwareRating)
        {
            LOGE("Cannot calculate rating for current hardware platform!");
        }
        else
        {
            vector<PackageInfo>::iterator found = packages.end();
            for (vector<PackageInfo>::iterator it = packages.begin(); it != packages.end(); ++it)
            {
                int PackageRating = GetHardwareRating(it->GetPlatform(), it->GetCpuID(), group);
                LOGD("Package \"%s\" rating %d for (%d,%d)", it->GetFullName().c_str(), PackageRating, it->GetPlatform(), it->GetCpuID());
                if ((PackageRating >= 0) && (PackageRating <= HardwareRating))
                {
                    if (((it->GetVersion() >= OptVersion) && (PackageRating >= OptRating)) || (it->GetVersion() > OptVersion))
                    {
                        OptRating = PackageRating;
                        OptVersion = it->GetVersion();
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

    return result;
}

bool CommonPackageManager::IsVersionCompatible(int target_version, int package_version)
{
    assert(target_version);
    assert(package_version);

    // major version is the same and minor package version is above or the same as target.
    return ( (package_version/10000 == target_version/10000) && (package_version%10000 >= target_version%10000) );
}

int CommonPackageManager::GetHardwareRating(int platform, int cpu_id, const std::vector<std::pair<int, int> >& group)
{
    int result = -1;

    if ((cpu_id & ARCH_X86) || (cpu_id & ARCH_X64) || (cpu_id & ARCH_MIPS))
        // Note: No raiting for x86, x64 and MIPS
        // only one package is used
        result = 0;
    else
    {
        // Calculate rating for Arm
        for (size_t i = 0; i < group.size(); i++)
        {
            if (group[i] == std::pair<int, int>(platform, cpu_id))
            {
                result = i;
                break;
            }
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
    result.push_back(std::pair<int, int>(PLATFORM_TEGRA2,  ARCH_ARMv7 | FEATURES_HAS_VFPv3d16));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3d16 | FEATURES_HAS_VFPv3));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3d16 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_VFPv3d16 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_TEGRA3,  ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON));

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
