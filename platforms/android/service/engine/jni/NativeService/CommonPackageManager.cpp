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
        int platform_group = 0;

        if ((cpu_id & ARCH_X86) || (cpu_id & ARCH_X64))
            platform_group = 1;

        if (cpu_id & ARCH_MIPS)
            platform_group = 2;

        int opt_rating = -1;
        int opt_version = 0;

        const int hardware_rating = GetHardwareRating(platform, cpu_id, ArchRatings[platform_group]);
        LOGD("Current hardware platform rating %d for (%d,%d)", hardware_rating, platform, cpu_id);

        if (-1 == hardware_rating)
        {
            LOGE("Cannot calculate rating for current hardware platform!");
        }
        else
        {
            vector<PackageInfo>::iterator found = packages.end();
            for (vector<PackageInfo>::iterator it = packages.begin(); it != packages.end(); ++it)
            {
                int package_group = 0;

                if ((it->GetCpuID() & ARCH_X86) || (it->GetCpuID() & ARCH_X64))
                    package_group = 1;

                if (it->GetCpuID() & ARCH_MIPS)
                    package_group = 2;

                if (package_group != platform_group)
                    continue;

                const int package_rating = GetHardwareRating(it->GetPlatform(), it->GetCpuID(), ArchRatings[package_group]);

                LOGD("Package \"%s\" rating %d for (%d,%d)", it->GetFullName().c_str(), package_rating, it->GetPlatform(), it->GetCpuID());
                if ((package_rating >= 0) && (package_rating <= hardware_rating))
                {
                    if (((it->GetVersion() >= opt_version) && (package_rating >= opt_rating)) || (it->GetVersion() > opt_version))
                    {
                        opt_rating = package_rating;
                        opt_version = it->GetVersion();
                        found = it;
                    }
                }
            }

            if ((-1 != opt_rating) && (packages.end() != found))
            {
                result = found->GetInstalationPath();
            }
            else
            {
                LOGI("No compatible packages found!");
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
        LOGD("!!! Calculating rating for ARM\n");
        for (size_t i = 0; i < group.size(); i++)
        {
            LOGD("Checking (%d, %d) against (%d,%d)\n", group[i].first, group[i].second, platform, cpu_id);
            if (group[i] == std::pair<int, int>(platform, cpu_id))
            {
                LOGD("Rating found: %d\n", i);
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

    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv4 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv4 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv4 | FEATURES_HAS_VFPv3 | FEATURES_HAS_VFPv3d16 | FEATURES_HAS_NEON));

    result.push_back(std::pair<int, int>(PLATFORM_TEGRA3,  ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_TEGRA4i, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_VFPv4 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_TEGRA4,  ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_VFPv4 | FEATURES_HAS_NEON));
    result.push_back(std::pair<int, int>(PLATFORM_TEGRA5,  ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_VFPv4 | FEATURES_HAS_NEON));

    return result;
}

// Stub for Intel platforms rating initialization. Common package for all Intel based devices is used now
std::vector<std::pair<int, int> > CommonPackageManager::InitIntelRating()
{
    std::vector<std::pair<int, int> > result;

    return result;
}

// Stub for MIPS platforms rating initialization. Common package for all MIPS based devices is used now
std::vector<std::pair<int, int> > CommonPackageManager::InitMipsRating()
{
    std::vector<std::pair<int, int> > result;

    return result;
}

const std::vector<std::pair<int, int> > CommonPackageManager::ArchRatings[] = {
                                           CommonPackageManager::InitArmRating(),
                                           CommonPackageManager::InitIntelRating(),
                                           CommonPackageManager::InitMipsRating()
                                        };

CommonPackageManager::~CommonPackageManager()
{
}
