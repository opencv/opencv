#include "HardwareDetector.h"
#include "IPackageManager.h"
#include "CommonPackageManager.h"
#include "PackageManagerStub.h"
#include "IOpenCVEngine.h"
#include <utils/String16.h>
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <vector>

using namespace std;

TEST(PackageManager, InstalledVersions)
{
    PackageManagerStub pm;
    PackageInfo info("230", PLATFORM_UNKNOWN, ARCH_ARMv7);
    pm.InstalledPackages.push_back(info);
    std::set<string> versions = pm.GetInstalledVersions();
    EXPECT_EQ(1, versions.size());
    EXPECT_EQ("230", *versions.begin());
}

TEST(PackageManager, CheckVersionInstalled)
{
    PackageManagerStub pm;
    PackageInfo info("230", PLATFORM_TEGRA3, 0);
    pm.InstalledPackages.push_back(info);
    EXPECT_TRUE(pm.CheckVersionInstalled("230", PLATFORM_TEGRA3, 0));
}

TEST(PackageManager, InstallVersion)
{
    PackageManagerStub pm;
    PackageInfo info("230", PLATFORM_TEGRA3, 0);
    pm.InstalledPackages.push_back(info);
    EXPECT_TRUE(pm.InstallVersion("240", PLATFORM_TEGRA3, 0));
    EXPECT_EQ(2, pm.InstalledPackages.size());
    EXPECT_TRUE(pm.CheckVersionInstalled("240", PLATFORM_TEGRA3, 0));
}

TEST(PackageManager, GetPackagePathForArmv7)
{
    PackageManagerStub pm;
    PackageInfo info("230", PLATFORM_UNKNOWN, ARCH_ARMv7);
    pm.InstalledPackages.push_back(info);
    string path = pm.GetPackagePathByVersion("230", PLATFORM_UNKNOWN, ARCH_ARMv7);
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_armv7/lib", path.c_str());
}

TEST(PackageManager, GetPackagePathForArmv7Neon)
{
    PackageManagerStub pm;
    PackageInfo info("230", PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_NEON);
    pm.InstalledPackages.push_back(info);
    
    string path = pm.GetPackagePathByVersion("230", PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_NEON);
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_armv7/lib", path.c_str());
    // TODO: Replace if seporate package will be exists
    //EXPECT_STREQ("/data/data/org.opencv.lib_v23_armv7_neon/lib", path.c_str());
}

TEST(PackageManager, GetPackagePathForX86)
{
    PackageManagerStub pm;
    PackageInfo info("230", PLATFORM_UNKNOWN, ARCH_X86);
    pm.InstalledPackages.push_back(info);
    string path = pm.GetPackagePathByVersion("230", PLATFORM_UNKNOWN, ARCH_X86);
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_x86/lib", path.c_str());
}

TEST(PackageManager, GetPackagePathForX86SSE2)
{
    PackageManagerStub pm;
    PackageInfo info("230", PLATFORM_UNKNOWN, ARCH_X86 | FEATURES_HAS_SSE2);
    pm.InstalledPackages.push_back(info);
    string path = pm.GetPackagePathByVersion("230", PLATFORM_UNKNOWN, ARCH_X86 | FEATURES_HAS_SSE2);
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_x86/lib", path.c_str());
    // TODO: Replace if seporate package will be exists
    //EXPECT_STREQ("/data/data/org.opencv.lib_v23_x86_sse2/lib", path.c_str());
}

// TODO: Enable tests if seporate package will be exists
// TEST(PackageManager, GetPackagePathForTegra2)
// {
//     PackageManagerStub pm;
//     PackageInfo info("240", PLATFORM_TEGRA2, 0);
//     pm.InstalledPackages.push_back(info);
//     string path = pm.GetPackagePathByVersion("240", PLATFORM_TEGRA2, 0);
//     EXPECT_STREQ("/data/data/org.opencv.lib_v24_tegra2/lib", path.c_str());
// }
// 
// TEST(PackageManager, GetPackagePathForTegra3)
// {
//     PackageManagerStub pm;
//     PackageInfo info("230", PLATFORM_TEGRA3, 0);
//     pm.InstalledPackages.push_back(info);
//     string path = pm.GetPackagePathByVersion("230", PLATFORM_TEGRA3, 0);
//     EXPECT_STREQ("/data/data/org.opencv.lib_v23_tegra3/lib", path.c_str());
// }
