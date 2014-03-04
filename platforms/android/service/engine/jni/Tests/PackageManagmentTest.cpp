#include "HardwareDetector.h"
#include "IPackageManager.h"
#include "CommonPackageManager.h"
#include "PackageManagerStub.h"
#include "IOpenCVEngine.h"
#include <utils/String16.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace std;

TEST(PackageManager, InstalledVersions)
{
    PackageManagerStub pm;
    PackageInfo info(2030000, PLATFORM_UNKNOWN, ARCH_ARMv7);
    pm.InstalledPackages.push_back(info);
    std::vector<int> versions = pm.GetInstalledVersions();
    EXPECT_EQ(1, versions.size());
    EXPECT_EQ(2030000, *versions.begin());
}

TEST(PackageManager, CheckVersionInstalled)
{
    PackageManagerStub pm;
    PackageInfo info(2030000, PLATFORM_UNKNOWN, ARCH_ARMv7);
    pm.InstalledPackages.push_back(info);
    EXPECT_TRUE(pm.CheckVersionInstalled(2030000, PLATFORM_UNKNOWN, ARCH_ARMv7));
}

TEST(PackageManager, InstallVersion)
{
    PackageManagerStub pm;
    PackageInfo info(2030000, PLATFORM_UNKNOWN, ARCH_ARMv5);
    pm.InstalledPackages.push_back(info);
    EXPECT_TRUE(pm.InstallVersion(2040000, PLATFORM_UNKNOWN, ARCH_ARMv5));
    EXPECT_EQ(2, pm.InstalledPackages.size());
    EXPECT_TRUE(pm.CheckVersionInstalled(2040000, PLATFORM_UNKNOWN, ARCH_ARMv5));
}

TEST(PackageManager, GetPackagePathForArmv5)
{
    PackageManagerStub pm;
    EXPECT_TRUE(pm.InstallVersion(2040300, PLATFORM_UNKNOWN, ARCH_ARMv5));
    string path = pm.GetPackagePathByVersion(2040300, PLATFORM_UNKNOWN, ARCH_ARMv5);
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv5/lib", path.c_str());
}

TEST(PackageManager, GetPackagePathForArmv7)
{
    PackageManagerStub pm;
    EXPECT_TRUE(pm.InstallVersion(2030000, PLATFORM_UNKNOWN, ARCH_ARMv7));
    string path = pm.GetPackagePathByVersion(2030000, PLATFORM_UNKNOWN, ARCH_ARMv7);
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_armv7a/lib", path.c_str());
}

TEST(PackageManager, GetPackagePathForArmv7Neon)
{
    PackageManagerStub pm;
    EXPECT_TRUE(pm.InstallVersion(2030000, PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_NEON));
    string path = pm.GetPackagePathByVersion(2030000, PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_NEON);
#ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_armv7a_neon/lib", path.c_str());
#else
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_armv7a/lib", path.c_str());
#endif
}

TEST(PackageManager, GetPackagePathForX86)
{
    PackageManagerStub pm;
    EXPECT_TRUE(pm.InstallVersion(2030000, PLATFORM_UNKNOWN, ARCH_X86));
    string path = pm.GetPackagePathByVersion(2030000, PLATFORM_UNKNOWN, ARCH_X86);
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_x86/lib", path.c_str());
}

TEST(PackageManager, GetPackagePathForX86SSE2)
{
    PackageManagerStub pm;
    EXPECT_TRUE(pm.InstallVersion(2030000, PLATFORM_UNKNOWN, ARCH_X86 | FEATURES_HAS_SSE2));
    string path = pm.GetPackagePathByVersion(2030000, PLATFORM_UNKNOWN, ARCH_X86 | FEATURES_HAS_SSE2);
#ifdef __SUPPORT_INTEL_FEATURES
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_x86_sse2/lib", path.c_str());
#else
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_x86/lib", path.c_str());
#endif
}

TEST(PackageManager, GetPackagePathForTegra3)
{
    PackageManagerStub pm;
    EXPECT_TRUE(pm.InstallVersion(2030000, PLATFORM_TEGRA3, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON));
    string path = pm.GetPackagePathByVersion(2030000, PLATFORM_TEGRA3, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON);
#ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_tegra3/lib", path.c_str());
#else
#ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_armv7a_neon/lib", path.c_str());
#else
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_armv7a/lib", path.c_str());
#endif
#endif
}

TEST(PackageManager, GetPackagePathForTegra4)
{
    PackageManagerStub pm;
    EXPECT_TRUE(pm.InstallVersion(2040400, PLATFORM_TEGRA4, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_VFPv4 | FEATURES_HAS_NEON));
    string path = pm.GetPackagePathByVersion(2040400, PLATFORM_TEGRA4, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_VFPv4 | FEATURES_HAS_NEON);
    #ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_tegra4/lib", path.c_str());
    #else
    #ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a_neon/lib", path.c_str());
    #else
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", path.c_str());
    #endif
    #endif
}

TEST(PackageManager, GetPackagePathForTegra5)
{
    PackageManagerStub pm;
    EXPECT_TRUE(pm.InstallVersion(2040400, PLATFORM_TEGRA5, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_VFPv4 | FEATURES_HAS_NEON));
    string path = pm.GetPackagePathByVersion(2040400, PLATFORM_TEGRA5, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_VFPv4 | FEATURES_HAS_NEON);
    #ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_tegra5/lib", path.c_str());
    #else
    #ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a_neon/lib", path.c_str());
    #else
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", path.c_str());
    #endif
    #endif
}

#ifdef __SUPPORT_MIPS
TEST(PackageManager, GetPackagePathForMips)
{
    PackageManagerStub pm;
    EXPECT_TRUE(pm.InstallVersion(2040000, PLATFORM_UNKNOWN, ARCH_MIPS));
    string path = pm.GetPackagePathByVersion(2040000, PLATFORM_UNKNOWN, ARCH_MIPS);
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_mips/lib", path.c_str());
}
#endif

// TODO: Enable tests if separate package will be exists
// TEST(PackageManager, GetPackagePathForTegra2)
// {
//     PackageManagerStub pm;
//     PackageInfo info("240", PLATFORM_TEGRA2, 0);
//     pm.InstalledPackages.push_back(info);
//     string path = pm.GetPackagePathByVersion("240", PLATFORM_TEGRA2, 0);
//     EXPECT_STREQ("/data/data/org.opencv.lib_v24_tegra2/lib", path.c_str());
// }
