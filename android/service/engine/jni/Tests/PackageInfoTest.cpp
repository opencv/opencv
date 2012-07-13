#include "HardwareDetector.h"
#include "IPackageManager.h"
#include "IOpenCVEngine.h"
#include "PackageInfo.h"
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <vector>

using namespace std;

TEST(PackageInfo, FullNameArmv7)
{
    PackageInfo info("230", PLATFORM_UNKNOWN, ARCH_ARMv7);
    string name = info.GetFullName();
    EXPECT_STREQ("org.opencv.lib_v23_armv7a", name.c_str());
}

TEST(PackageInfo, FullNameArmv7Neon)
{
    PackageInfo info("241", PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_NEON);
    string name = info.GetFullName();
#ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("org.opencv.lib_v24_armv7a_neon", name.c_str());
#else
    EXPECT_STREQ("org.opencv.lib_v24_armv7a", name.c_str());
#endif    
}

TEST(PackageInfo, FullNameArmv7VFPv3)
{
    PackageInfo info("233", PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3);
    string name = info.GetFullName();
    EXPECT_STREQ("org.opencv.lib_v23_armv7a", name.c_str());
}

TEST(PackageInfo, FullNameArmv7VFPv3Neon)
{
    PackageInfo info("230", PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON);
    string name = info.GetFullName();
#ifdef __SUPPORT_ARMEABI_V7A_FEATURES    
    EXPECT_STREQ("org.opencv.lib_v23_armv7a_neon", name.c_str());
#else    
    EXPECT_STREQ("org.opencv.lib_v23_armv7a", name.c_str());
#endif
}

TEST(PackageInfo, FullNameArmv5)
{
    PackageInfo info("230", PLATFORM_UNKNOWN, ARCH_ARMv5);
    string name = info.GetFullName();
    EXPECT_STREQ("org.opencv.lib_v23_armv5", name.c_str());
}

TEST(PackageInfo, FullNameArmv6)
{
    PackageInfo info("230", PLATFORM_UNKNOWN, ARCH_ARMv6);
    string name = info.GetFullName();
    EXPECT_STREQ("org.opencv.lib_v23_armv5", name.c_str());
}

TEST(PackageInfo, FullNameArmv6VFPv3)
{
    PackageInfo info("232", PLATFORM_UNKNOWN, ARCH_ARMv6 | FEATURES_HAS_VFPv3);
    string name = info.GetFullName();
#ifdef __SUPPORT_ARMEABI_FEATURES    
    EXPECT_STREQ("org.opencv.lib_v23_armv5_vfpv3", name.c_str());
#else
    EXPECT_STREQ("org.opencv.lib_v23_armv5", name.c_str());
#endif
}

TEST(PackageInfo, FullNameTegra3)
{
    PackageInfo info("230", PLATFORM_TEGRA3, ARCH_ARMv7 | FEATURES_HAS_NEON);
    string name = info.GetFullName();
    #ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("org.opencv.lib_v23_tegra3", name.c_str());
    #else
    #ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("org.opencv.lib_v23_armv7a_neon", name.c_str());
    #else
    EXPECT_STREQ("org.opencv.lib_v23_armv7a", name.c_str());
    #endif
    #endif
}

TEST(PackageInfo, FullNameX86SSE2)
{
    PackageInfo info("230", PLATFORM_UNKNOWN, ARCH_X86 | FEATURES_HAS_SSE2);
    string name = info.GetFullName();
#ifdef __SUPPORT_INTEL_FEATURES    
    EXPECT_STREQ("org.opencv.lib_v23_x86_sse2", name.c_str());
#else    
    EXPECT_STREQ("org.opencv.lib_v23_x86", name.c_str());
#endif
}

TEST(PackageInfo, Armv7NeonFromFullName)
{
    PackageInfo info("org.opencv.lib_v23_armv7a_neon", "/data/data/org.opencv.lib_v23_armv7_neon");
    EXPECT_EQ("230", info.GetVersion());
    EXPECT_EQ(ARCH_ARMv7 | FEATURES_HAS_NEON, info.GetCpuID());    
}

TEST(PackageInfo, Armv5FromFullName)
{
    PackageInfo info("org.opencv.lib_v23_armv5", "/data/data/org.opencv.lib_v23_armv5");
    EXPECT_EQ("230", info.GetVersion());
    EXPECT_EQ(ARCH_ARMv5, info.GetCpuID());    
}

TEST(PackageInfo, Armv5VFPv3FromFullName)
{
    PackageInfo info("org.opencv.lib_v23_armv5_vfpv3", "/data/data/org.opencv.lib_v23_armv5_vfpv3");
    EXPECT_EQ("230", info.GetVersion());
    EXPECT_EQ(ARCH_ARMv5 | FEATURES_HAS_VFPv3, info.GetCpuID());    
}

TEST(PackageInfo, X86SSE2FromFullName)
{
    PackageInfo info("org.opencv.lib_v24_x86_sse2", "/data/data/org.opencv.lib_v24_x86_sse2");
    EXPECT_EQ(PLATFORM_UNKNOWN, info.GetPlatform());
    EXPECT_EQ(ARCH_X86 | FEATURES_HAS_SSE2, info.GetCpuID());
    EXPECT_EQ("240", info.GetVersion());
}

TEST(PackageInfo, Tegra2FromFullName)
{
    PackageInfo info("org.opencv.lib_v23_tegra2", "/data/data/org.opencv.lib_v23_tegra2");
    EXPECT_EQ("230", info.GetVersion());
    EXPECT_EQ(PLATFORM_TEGRA2, info.GetPlatform());
}

TEST(PackageInfo, Tegra3FromFullName)
{
    PackageInfo info("org.opencv.lib_v24_tegra3", "/data/data/org.opencv.lib_v24_tegra3");
    EXPECT_EQ("240", info.GetVersion());
    EXPECT_EQ(PLATFORM_TEGRA3, info.GetPlatform());
}

TEST(PackageInfo, Comparator1)
{
    PackageInfo info1("240", PLATFORM_UNKNOWN, ARCH_X86);
    PackageInfo info2("org.opencv.lib_v24_x86", "/data/data/org.opencv.lib_v24_x86");
    EXPECT_STREQ(info1.GetFullName().c_str(), info2.GetFullName().c_str());
    EXPECT_EQ(info1, info2);
}

TEST(PackageInfo, Comparator2)
{
    PackageInfo info1("240", PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_NEON | FEATURES_HAS_VFPv3);
#ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    PackageInfo info2("org.opencv.lib_v24_armv7a_neon", "/data/data/org.opencv.lib_v24_armv7a_neon");
#else
    PackageInfo info2("org.opencv.lib_v24_armv7a", "/data/data/org.opencv.lib_v24_armv7a");
#endif
    EXPECT_STREQ(info1.GetFullName().c_str(), info2.GetFullName().c_str());
    EXPECT_EQ(info1, info2);
}

#ifdef __SUPPORT_TEGRA3 
TEST(PackageInfo, Comparator3)
{
    PackageInfo info1("230", PLATFORM_TEGRA3, 0);
    PackageInfo info2("org.opencv.lib_v23_tegra3", "/data/data/org.opencv.lib_v23_tegra3");
    EXPECT_STREQ(info1.GetFullName().c_str(), info2.GetFullName().c_str());
    EXPECT_EQ(info1, info2);
}
#endif