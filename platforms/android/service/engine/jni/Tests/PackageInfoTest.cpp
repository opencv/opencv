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
    PackageInfo info(2030000, PLATFORM_UNKNOWN, ARCH_ARMv7);
    string name = info.GetFullName();
    EXPECT_STREQ("org.opencv.lib_v23_armv7a", name.c_str());
}

TEST(PackageInfo, FullNameArmv7Neon)
{
    PackageInfo info(2040100, PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_NEON);
    string name = info.GetFullName();
#ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("org.opencv.lib_v24_armv7a_neon", name.c_str());
#else
    EXPECT_STREQ("org.opencv.lib_v24_armv7a", name.c_str());
#endif
}

TEST(PackageInfo, FullNameArmv7VFPv3)
{
    PackageInfo info(2030300, PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3);
    string name = info.GetFullName();
    EXPECT_STREQ("org.opencv.lib_v23_armv7a", name.c_str());
}

TEST(PackageInfo, FullNameArmv7VFPv4)
{
    PackageInfo info(2030300, PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv4);
    string name = info.GetFullName();
    EXPECT_STREQ("org.opencv.lib_v23_armv7a", name.c_str());
}

TEST(PackageInfo, FullNameArmv7VFPv3Neon)
{
    PackageInfo info(2030000, PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON);
    string name = info.GetFullName();
#ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("org.opencv.lib_v23_armv7a_neon", name.c_str());
#else
    EXPECT_STREQ("org.opencv.lib_v23_armv7a", name.c_str());
#endif
}

TEST(PackageInfo, FullNameAarch64)
{
    PackageInfo info(2041000, PLATFORM_UNKNOWN, ARCH_AARCH64);
    string name = info.GetFullName();
    EXPECT_STREQ("org.opencv.lib_v24_aarch64", name.c_str());
}

TEST(PackageInfo, FullNameArmv5)
{
    PackageInfo info(2030000, PLATFORM_UNKNOWN, ARCH_ARMv5);
    string name = info.GetFullName();
    EXPECT_STREQ("org.opencv.lib_v23_armv5", name.c_str());
}

TEST(PackageInfo, FullNameArmv6)
{
    PackageInfo info(2030000, PLATFORM_UNKNOWN, ARCH_ARMv6);
    string name = info.GetFullName();
    EXPECT_STREQ("org.opencv.lib_v23_armv5", name.c_str());
}

TEST(PackageInfo, FullNameArmv6VFPv3)
{
    PackageInfo info(2030200, PLATFORM_UNKNOWN, ARCH_ARMv6 | FEATURES_HAS_VFPv3);
    string name = info.GetFullName();
#ifdef __SUPPORT_ARMEABI_FEATURES
    EXPECT_STREQ("org.opencv.lib_v23_armv5_vfpv3", name.c_str());
#else
    EXPECT_STREQ("org.opencv.lib_v23_armv5", name.c_str());
#endif
}

TEST(PackageInfo, FullNameTegra3)
{
    PackageInfo info(2030000, PLATFORM_TEGRA3, ARCH_ARMv7 | FEATURES_HAS_NEON);
    string name = info.GetFullName();
#ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("org.opencv.lib_v23_tegra3", name.c_str());
#else
# ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("org.opencv.lib_v23_armv7a_neon", name.c_str());
# else
    EXPECT_STREQ("org.opencv.lib_v23_armv7a", name.c_str());
# endif
#endif
}

TEST(PackageInfo, FullNameTegra4)
{
    PackageInfo info(2040400, PLATFORM_TEGRA4, ARCH_ARMv7 | FEATURES_HAS_NEON);
    string name = info.GetFullName();
#ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("org.opencv.lib_v24_tegra4", name.c_str());
#else
# ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("org.opencv.lib_v24_armv7a_neon", name.c_str());
# else
    EXPECT_STREQ("org.opencv.lib_v24_armv7a", name.c_str());
# endif
#endif
}

TEST(PackageInfo, FullNameTegra4i)
{
    PackageInfo info(2040700, PLATFORM_TEGRA4i, ARCH_ARMv7 | FEATURES_HAS_NEON);
    string name = info.GetFullName();
#ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("org.opencv.lib_v24_tegra4", name.c_str());
#else
# ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("org.opencv.lib_v24_armv7a_neon", name.c_str());
# else
    EXPECT_STREQ("org.opencv.lib_v24_armv7a", name.c_str());
# endif
#endif
}

TEST(PackageInfo, FullNameTegra5)
{
    PackageInfo info(2040700, PLATFORM_TEGRA5, ARCH_ARMv7 | FEATURES_HAS_NEON);
    string name = info.GetFullName();
#ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("org.opencv.lib_v24_tegra5", name.c_str());
#else
# ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("org.opencv.lib_v24_armv7a_neon", name.c_str());
# else
    EXPECT_STREQ("org.opencv.lib_v24_armv7a", name.c_str());
# endif
#endif
}

TEST(PackageInfo, FullNameX86SSE2)
{
    PackageInfo info(2030000, PLATFORM_UNKNOWN, ARCH_X86 | FEATURES_HAS_SSE2);
    string name = info.GetFullName();
#ifdef __SUPPORT_INTEL_FEATURES
    EXPECT_STREQ("org.opencv.lib_v23_x86_sse2", name.c_str());
#else
    EXPECT_STREQ("org.opencv.lib_v23_x86", name.c_str());
#endif
}

#ifdef __SUPPORT_MIPS
TEST(PackageInfo, FullNameMips)
{
    PackageInfo info(2040300, PLATFORM_UNKNOWN, ARCH_MIPS);
    string name = info.GetFullName();
    EXPECT_STREQ("org.opencv.lib_v24_mips", name.c_str());
}
#endif

TEST(PackageInfo, Armv7NeonFromFullName)
{
    PackageInfo info("org.opencv.lib_v23_armv7a_neon", "/data/data/org.opencv.lib_v23_armv7_neon");
    EXPECT_EQ(2030000, info.GetVersion());
    EXPECT_EQ(ARCH_ARMv7 | FEATURES_HAS_NEON, info.GetCpuID());
}

TEST(PackageInfo, Armv5FromFullName)
{
    PackageInfo info("org.opencv.lib_v23_armv5", "/data/data/org.opencv.lib_v23_armv5");
    EXPECT_EQ(2030000, info.GetVersion());
    EXPECT_EQ(ARCH_ARMv5, info.GetCpuID());
}

TEST(PackageInfo, Armv5VFPv3FromFullName)
{
    PackageInfo info("org.opencv.lib_v23_armv5_vfpv3", "/data/data/org.opencv.lib_v23_armv5_vfpv3");
    EXPECT_EQ(2030000, info.GetVersion());
    EXPECT_EQ(ARCH_ARMv5 | FEATURES_HAS_VFPv3, info.GetCpuID());
}

TEST(PackageInfo, X86SSE2FromFullName)
{
    PackageInfo info("org.opencv.lib_v24_x86_sse2", "/data/data/org.opencv.lib_v24_x86_sse2");
    EXPECT_EQ(PLATFORM_UNKNOWN, info.GetPlatform());
    EXPECT_EQ(ARCH_X86 | FEATURES_HAS_SSE2, info.GetCpuID());
    EXPECT_EQ(2040000, info.GetVersion());
}

TEST(PackageInfo, Tegra2FromFullName)
{
    PackageInfo info("org.opencv.lib_v23_tegra2", "/data/data/org.opencv.lib_v23_tegra2");
    EXPECT_EQ(2030000, info.GetVersion());
    EXPECT_EQ(PLATFORM_TEGRA2, info.GetPlatform());
}

TEST(PackageInfo, Tegra3FromFullName)
{
    PackageInfo info("org.opencv.lib_v24_tegra3", "/data/data/org.opencv.lib_v24_tegra3");
    EXPECT_EQ(2040000, info.GetVersion());
    EXPECT_EQ(PLATFORM_TEGRA3, info.GetPlatform());
}

TEST(PackageInfo, Tegra4FromFullName)
{
    PackageInfo info("org.opencv.lib_v24_tegra4", "/data/data/org.opencv.lib_v24_tegra4");
    EXPECT_EQ(2040000, info.GetVersion());
    EXPECT_EQ(PLATFORM_TEGRA4, info.GetPlatform());
}

#ifdef __SUPPORT_MIPS
TEST(PackageInfo, MipsFromFullName)
{
    PackageInfo info("org.opencv.lib_v24_mips", "/data/data/org.opencv.lib_v24_mips");
    EXPECT_EQ(2040000, info.GetVersion());
    EXPECT_EQ(ARCH_MIPS, info.GetCpuID());
}
#endif

TEST(PackageInfo, Check2DigitRevision)
{
    PackageInfo info("org.opencv.lib_v23_armv7a_neon", "/data/data/org.opencv.lib_v23_armv7_neon", "4.1");
    EXPECT_EQ(2030400, info.GetVersion());
    EXPECT_EQ(ARCH_ARMv7 | FEATURES_HAS_NEON, info.GetCpuID());
}

TEST(PackageInfo, Check3DigitRevision)
{
    PackageInfo info("org.opencv.lib_v23_armv7a_neon", "/data/data/org.opencv.lib_v23_armv7_neon", "4.1.5");
    EXPECT_EQ(2030401, info.GetVersion());
    EXPECT_EQ(ARCH_ARMv7 | FEATURES_HAS_NEON, info.GetCpuID());
}

TEST(PackageInfo, Comparator1)
{
    PackageInfo info1(2040000, PLATFORM_UNKNOWN, ARCH_X86);
    PackageInfo info2("org.opencv.lib_v24_x86", "/data/data/org.opencv.lib_v24_x86");
    EXPECT_STREQ(info1.GetFullName().c_str(), info2.GetFullName().c_str());
    EXPECT_EQ(info1, info2);
}

TEST(PackageInfo, Comparator2)
{
    PackageInfo info1(2040000, PLATFORM_UNKNOWN, ARCH_ARMv7 | FEATURES_HAS_NEON | FEATURES_HAS_VFPv3);
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
    PackageInfo info1(2030000, PLATFORM_TEGRA3, 0);
    PackageInfo info2("org.opencv.lib_v23_tegra3", "/data/data/org.opencv.lib_v23_tegra3");
    EXPECT_STREQ(info1.GetFullName().c_str(), info2.GetFullName().c_str());
    EXPECT_EQ(info1, info2);
}
#endif
