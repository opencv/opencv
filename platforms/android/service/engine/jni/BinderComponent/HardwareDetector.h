#ifndef __HARDWARE_DETECTOR_H__
#define __HARDWARE_DETECTOR_H__

#include <string>

#define ARCH_UNKNOWN 0L
#define ARCH_X86 16777216L
#define ARCH_X64 33554432L
#define ARCH_ARMv5 67108864L
#define ARCH_ARMv6 134217728L
#define ARCH_ARMv7 268435456L
#define ARCH_ARMv8 536870912L
#define ARCH_MIPS 1073741824L

#define FEATURES_HAS_VFPv3d16 1L
#define FEATURES_HAS_VFPv3 2L
#define FEATURES_HAS_VFPv4 4L
#define FEATURES_HAS_NEON 8L
#define FEATURES_HAS_NEON2 16L

#define FEATURES_HAS_SSE 1L
#define FEATURES_HAS_SSE2 2L
#define FEATURES_HAS_SSSE3 4L
#define FEATURES_HAS_GPU 65536L

// TODO: Do not forget to add Platrfom name to PackageInfo::PlatformNameMap
// in method PackageInfo::InitPlatformNameMap()
#define PLATFORM_UNKNOWN 0L
#define PLATFORM_TEGRA   1L
#define PLATFORM_TEGRA2  2L
#define PLATFORM_TEGRA3  3L
#define PLATFORM_TEGRA4i 4L
#define PLATFORM_TEGRA4  5L
#define PLATFORM_TEGRA5  6L

int DetectKnownPlatforms();
int GetProcessorCount();
std::string GetPlatformName();
int GetCpuID();

#endif
