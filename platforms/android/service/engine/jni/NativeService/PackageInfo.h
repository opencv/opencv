#ifndef __PACKAGE_INFO_H__
#define __PACKAGE_INFO_H__

#include <map>
#include <string>

#define ARCH_X86_NAME "x86"
#define ARCH_X64_NAME "x64"
#define ARCH_MIPS_NAME "mips"
#define ARCH_ARMv5_NAME "armv5"
#define ARCH_ARMv6_NAME "armv6"
#define ARCH_ARMv7_NAME "armv7a"
#define ARCH_AARCH64_NAME "aarch64"

#define FEATURES_HAS_VFPv3d16_NAME "vfpv3d16"
#define FEATURES_HAS_VFPv3_NAME "vfpv3"
#define FEATURES_HAS_NEON_NAME "neon"
#define FEATURES_HAS_NEON2_NAME "neon2"
#define FEATURES_HAS_SSE_NAME "sse"
#define FEATURES_HAS_SSE2_NAME "sse2"
#define FEATURES_HAS_SSSE3_NAME "ssse3"
#define FEATURES_HAS_GPU_NAME "gpu"

// TODO: Do not forget to update PackageInfo::InitPlatformNameMap() after constant changes
#define PLATFORM_TEGRA_NAME "tegra"
#define PLATFORM_TEGRA2_NAME "tegra2"
#define PLATFORM_TEGRA3_NAME "tegra3"
#define PLATFORM_TEGRA4_NAME "tegra4"
#define PLATFORM_TEGRA5_NAME "tegra5"

class PackageInfo
{
public:
    PackageInfo(int version, int platform, int cpu_id, std::string install_path = "/data/data/");
    PackageInfo(const std::string& fullname, const std::string& install_path, std::string package_version = "0.0");
    std::string GetFullName() const;
    int GetVersion() const;
    int GetPlatform() const;
    int GetCpuID() const;
    std::string GetInstalationPath() const;
    bool operator==(const PackageInfo& package) const;
    static const std::map<int, std::string> PlatformNameMap;
    bool IsValid() const;

protected:
    static std::map<int, std::string> InitPlatformNameMap();
    int Version;
    int Platform;
    int CpuID;
    std::string FullName;
    std::string InstallPath;
    static const std::string BasePackageName;
};

#endif
