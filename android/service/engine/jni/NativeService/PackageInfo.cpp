#include "EngineCommon.h"
#include "PackageInfo.h"
#include "HardwareDetector.h"
#include "IOpenCVEngine.h"
#include "StringUtils.h"
#include <assert.h>
#include <vector>
#include <utils/Log.h>

using namespace std;

map<int, string> PackageInfo::InitPlatformNameMap()
{
    map<int, string> result;
    
    // TODO: Do not forget to add Platrfom constant to HardwareDetector.h
    result[PLATFORM_TEGRA] = PLATFORM_TEGRA_NAME;
    result[PLATFORM_TEGRA2] = PLATFORM_TEGRA2_NAME;
    result[PLATFORM_TEGRA3] = PLATFORM_TEGRA3_NAME;
    
    return result;
}

const map<int, string> PackageInfo::PlatformNameMap = InitPlatformNameMap();
const string PackageInfo::BasePackageName = "org.opencv.lib";

inline string JoinARMFeatures(int cpu_id)
{
    string result;

    if (FEATURES_HAS_NEON2 & cpu_id)
    {
	if (!((ARCH_ARMv5 & cpu_id) || (ARCH_ARMv6 & cpu_id) ||(ARCH_ARMv7 & cpu_id)))
	    result = string(FEATURES_HAS_NEON2_NAME);
    }    
    else if (FEATURES_HAS_NEON & cpu_id)
    {
	if (!((ARCH_ARMv5 & cpu_id) || (ARCH_ARMv6 & cpu_id)))
	    result = string(FEATURES_HAS_NEON_NAME);
    }
    else if (FEATURES_HAS_VFPv3 & cpu_id)
    {
	if ((ARCH_ARMv5 & cpu_id) || (ARCH_ARMv6 & cpu_id))
	    result = string(FEATURES_HAS_VFPv3_NAME);
    }
    else if (FEATURES_HAS_VFPv3d16 & cpu_id)
    {
	if ((ARCH_ARMv5 & cpu_id) || (ARCH_ARMv6 & cpu_id))
	    result = string(FEATURES_HAS_VFPv3d16_NAME);
    }
    
    return result;
}

inline int SplitARMFeatures(const vector<string>& features)
{
    int result = 0;
    
    for (size_t i = 3; i < features.size(); i++)
    {
	if (FEATURES_HAS_VFPv3_NAME == features[i])
	{
	    result |= FEATURES_HAS_VFPv3;
	}
	else if (FEATURES_HAS_VFPv3d16_NAME == features[i])
	{
	    result |= FEATURES_HAS_VFPv3d16;
	}
	else if (FEATURES_HAS_NEON_NAME == features[i])
	{
	    result |= FEATURES_HAS_NEON;
	}
	else if (FEATURES_HAS_NEON2_NAME == features[i])
	{
	    result |= FEATURES_HAS_NEON2;
	}
    }
    
    return result;
}

inline string JoinIntelFeatures(int cpu_id)
{
    string result;
    
    if (FEATURES_HAS_SSSE3 & cpu_id)
    {
	result = FEATURES_HAS_SSSE3_NAME;
    }
    else if (FEATURES_HAS_SSE2 & cpu_id)
    {
	result = FEATURES_HAS_SSE2_NAME;
    }
    else if (FEATURES_HAS_SSE & cpu_id)
    {
	result = FEATURES_HAS_SSE_NAME;
    }
    
    return result;
}

inline int SplitIntelFeatures(const vector<string>& features)
{
    int result = 0;
    
    for (size_t i = 3; i < features.size(); i++)
    {
	if (FEATURES_HAS_SSSE3_NAME == features[i])
	{
	    result |= FEATURES_HAS_SSSE3;
	}
	else if (FEATURES_HAS_SSE2_NAME == features[i])
	{
	    result |= FEATURES_HAS_SSE2;
	}
	else if (FEATURES_HAS_SSE_NAME == features[i])
	{
	    result |= FEATURES_HAS_SSE;
	}
    }    
    
    return result;
}

inline string SplitVersion(const vector<string>& features, const string& package_version)
{
    string result;
    
    if ((features.size() > 1) && ('v' == features[1][0]))
    {
	result = features[1].substr(1);
	result += SplitStringVector(package_version, '.')[0];
    }
    else
    {
	// TODO: Report package name format error
    }
    
    return result;
}

inline string JoinPlatform(int platform)
{
    string result;
    map<int, string>::const_iterator it = PackageInfo::PlatformNameMap.find(platform);
    
    assert(PackageInfo::PlatformNameMap.end() != it);
    result = it->second;
    
    return result;
}

inline int SplitPlatfrom(const vector<string>& features)
{
    int result = 0;
    
    if (features.size() > 2)
    {
	string tmp = features[2];
	if (PLATFORM_TEGRA_NAME == tmp)
	{
	    result = PLATFORM_TEGRA;
	}
	else if (PLATFORM_TEGRA2_NAME == tmp)
	{
	    result = PLATFORM_TEGRA2;
	}
	else if (PLATFORM_TEGRA3_NAME == tmp)
	{
	    result = PLATFORM_TEGRA3;
	}
    }
    else
    {
	// TODO: Report package name format error
    }
	
    return result;
}

/* Package naming convention
 * All parts of package name seporated by "_" symbol
 * First part is base namespace.
 * Second part is version. Version starts from "v" symbol. After "v" symbol version nomber without dot symbol added.
 * If platform is known third part is platform name
 * If platform is unknown it is defined by hardware capabilities using pattern: <arch>_<floating point and vectorization features>_<other features>
 * Example: armv7_neon, armv5_vfpv3
 */ 
PackageInfo::PackageInfo(const string& version, int platform, int cpu_id, std::string install_path):
    Version(version),
    Platform(platform),
    CpuID(cpu_id),
    InstallPath("")
{
#ifndef __SUPPORT_TEGRA3
    Platform = PLATFORM_UNKNOWN;
#endif
    FullName = BasePackageName + "_v" + Version.substr(0, Version.size()-1);
    if (PLATFORM_UNKNOWN != Platform)
    {
	FullName += string("_") + JoinPlatform(platform);
    }
    else
    {
	if (ARCH_UNKNOWN != CpuID)
	{
	    if (ARCH_X86 & CpuID)
	    {
		LOGD("Found processor with x86 arch");
		FullName += string("_") + ARCH_X86_NAME;
#ifdef __SUPPORT_INTEL_FEATURES
		string features = JoinIntelFeatures(CpuID);
#else
		string features;
#endif
		if (!features.empty())
		{
		    FullName += string("_") + features;
		}
	    }
	    else if (ARCH_X64 & CpuID)
	    {
		LOGD("Found processor with x64 arch");
#ifdef __SUPPORT_INTEL_x64
		FullName += string("_") + ARCH_X64_NAME;
#else
		FullName += string("_") + ARCH_X86_NAME;
#endif
#ifdef __SUPPORT_INTEL_FEATURES
		string features = JoinIntelFeatures(CpuID);
#else
		string features;
#endif
		if (!features.empty())
		{
		    FullName += string("_") + features;
		}
	    }
	    else if (ARCH_ARMv5 & CpuID)
	    {
		LOGD("Found processor with ARMv5 arch");
		FullName += string("_") + ARCH_ARMv5_NAME;
#ifdef __SUPPORT_ARMEABI_FEATURES
		string features = JoinARMFeatures(CpuID);
#else
		string features;
#endif
		if (!features.empty())
		{
		    FullName += string("_") + features;
		}
	    }
	    else if (ARCH_ARMv6 & CpuID)
	    {
		LOGD("Found processor with ARMv6 arch");
		// NOTE: ARM v5 used instead ARM v6
		//FullName += string("_") + ARCH_ARMv6_NAME;
		FullName += string("_") + ARCH_ARMv5_NAME;
#ifdef __SUPPORT_ARMEABI_FEATURES
		string features = JoinARMFeatures(CpuID);
#else
		string features;
#endif
		if (!features.empty())
		{
		    FullName += string("_") + features;
		}
	    }
	    else if (ARCH_ARMv7 & CpuID)
	    {
		LOGD("Found processor with ARMv7 arch");
		FullName += string("_") + ARCH_ARMv7_NAME;
#ifdef __SUPPORT_ARMEABI_V7A_FEATURES
		string features = JoinARMFeatures(CpuID);
#else
		string features;
#endif
		if (!features.empty())
		{
		    FullName += string("_") + features;
		}
	    }
	    else if (ARCH_ARMv8 & CpuID)
	    {
		LOGD("Found processor with ARMv8 arch");
#ifdef __SUPPORT_ARMEABI_V8		
		FullName += string("_") + ARCH_ARMv8_NAME;
#else
		FullName += string("_") + ARCH_ARMv7_NAME;
#endif
		//string features = JoinARMFeatures(CpuID);
		//if (!features.empty())
		//{
		//    FullName += string("_") + features;
		//}
	    }
	    else
	    {
		LOGD("Found processor with unknown arch");
		Version.clear();
		CpuID = ARCH_UNKNOWN;
		Platform = PLATFORM_UNKNOWN;
	    }
	}
	else
	{
	    LOGD("Found processor with unknown arch");
	    Version.clear();
	    CpuID = ARCH_UNKNOWN;
	    Platform = PLATFORM_UNKNOWN;
	}
    }
    
    if (!FullName.empty())
    {
	InstallPath = install_path + FullName + "/lib";
    }
}

PackageInfo::PackageInfo(const string& fullname, const string& install_path, const string& package_version):
    FullName(fullname),
    InstallPath(install_path)
{
    LOGD("PackageInfo::PackageInfo(\"%s\", \"%s\", \"%s\")", fullname.c_str(), install_path.c_str(), package_version.c_str());
    
    assert(!fullname.empty());
    assert(!install_path.empty());
    
    vector<string> features = SplitStringVector(FullName, '_');
    
    if (!features.empty() && (BasePackageName == features[0])) 
    {
	Version = SplitVersion(features, package_version);
	if (Version.empty())
	{
	    CpuID = ARCH_UNKNOWN;
	    Platform = PLATFORM_UNKNOWN;
	    return;
	}
	
	Platform = SplitPlatfrom(features);
	if (PLATFORM_UNKNOWN != Platform)
	{
	   CpuID = 0;
	}
	else
	{
	    if (features.size() < 3)
	    {
		LOGD("It is not OpenCV library package for this platform");
		Version.clear();
		CpuID = ARCH_UNKNOWN;
		Platform = PLATFORM_UNKNOWN;
		return;
	    }
	    else if (ARCH_ARMv5_NAME == features[2])
	    {
		CpuID = ARCH_ARMv5 | SplitARMFeatures(features);
	    }
	    else if (ARCH_ARMv6_NAME == features[2])
	    {
		CpuID = ARCH_ARMv6 | SplitARMFeatures(features);
	    }
	    else if (ARCH_ARMv7_NAME == features[2])
	    {
		CpuID = ARCH_ARMv7 | SplitARMFeatures(features);
	    }
	    else if (ARCH_X86_NAME == features[2])
	    {
		CpuID = ARCH_X86 | SplitIntelFeatures(features);
	    }
	    else if (ARCH_X64_NAME == features[2])
	    {
		CpuID = ARCH_X64 | SplitIntelFeatures(features);
	    }
	    else
	    {
		LOGD("It is not OpenCV library package for this platform");
		Version.clear();
		CpuID = ARCH_UNKNOWN;
		Platform = PLATFORM_UNKNOWN;
		return;
	    }
	}
    }
    else
    {
	LOGD("It is not OpenCV library package for this platform");
	Version.clear();
	CpuID = ARCH_UNKNOWN;
	Platform = PLATFORM_UNKNOWN;
	return;
    }
}

bool PackageInfo::IsValid() const
{
    return !(Version.empty() && (PLATFORM_UNKNOWN == Platform) && (ARCH_UNKNOWN == CpuID));
}

int PackageInfo::GetPlatform() const
{
    return Platform;
}

int PackageInfo::GetCpuID() const
{
    return CpuID;
}

string PackageInfo::GetFullName() const
{
    return FullName;
}

string PackageInfo::GetVersion() const
{
    return Version;
}

string PackageInfo::GetInstalationPath() const
{
    return InstallPath;
}

bool PackageInfo::operator==(const PackageInfo& package) const
{
    return (package.FullName == FullName);
}