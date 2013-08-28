#include "HardwareDetector.h"
#include "TegraDetector.h"
#include "ProcReader.h"
#include "EngineCommon.h"
#include "StringUtils.h"
#include <utils/Log.h>

using namespace std;

int GetCpuID()
{
    int result = 0;
    map<string, string> cpu_info = GetCpuInfo();
    map<string, string>::const_iterator it;

    #if defined(__i386__)
    LOGD("Using X86 HW detector");
    result |= ARCH_X86;
    it = cpu_info.find("flags");
    if (cpu_info.end() != it)
    {
        set<string> features = SplitString(it->second, ' ');
        if (features.end() != features.find(CPU_INFO_SSE_STR))
        {
            result |= FEATURES_HAS_SSE;
        }
        if (features.end() != features.find(CPU_INFO_SSE2_STR))
        {
            result |= FEATURES_HAS_SSE2;
        }
        if (features.end() != features.find(CPU_INFO_SSSE3_STR))
        {
            result |= FEATURES_HAS_SSSE3;
        }
    }
#elif defined(__mips)
#ifdef __SUPPORT_MIPS
    result |= ARCH_MIPS;
#else
    result = ARCH_UNKNOWN;
#endif
#else
    LOGD("Using ARM HW detector");
    it = cpu_info.find("Processor");

    if (cpu_info.end() != it)
    {
        size_t proc_name_pos = it->second.find(CPU_INFO_ARCH_X86_STR);
        if (string::npos != proc_name_pos)
        {
        }
        else
        {
            proc_name_pos = it->second.find(CPU_INFO_ARCH_ARMV7_STR);
            if (string::npos != proc_name_pos)
            {
                result |= ARCH_ARMv7;
            }
            else
            {
                proc_name_pos = it->second.find(CPU_INFO_ARCH_ARMV6_STR);
                if (string::npos != proc_name_pos)
                {
                    result |= ARCH_ARMv6;
                }
                else
                {
                    proc_name_pos = it->second.find(CPU_INFO_ARCH_ARMV5_STR);
                    if (string::npos != proc_name_pos)
                    {
                        result |= ARCH_ARMv5;
                    }
                }
            }
        }
    }
    else
    {
        return ARCH_UNKNOWN;
    }

    it = cpu_info.find("Features");
    if (cpu_info.end() != it)
    {
        set<string> features = SplitString(it->second, ' ');
        if (features.end() != features.find(CPU_INFO_NEON_STR))
        {
            result |= FEATURES_HAS_NEON;
        }
        if (features.end() != features.find(CPU_INFO_NEON2_STR))
        {
            result |= FEATURES_HAS_NEON2;
        }
        if (features.end() != features.find(CPU_INFO_VFPV3_STR))
        {
            if (features.end () != features.find(CPU_INFO_VFPV3D16_STR))
            {
                result |= FEATURES_HAS_VFPv3d16;
            }
            else
            {
                result |= FEATURES_HAS_VFPv3;
            }
        }
    }
    #endif

    return result;
}

string GetPlatformName()
{
    map<string, string> cpu_info = GetCpuInfo();
    string hardware_name = "";
    map<string, string>::const_iterator hw_iterator = cpu_info.find("Hardware");

    if (cpu_info.end() != hw_iterator)
    {
        hardware_name = hw_iterator->second;
    }

    return hardware_name;
}

int GetProcessorCount()
{
    FILE* cpuPossible = fopen("/sys/devices/system/cpu/possible", "r");
    if(!cpuPossible)
        return 1;

    char buf[2000]; //big enough for 1000 CPUs in worst possible configuration
    char* pbuf = fgets(buf, sizeof(buf), cpuPossible);
    fclose(cpuPossible);
    if(!pbuf)
        return 1;

    //parse string of form "0-1,3,5-7,10,13-15"
        int cpusAvailable = 0;

        while(*pbuf)
        {
            const char* pos = pbuf;
            bool range = false;
            while(*pbuf && *pbuf != ',')
            {
                if(*pbuf == '-') range = true;
                ++pbuf;
            }
            if(*pbuf) *pbuf++ = 0;
            if(!range)
                ++cpusAvailable;
            else
            {
                int rstart = 0, rend = 0;
                sscanf(pos, "%d-%d", &rstart, &rend);
                cpusAvailable += rend - rstart + 1;
            }
        }
        return cpusAvailable ? cpusAvailable : 1;
}

int DetectKnownPlatforms()
{
    int tegra_status = DetectTegra();

    // All Tegra platforms since Tegra3
    if (2 < tegra_status)
    {
        return PLATFORM_TEGRA + tegra_status - 1;
    }
    else
    {
        return PLATFORM_UNKNOWN;
    }
}
