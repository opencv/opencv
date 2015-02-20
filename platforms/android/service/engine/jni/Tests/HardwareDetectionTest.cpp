#include <gtest/gtest.h>
#include "ProcReader.h"
#include "TegraDetector.h"
#include "HardwareDetector.h"
#include "StringUtils.h"

using namespace std;

TEST(Strip, StripEmptyString)
{
   string a = "";
   EXPECT_FALSE(StripString(a));
}

TEST(Strip, StripClearString)
{
    string a = "qqqwww";
    EXPECT_TRUE(StripString(a));
    EXPECT_STREQ("qqqwww", a.c_str());
}

TEST(Strip, StripStringLeft)
{
    string a = "  qqqwww";
    EXPECT_TRUE(StripString(a));
    EXPECT_STREQ("qqqwww", a.c_str());
}

TEST(Strip, StripStringRight)
{
    string a = "qqqwww  ";
    EXPECT_TRUE(StripString(a));
    EXPECT_STREQ("qqqwww", a.c_str());
}

TEST(Strip, StripStringLeftRight)
{
    string a = "qqqwww  ";
    EXPECT_TRUE(StripString(a));
    EXPECT_STREQ("qqqwww", a.c_str());
}

TEST(Strip, StripStringWithSpaces)
{
    string a = "   qqq www  ";
    EXPECT_TRUE(StripString(a));
    EXPECT_STREQ("qqq www", a.c_str());
}

TEST(Parse, ParseEmptyString)
{
    string a = "";
    string key;
    string value;
    EXPECT_FALSE(ParseString(a, key, value));
}

TEST(Parse, ParseStringWithoutSeparator)
{
    string a = "qqqwww";
    string key;
    string value;
    EXPECT_FALSE(ParseString(a, key, value));
}

TEST(Parse, ParseClearString)
{
    string a = "qqq:www";
    string key;
    string value;
    EXPECT_TRUE(ParseString(a, key, value));
    EXPECT_STREQ("qqq", key.c_str());
    EXPECT_STREQ("www", value.c_str());
}

TEST(Parse, ParseDirtyString)
{
    string a = "qqq :  www";
    string key;
    string value;
    EXPECT_TRUE(ParseString(a, key, value));
    EXPECT_STREQ("qqq", key.c_str());
    EXPECT_STREQ("www", value.c_str());
}

TEST(Split, SplitEmptyString)
{
    string a = "";
    set<string> b = SplitString(a, ' ');
    EXPECT_EQ(0, b.size());
}

TEST(Split, SplitOneElementString)
{
    string a = "qqq";
    set<string> b = SplitString(a, ' ');
    EXPECT_EQ(1, b.size());
    EXPECT_FALSE(b.find("qqq") == b.end());
}

TEST(Split, SplitMultiElementString)
{
    string a = "qqq www eee";
    set<string> b = SplitString(a, ' ');
    EXPECT_EQ(3, b.size());
    EXPECT_FALSE(b.find("qqq") == b.end());
    EXPECT_FALSE(b.find("www") == b.end());
    EXPECT_FALSE(b.find("eee") == b.end());
}

TEST(CpuCount, CheckNonZero)
{
    EXPECT_TRUE(GetProcessorCount() != 0);
}

TEST(GetCpuInfo, GetCpuInfo)
{
    map<string, string> a = GetCpuInfo();
    EXPECT_FALSE(a.empty());
    EXPECT_TRUE(a.find("") == a.end());
}

TEST(CpuID, CheckNotEmpy)
{
    int cpu_id = GetCpuID();
    EXPECT_NE(0, cpu_id);
}

#if defined(__i386__)
TEST(CpuID, CheckX86)
{
    int cpu_id = GetCpuID();
    EXPECT_TRUE(cpu_id & ARCH_X86);
}

TEST(CpuID, CheckSSE2)
{
    int cpu_id = GetCpuID();
    EXPECT_TRUE(cpu_id & FEATURES_HAS_SSE2);
}
#elif defined(__mips)
#ifdef __SUPPORT_MIPS
TEST(CpuID, CheckMips)
{
    int cpu_id = GetCpuID();
    EXPECT_TRUE(cpu_id & ARCH_MIPS);
}
#endif
#else
TEST(TegraDetector, Detect)
{
    EXPECT_TRUE(DetectTegra() != 0);
}

TEST(CpuID, CheckArmV7)
{
    int cpu_id = GetCpuID();
    EXPECT_TRUE(cpu_id & ARCH_ARMv7);
}

TEST(CpuID, CheckNeon)
{
    int cpu_id = GetCpuID();
    EXPECT_TRUE(cpu_id & FEATURES_HAS_NEON);
}

TEST(CpuID, CheckVFPv3)
{
    int cpu_id = GetCpuID();
    EXPECT_TRUE(cpu_id & FEATURES_HAS_VFPv3);
}

TEST(PlatformDetector, CheckTegra)
{
    EXPECT_NE(PLATFORM_UNKNOWN, DetectKnownPlatforms());
}
#endif
