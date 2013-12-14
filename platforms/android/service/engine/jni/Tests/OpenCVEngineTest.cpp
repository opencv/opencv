#include "IOpenCVEngine.h"
#include "EngineCommon.h"
#include "OpenCVEngine.h"
#include "IPackageManager.h"
#include "PackageManagerStub.h"
#include "PackageInfo.h"
#include "HardwareDetector.h"

#include <gtest/gtest.h>

#include <binder/IPCThreadState.h>
#include <binder/ProcessState.h>
#include <binder/IServiceManager.h>
#include <utils/Log.h>

using namespace android;

class ServiceStarter
{
public:
    ServiceStarter()
    {
        PackageManager = new PackageManagerStub();
        Engine = new OpenCVEngine(PackageManager);

        defaultServiceManager()->addService(IOpenCVEngine::descriptor, Engine);
        LOGI("OpenCVEngine native service started successfully");
        ProcessState::self()->startThreadPool();
    }
    ~ServiceStarter()
    {
        delete PackageManager;
    }

    PackageManagerStub* PackageManager;
    sp<IBinder> Engine;
};

static ServiceStarter Starter;

sp<IOpenCVEngine> InitConnect()
{
    sp<IServiceManager> ServiceManager = defaultServiceManager();
    sp<IBinder> EngineService;
    sp<IOpenCVEngine> Engine;

    do
    {
        EngineService = ServiceManager->getService(IOpenCVEngine::descriptor);
        if (EngineService != 0) break;
        usleep(500000); // 0.5 s
    } while(true);

    Engine = interface_cast<IOpenCVEngine>(EngineService);

    return Engine;
}

TEST(OpenCVEngineTest, GetVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    EXPECT_FALSE(NULL == Engine.get());
    int32_t Version = Engine->GetVersion();
    EXPECT_EQ(OPEN_CV_ENGINE_VERSION, Version);
}

#if defined(__i386__)
TEST(OpenCVEngineTest, GetPathForExecHWExistVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040000, PLATFORM_UNKNOWN, ARCH_X86);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_x86/lib", String8(result).string());
}

TEST(OpenCVEngineTest, GetPathForExecHWOldVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040200, PLATFORM_UNKNOWN, ARCH_X86);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4.1"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_x86/lib", String8(result).string());
}

TEST(OpenCVEngineTest, GetPathForExecHWNewVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040100, PLATFORM_UNKNOWN, ARCH_X86);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4.2"));
    EXPECT_EQ(0, result.size());
}

#elif defined (__mips)
TEST(OpenCVEngineTest, GetPathForExecHWExistVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040000, PLATFORM_UNKNOWN, ARCH_MIPS);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_mips/lib", String8(result).string());
}

TEST(OpenCVEngineTest, GetPathForExecHWOldVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040200, PLATFORM_UNKNOWN, ARCH_MIPS);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4.1"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_mips/lib", String8(result).string());
}

TEST(OpenCVEngineTest, GetPathForExecHWNewVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040100, PLATFORM_UNKNOWN, ARCH_MIPS);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4.2"));
    EXPECT_EQ(0, result.size());
}

#else // armeabi
TEST(OpenCVEngineTest, GetPathForExecHWExistVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040000, PLATFORM_TEGRA3, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4"));
#ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_tegra3/lib", String8(result).string());
#else
    #ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a_neon/lib", String8(result).string());
    #else
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", String8(result).string());
    #endif
#endif
}

TEST(OpenCVEngineTest, GetPathForExecHWOldVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040200, PLATFORM_TEGRA3, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4.1"));
#ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_tegra3/lib", String8(result).string());
#else
    #ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a_neon/lib", String8(result).string());
    #else
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", String8(result).string());
    #endif
#endif
}

TEST(OpenCVEngineTest, GetPathForExecHWNewVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040100, PLATFORM_TEGRA3, ARCH_ARMv7 | FEATURES_HAS_NEON);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4.2"));
    EXPECT_EQ(0, result.size());
}

TEST(OpenCVEngineTest, GetPathForCompatiblePackage1)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040200, PLATFORM_UNKNOWN, ARCH_ARMv5);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv5/lib", String8(result).string());
}

TEST(OpenCVEngineTest, GetPathForCompatiblePackage2)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040200, PLATFORM_TEGRA3, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_NEON);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4"));
#ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_tegra3/lib", String8(result).string());
#else
# ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a_neon/lib", String8(result).string());
# else
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", String8(result).string());
# endif
#endif
}

TEST(OpenCVEngineTest, GetPathForCompatiblePackage3)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040400, PLATFORM_TEGRA4, ARCH_ARMv7 | FEATURES_HAS_VFPv3 | FEATURES_HAS_VFPv4 | FEATURES_HAS_NEON);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4"));
#ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_tegra4/lib", String8(result).string());
#else
# ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a_neon/lib", String8(result).string());
# else
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", String8(result).string());
# endif
#endif
}

TEST(OpenCVEngineTest, InstallAndGetVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    EXPECT_FALSE(NULL == Engine.get());
    EXPECT_TRUE(Engine->InstallVersion(String16("2.4")));
    String16 result = Engine->GetLibPathByVersion(String16("2.4"));
#ifdef __SUPPORT_TEGRA3
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_tegra3/lib", String8(result).string());
#else
# ifdef __SUPPORT_ARMEABI_V7A_FEATURES
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a_neon/lib", String8(result).string());
# else
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", String8(result).string());
# endif
#endif
}

TEST(OpenCVEngineTest, GetPathFor2_4_2)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040200, PLATFORM_UNKNOWN, ARCH_ARMv7);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4.2"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", String8(result).string());
}

TEST(OpenCVEngineTest, GetPathFor2_4_3)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040300, PLATFORM_UNKNOWN, ARCH_ARMv7);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4.3"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", String8(result).string());
}

TEST(OpenCVEngineTest, GetPathFor2_4_3_1)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040301, PLATFORM_UNKNOWN, ARCH_ARMv7);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4.3.1"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", String8(result).string());
}

TEST(OpenCVEngineTest, GetPathFor2_4_3_2)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040302, PLATFORM_UNKNOWN, ARCH_ARMv7);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4.3.2"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", String8(result).string());
}

TEST(OpenCVEngineTest, GetPathFor2_4_4)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040400, PLATFORM_UNKNOWN, ARCH_ARMv7);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4.4"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", String8(result).string());
}

TEST(OpenCVEngineTest, GetPathFor2_4_5)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040500, PLATFORM_UNKNOWN, ARCH_ARMv7);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4.5"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v24_armv7a/lib", String8(result).string());
}
#endif

#ifndef __i386__
TEST(OpenCVEngineTest, GetPathForInCompatiblePackage1)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040200, PLATFORM_UNKNOWN, ARCH_X64);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4"));
    EXPECT_EQ(0, result.size());
}
#else
TEST(OpenCVEngineTest, GetPathForInCompatiblePackage1)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    Starter.PackageManager->InstalledPackages.clear();
    Starter.PackageManager->InstallVersion(2040200, PLATFORM_UNKNOWN, ARCH_ARMv7);
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.4"));
    EXPECT_EQ(0, result.size());
}
#endif

TEST(OpenCVEngineTest, GetPathForUnExistVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.5"));
    EXPECT_EQ(0, result.size());
}
