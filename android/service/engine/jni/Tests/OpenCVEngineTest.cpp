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
	PackageInfo info("230", PLATFORM_UNKNOWN, ARCH_ARMv7);
	PackageManager->InstalledPackages.push_back(info);
	
	Engine = new OpenCVEngine(PackageManager);
	
	defaultServiceManager()->addService(IOpenCVEngine::descriptor, Engine);
	     LOGI("OpenCVEngine native service started successfully");
	ProcessState::self()->startThreadPool();
    }
    ~ServiceStarter()
    {
	delete PackageManager;
    }
private:
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

TEST(OpenCVEngineTest, InstallVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    EXPECT_FALSE(NULL == Engine.get());
    bool result = Engine->InstallVersion(String16("2.4"));
    EXPECT_EQ(true, result);
}

TEST(OpenCVEngineTest, GetPathForExistVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.3"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v23_armv7/lib",String8(result).string());
}

TEST(OpenCVEngineTest, GetPathForUnExistVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    EXPECT_FALSE(NULL == Engine.get());
    String16 result = Engine->GetLibPathByVersion(String16("2.5"));
    EXPECT_EQ(0, result.size());
}

TEST(OpenCVEngineTest, InstallAndGetVersion)
{
    sp<IOpenCVEngine> Engine = InitConnect();
    EXPECT_FALSE(NULL == Engine.get());
    EXPECT_TRUE(Engine->InstallVersion(String16("2.5")));
    String16 result = Engine->GetLibPathByVersion(String16("2.5"));
    EXPECT_STREQ("/data/data/org.opencv.lib_v25_tegra3/lib", String8(result).string());
}