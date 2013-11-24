#include "EngineCommon.h"
#include "IOpenCVEngine.h"
#include "OpenCVEngine.h"
#include "IPackageManager.h"
#include "NativePackageManager.h"

#include <sys/types.h>
#include <unistd.h>
#include <grp.h>
#include <binder/IPCThreadState.h>
#include <binder/ProcessState.h>
#include <binder/IServiceManager.h>
#include <utils/Log.h>

using namespace android;

int main(int argc, char *argv[])
{
    LOGI("OpenCVEngine native service starting");
    IPackageManager* PackageManager = new NativePackageManager();
    sp<IBinder> Engine = new OpenCVEngine(PackageManager);

    defaultServiceManager()->addService(IOpenCVEngine::descriptor, Engine);
    LOGI("OpenCVEngine native service started successfully");
    ProcessState::self()->startThreadPool();
    IPCThreadState::self()->joinThreadPool();
    LOGI("OpenCVEngine native service finished");

    delete PackageManager;

    return 0;
}
