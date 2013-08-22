#include "EngineCommon.h"
#include "IOpenCVEngine.h"

#include <sys/types.h>
#include <unistd.h>
#include <grp.h>

#include <binder/IPCThreadState.h>
#include <binder/ProcessState.h>
#include <binder/IServiceManager.h>
#include <utils/Log.h>

#include <stdio.h>

using namespace android;

int main(int argc, char *argv[])
{
    LOGI("OpenCVEngine client is now starting");

    sp<IServiceManager> ServiceManager = defaultServiceManager();
    sp<IBinder> EngineService;
    sp<IOpenCVEngine> Engine;

    LOGI("Trying to contect to service");

    do {
        EngineService = ServiceManager->getService(IOpenCVEngine::descriptor);
        if (EngineService != 0) break;
        LOGW("OpenCVEngine not published, waiting...");
        usleep(500000); // 0.5 s
    } while(true);

    LOGI("Connection established");

    Engine = interface_cast<IOpenCVEngine>(EngineService);
    int32_t EngineVersion = Engine->GetVersion();

    printf("OpenCVEngine version %d started", EngineVersion);

    return 0;
}
