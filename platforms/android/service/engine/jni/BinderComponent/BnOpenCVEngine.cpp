#include "EngineCommon.h"
#include "IOpenCVEngine.h"
#include "BnOpenCVEngine.h"
#include <utils/Log.h>
#include <utils/String8.h>
#include <utils/String16.h>

using namespace android;

BnOpenCVEngine::~BnOpenCVEngine()
{
}

// Notes about data transaction:
// Java Binder Wrapper call readInt32 before reading return data
// It treet this in value as exception code
// OnTransact method support this feature
status_t BnOpenCVEngine::onTransact(uint32_t code, const Parcel& data, android::Parcel* reply, uint32_t flags)
{
    LOGD("OpenCVEngine::OnTransact(%u,%u)", code, flags);

    switch(code)
    {
        case OCVE_GET_ENGINE_VERSION:
        {
            LOGD("OpenCVEngine OCVE_GET_ENGINE_VERSION request");
            CHECK_INTERFACE(IOpenCVEngine, data, reply);
            LOGD("OpenCVEngine::GetVersion()");
            reply->writeInt32(0);
            return reply->writeInt32(GetVersion());
        } break;
        case OCVE_GET_LIB_PATH_BY_VERSION:
        {
            LOGD("OpenCVEngine OCVE_GET_LIB_PATH_BY_VERSION request");
            CHECK_INTERFACE(IOpenCVEngine, data, reply);
            const String16 version = data.readString16();
            LOGD("OpenCVEngine::GetLibPathByVersion(%s)", String8(version).string());
            String16 path = GetLibPathByVersion(version);
            reply->writeInt32(0);
            return reply->writeString16(path);
        } break;
        case OCVE_GET_LIB_LIST:
        {
            LOGD("OpenCVEngine OCVE_GET_LIB_LIST request");
            CHECK_INTERFACE(IOpenCVEngine, data, reply);
            const String16 version = data.readString16();
            LOGD("OpenCVEngine::GetLibraryList(%s)", String8(version).string());
            String16 path = GetLibraryList(version);
            reply->writeInt32(0);
            return reply->writeString16(path);
        } break;
        case OCVE_INSTALL_VERSION:
        {
            LOGD("OpenCVEngine OCVE_INSTALL_VERSION request");
            CHECK_INTERFACE(IOpenCVEngine, data, reply);
            const String16 version = data.readString16();
            LOGD("OpenCVEngine::InstallVersion(%s)", String8(version).string());
            bool result = InstallVersion(version);
            reply->writeInt32(0);
            int res = reply->writeInt32(static_cast<int32_t>(result));
            LOGD("InstallVersion call to Binder finished with res %d", res);
            return res;
        } break;
        default:
        {
            LOGD("OpenCVEngine unknown request");
            return BBinder::onTransact(code, data, reply, flags);
        }
    }

    return android::NO_ERROR;
}
