#include "IOpenCVEngine.h"
#include "BpOpenCVEngine.h"

using namespace android;

BpOpenCVEngine::BpOpenCVEngine(const sp<IBinder>& impl):
    BpInterface<IOpenCVEngine>(impl)
{
}

BpOpenCVEngine::~BpOpenCVEngine()
{
}

// Notes about data transaction:
// Java Binder Wrapper call readInt32 before reading return data
// It treet this in value as exception code
// This implementation support this feature

int BpOpenCVEngine::GetVersion()
{
    Parcel data, reply;

    data.writeInterfaceToken(IOpenCVEngine::descriptor);
    remote()->transact(OCVE_GET_ENGINE_VERSION, data, &reply, 0);
    // read exception code
    reply.readInt32();

    return reply.readInt32();
}

String16 BpOpenCVEngine::GetLibPathByVersion(String16 version)
{
    Parcel data, reply;

    data.writeInterfaceToken(IOpenCVEngine::descriptor);
    data.writeString16(version);
    remote()->transact(OCVE_GET_LIB_PATH_BY_VERSION, data, &reply, 0);
    // read exception code
    reply.readInt32();

    return reply.readString16();
}

android::String16 BpOpenCVEngine::GetLibraryList(String16 version)
{
    Parcel data, reply;

    data.writeInterfaceToken(IOpenCVEngine::descriptor);
    data.writeString16(version);
    remote()->transact(OCVE_GET_LIB_LIST, data, &reply, 0);
    // read exception code
    reply.readInt32();

    return reply.readString16();
}

bool BpOpenCVEngine::InstallVersion(String16 version)
{
    Parcel data, reply;

    data.writeInterfaceToken(IOpenCVEngine::descriptor);
    data.writeString16(version);
    remote()->transact(OCVE_INSTALL_VERSION, data, &reply, 0);
    // read exception code
    reply.readInt32();

    return static_cast<bool>(reply.readInt32());
}

IMPLEMENT_META_INTERFACE(OpenCVEngine, OPECV_ENGINE_CLASSNAME)
