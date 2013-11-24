#ifndef __BP_OPENCV_ENGINE_H__
#define __BP_OPENCV_ENGINE_H__

#include "IOpenCVEngine.h"
#include <binder/IInterface.h>
#include <binder/Parcel.h>
#include <utils/String16.h>

class BpOpenCVEngine: public android::BpInterface<IOpenCVEngine>
{
public:
    BpOpenCVEngine(const android::sp<android::IBinder>& impl);
    virtual ~BpOpenCVEngine();
    virtual int GetVersion();
    virtual android::String16 GetLibPathByVersion(android::String16 version);
    virtual android::String16 GetLibraryList(android::String16 version);
    virtual bool InstallVersion(android::String16 version);
};

#endif
