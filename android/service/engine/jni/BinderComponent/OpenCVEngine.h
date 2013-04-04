#ifndef __OPEN_CV_ENGINE_H__
#define __OPEN_CV_ENGINE_H__

#include "EngineCommon.h"
#include "IOpenCVEngine.h"
#include "BnOpenCVEngine.h"
#include "IPackageManager.h"
#include <binder/IInterface.h>
#include <binder/Parcel.h>
#include <utils/String8.h>
#include <utils/String16.h>
#include <string>
#include <set>

class OpenCVEngine: public BnOpenCVEngine
{
public:
    OpenCVEngine(IPackageManager* PkgManager);
    int32_t GetVersion();
    android::String16 GetLibPathByVersion(android::String16 version);
    virtual android::String16 GetLibraryList(android::String16 version);
    bool InstallVersion(android::String16 version);

protected:
    IPackageManager* PackageManager;
    static const int KnownVersions[];

    OpenCVEngine();
    bool ValidateVersion(int version);
    int NormalizeVersionString(std::string version);
    bool FixPermissions(const std::string& path);

    static const int Platform;
    static const int CpuID;
};

#endif
