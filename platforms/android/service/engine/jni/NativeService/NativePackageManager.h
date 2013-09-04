#ifndef __NATIVE_PACKAGE_MANAGER_STUB_H__
#define __NATIVE_PACKAGE_MANAGER_STUB_H__

#include "IPackageManager.h"
#include "CommonPackageManager.h"

class NativePackageManager: public CommonPackageManager
{
public:
    virtual ~NativePackageManager();
protected:
    virtual bool InstallPackage(const PackageInfo& package);
    virtual std::vector<PackageInfo> GetInstalledPackages();
};

#endif
