#ifndef __PACKAGE_MANAGER_STUB_H__
#define __PACKAGE_MANAGER_STUB_H__

#include "IPackageManager.h"
#include "CommonPackageManager.h"

class PackageManagerStub: public CommonPackageManager
{
public:
    std::vector<PackageInfo> InstalledPackages;
    virtual ~PackageManagerStub();
protected:
    virtual bool InstallPackage(const PackageInfo& package);
    virtual std::vector<PackageInfo> GetInstalledPackages();
};

#endif
