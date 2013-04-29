#ifndef __COMMON_PACKAGE_MANAGER_H__
#define __COMMON_PACKAGE_MANAGER_H__

#include "IPackageManager.h"
#include "PackageInfo.h"
#include <vector>
#include <string>

class CommonPackageManager: public IPackageManager
{
public:
    std::vector<int> GetInstalledVersions();
    bool CheckVersionInstalled(int version, int platform, int cpu_id);
    bool InstallVersion(int version, int platform, int cpu_id);
    std::string GetPackagePathByVersion(int version, int platform, int cpu_id);
    virtual ~CommonPackageManager();

protected:
    static const std::vector<std::pair<int, int> > ArchRatings[];

    static std::vector<std::pair<int, int> > InitArmRating();
    static std::vector<std::pair<int, int> > InitIntelRating();
    static std::vector<std::pair<int, int> > InitMipsRating();

    bool IsVersionCompatible(int target_version, int package_version);
    int GetHardwareRating(int platform, int cpu_id, const std::vector<std::pair<int, int> >& group);

    virtual bool InstallPackage(const PackageInfo& package) = 0;
    virtual std::vector<PackageInfo> GetInstalledPackages() = 0;
};


#endif
