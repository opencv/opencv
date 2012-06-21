#ifndef __COMMON_PACKAGE_MANAGER_H__
#define __COMMON_PACKAGE_MANAGER_H__

#include "IPackageManager.h"
#include "PackageInfo.h"
#include <set>
#include <vector>
#include <string>

class CommonPackageManager: public IPackageManager
{
public:
    std::set<std::string> GetInstalledVersions();
    bool CheckVersionInstalled(const std::string& version, int platform, int cpu_id);
    bool InstallVersion(const std::string& version, int platform, int cpu_id);
    std::string GetPackagePathByVersion(const std::string& version, int platform, int cpu_id);
    virtual ~CommonPackageManager();

protected:
    static std::vector<std::pair<int, int> > ArmRating;
    static std::vector<std::pair<int, int> > IntelRating;
    
    static std::vector<std::pair<int, int> > InitArmRating();
    static std::vector<std::pair<int, int> > InitIntelRating();
    
    bool IsVersionCompatible(const std::string& target_version, const std::string& package_version);
    int GetHardwareRating(int platform, int cpu_id, const std::vector<std::pair<int, int> >& group);
    
    virtual bool InstallPackage(const PackageInfo& package) = 0;
    virtual std::vector<PackageInfo> GetInstalledPackages() = 0;   
};


#endif