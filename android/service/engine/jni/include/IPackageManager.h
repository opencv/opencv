#ifndef __IPACKAGE_MANAGER__
#define __IPACKAGE_MANAGER__

#include <set>
#include <string>

class IPackageManager
{
public:
    virtual std::set<std::string> GetInstalledVersions() = 0;
    virtual bool CheckVersionInstalled(const std::string& version, int platform, int cpu_id) = 0;
    virtual bool InstallVersion(const std::string&, int platform, int cpu_id) = 0;
    virtual std::string GetPackagePathByVersion(const std::string&, int platform, int cpu_id) = 0;
    virtual ~IPackageManager(){};
};

#endif