#ifndef __IPACKAGE_MANAGER__
#define __IPACKAGE_MANAGER__

#include <vector>
#include <string>

class IPackageManager
{
public:
    virtual std::vector<int> GetInstalledVersions() = 0;
    virtual bool CheckVersionInstalled(int version, int platform, int cpu_id) = 0;
    virtual bool InstallVersion(int version, int platform, int cpu_id) = 0;
    virtual std::string GetPackagePathByVersion(int version, int platform, int cpu_id) = 0;
    virtual ~IPackageManager(){};
};

#endif
