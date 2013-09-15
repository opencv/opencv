#include "NativePackageManager.h"

using namespace std;

bool NativePackageManager::InstallPackage(const PackageInfo& package)
{
    return false;
}

vector<PackageInfo> NativePackageManager::GetInstalledPackages()
{
    vector<PackageInfo> result;

    return result;
}

NativePackageManager::~NativePackageManager()
{
}
