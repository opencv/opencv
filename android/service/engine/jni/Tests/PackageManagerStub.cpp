#include "PackageManagerStub.h"

using namespace std;

bool PackageManagerStub::InstallPackage(const PackageInfo& package)
{
    InstalledPackages.push_back(PackageInfo(package.GetFullName(), "/data/data/" + package.GetFullName() + "/lib"));
    return true;
}

vector<PackageInfo> PackageManagerStub::GetInstalledPackages()
{
    return InstalledPackages;
}

PackageManagerStub::~PackageManagerStub()
{
}