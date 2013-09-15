#include "PackageManagerStub.h"

using namespace std;

bool PackageManagerStub::InstallPackage(const PackageInfo& package)
{
    InstalledPackages.push_back(package);
    return true;
}

vector<PackageInfo> PackageManagerStub::GetInstalledPackages()
{
    return InstalledPackages;
}

PackageManagerStub::~PackageManagerStub()
{
}
