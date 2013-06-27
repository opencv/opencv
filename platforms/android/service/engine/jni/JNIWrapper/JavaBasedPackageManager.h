#include "IPackageManager.h"
#include "CommonPackageManager.h"
#include <jni.h>
#include <vector>

class JavaBasedPackageManager: public CommonPackageManager
{
public:
    JavaBasedPackageManager(JavaVM* JavaMashine, jobject MarketConector);
    virtual ~JavaBasedPackageManager();

protected:
    virtual bool InstallPackage(const PackageInfo& package);
    virtual std::vector<PackageInfo> GetInstalledPackages();

private:
    JavaVM* JavaContext;
    jobject JavaPackageManager;

    JavaBasedPackageManager();
    PackageInfo ConvertPackageFromJava(jobject package, JNIEnv* jenv);
};
