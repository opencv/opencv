#include <acl/acl.h>
#include <unistd.h> // fork()
#include <iostream>

int main(int /*argc*/, char** /*argv*/)
{
    int ret = aclInit(NULL);
    if (ret != 0)
    {
        std::cerr << "Failed to initialize Ascend, ret = " << ret;
    }

    ret = aclFinalize();
    if (ret != 0)
    {
        std::cerr << "Failed to de-initialize Ascend, ret = " << ret;
    }

    return 0;
}
