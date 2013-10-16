#if defined __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main(int argc, char** argv)
{
#ifdef CL_VERSION_1_2
#else
#error OpenCL 1.2 not found
#endif
    return 0;
}
