#if defined __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main(int argc, char** argv)
{
#ifdef CL_VERSION_1_2
#error OpenCL is valid
#else
#error OpenCL check failed
#endif
    return 0;
}
