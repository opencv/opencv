#if defined __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifndef _MSC_VER
#ifdef CL_VERSION_1_2
#error OpenCL is valid
#else
#error OpenCL check failed
#endif
#else
#ifdef CL_VERSION_1_2
#pragma message ("OpenCL is valid")
#else
#pragma message ("OpenCL check failed")
#endif
#endif

int main(int /*argc*/, char** /*argv*/)
{
    return 0;
}
