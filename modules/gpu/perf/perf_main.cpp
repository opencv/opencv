#include "perf_precomp.hpp"

namespace{

static void printOsInfo()
{
#if defined _WIN32
#   if defined _WIN64
        printf("[----------]\n[ GPU INFO ] \tRun on OS Windows x64.\n[----------]\n"), fflush(stdout);
#   else
        printf("[----------]\n[ GPU INFO ] \tRun on OS Windows x32.\n[----------]\n"), fflush(stdout);
#   endif
#elif defined linux
#   if defined _LP64
        printf("[----------]\n[ GPU INFO ] \tRun on OS Linux x64.\n[----------]\n"), fflush(stdout);
#   else
        printf("[----------]\n[ GPU INFO ] \tRun on OS Linux x32.\n[----------]\n"), fflush(stdout);
#   endif
#elif defined __APPLE__
#   if defined _LP64
        printf("[----------]\n[ GPU INFO ] \tRun on OS Apple x64.\n[----------]\n"), fflush(stdout);
#   else
        printf("[----------]\n[ GPU INFO ] \tRun on OS Apple x32.\n[----------]\n"), fflush(stdout);
#   endif
#endif

}

static void printCudaInfo()
{
    printOsInfo();
#ifndef HAVE_CUDA
    printf("[----------]\n[ GPU INFO ] \tOpenCV was built without CUDA support.\n[----------]\n"), fflush(stdout);
#else
    int driver;
    cudaDriverGetVersion(&driver);

    printf("[----------]\n"), fflush(stdout);
    printf("[ GPU INFO ] \tCUDA Driver  version: %d.\n", driver), fflush(stdout);
    printf("[ GPU INFO ] \tCUDA Runtime version: %d.\n", CUDART_VERSION), fflush(stdout);
    printf("[----------]\n"), fflush(stdout);

    printf("[----------]\n"), fflush(stdout);
    printf("[ GPU INFO ] \tGPU module was compiled for the following GPU archs.\n"), fflush(stdout);
    printf("[      BIN ] \t%s.\n", CUDA_ARCH_BIN), fflush(stdout);
    printf("[      PTX ] \t%s.\n", CUDA_ARCH_PTX), fflush(stdout);
    printf("[----------]\n"), fflush(stdout);

    printf("[----------]\n"), fflush(stdout);
    int deviceCount = cv::gpu::getCudaEnabledDeviceCount();
    printf("[ GPU INFO ] \tCUDA device count:: %d.\n", deviceCount), fflush(stdout);
    printf("[----------]\n"), fflush(stdout);

    for (int i = 0; i < deviceCount; ++i)
    {
        cv::gpu::DeviceInfo info(i);

        printf("[----------]\n"), fflush(stdout);
        printf("[ DEVICE   ] \t# %d %s.\n", i, info.name().c_str()), fflush(stdout);
        printf("[          ] \tCompute capability: %d.%d\n", (int)info.majorVersion(), (int)info.minorVersion()), fflush(stdout);
        printf("[          ] \tMulti Processor Count:  %d\n", info.multiProcessorCount()), fflush(stdout);
        printf("[          ] \tTotal memory: %d Mb\n", static_cast<int>(static_cast<int>(info.totalMemory() / 1024.0) / 1024.0)), fflush(stdout);
        printf("[          ] \tFree  memory: %d Mb\n", static_cast<int>(static_cast<int>(info.freeMemory()  / 1024.0) / 1024.0)), fflush(stdout);
        if (!info.isCompatible())
            printf("[ GPU INFO ] \tThis device is NOT compatible with current GPU module build\n");
        printf("[----------]\n"), fflush(stdout);
    }

#endif
}

}

CV_PERF_TEST_MAIN(gpu, printCudaInfo())