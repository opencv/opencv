/* This sample demonstrates the way you can perform independed tasks 
   on the different GPUs */

// Disable some warnings which are caused with CUDA headers
#if defined(_MSC_VER)
#pragma warning(disable: 4201 4408 4100)
#endif

#include <iostream>
#include "cvconfig.h"
#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"

#if !defined(HAVE_CUDA) || !defined(HAVE_TBB)

int main()
{
#if !defined(HAVE_CUDA)
    std::cout << "CUDA support is required (CMake key 'WITH_CUDA' must be true).\n";
#endif

#if !defined(HAVE_TBB)
    std::cout << "TBB support is required (CMake key 'WITH_TBB' must be true).\n";
#endif

    return 0;
}

#else

#include <cuda.h>
#include <cuda_runtime.h>
#include "opencv2/core/internal.hpp" // For TBB wrappers

using namespace std;
using namespace cv;
using namespace cv::gpu;

struct Worker { void operator()(int device_id) const; };
void destroyContexts();

#define safeCall(expr) safeCall_(expr, #expr, __FILE__, __LINE__)
inline void safeCall_(int code, const char* expr, const char* file, int line)
{
    if (code != CUDA_SUCCESS)
    {
        std::cout << "CUDA driver API error: code " << code << ", expr " << expr
            << ", file " << file << ", line " << line << endl;
        destroyContexts();
        exit(-1);
    }
}

// Each GPU is associated with its own context
CUcontext contexts[2];

int main(int argc, char **argv)
{
    if (argc > 1)
    {
        cout << "CUDA driver API sample\n";
        return -1;
    }

    int num_devices = getCudaEnabledDeviceCount();
    if (num_devices < 2)
    {
        std::cout << "Two or more GPUs are required\n";
        return -1;
    }

    for (int i = 0; i < num_devices; ++i)
    {
        cv::gpu::printShortCudaDeviceInfo(i);

        DeviceInfo dev_info(i);
        if (!dev_info.isCompatible())
        {
            std::cout << "GPU module isn't built for GPU #" << i << " ("
                 << dev_info.name() << ", CC " << dev_info.majorVersion()
                 << dev_info.minorVersion() << "\n";
            return -1;
        }
    }

    // Init CUDA Driver API
    safeCall(cuInit(0));

    // Create context for GPU #0
    CUdevice device;
    safeCall(cuDeviceGet(&device, 0));
    safeCall(cuCtxCreate(&contexts[0], 0, device));

    CUcontext prev_context;
    safeCall(cuCtxPopCurrent(&prev_context));

    // Create context for GPU #1
    safeCall(cuDeviceGet(&device, 1));
    safeCall(cuCtxCreate(&contexts[1], 0, device));

    safeCall(cuCtxPopCurrent(&prev_context));

    // Execute calculation in two threads using two GPUs
    int devices[] = {0, 1};
    parallel_do(devices, devices + 2, Worker());

    destroyContexts();
    return 0;
}


void Worker::operator()(int device_id) const
{
    // Set the proper context
    safeCall(cuCtxPushCurrent(contexts[device_id]));

    Mat src(1000, 1000, CV_32F);
    Mat dst;

    RNG rng(0);
    rng.fill(src, RNG::UNIFORM, 0, 1);

    // CPU works
    transpose(src, dst);

    // GPU works
    GpuMat d_src(src);
    GpuMat d_dst;
    transpose(d_src, d_dst);

    // Check results
    bool passed = norm(dst - Mat(d_dst), NORM_INF) < 1e-3;
    std::cout << "GPU #" << device_id << " (" << DeviceInfo().name() << "): "
        << (passed ? "passed" : "FAILED") << endl;

    // Deallocate data here, otherwise deallocation will be performed
    // after context is extracted from the stack
    d_src.release();
    d_dst.release();

    CUcontext prev_context;
    safeCall(cuCtxPopCurrent(&prev_context));
}


void destroyContexts()
{
    safeCall(cuCtxDestroy(contexts[0]));
    safeCall(cuCtxDestroy(contexts[1]));
}

#endif
