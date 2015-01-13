/* This sample demonstrates the way you can perform independed tasks
   on the different GPUs */

// Disable some warnings which are caused with CUDA headers
#if defined(_MSC_VER)
#pragma warning(disable: 4201 4408 4100)
#endif

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"

#if defined(__arm__)
int main()
{
    std::cout << "Unsupported for ARM CUDA library." << std::endl;
    return 0;
}
#else

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

#define safeCall(expr) safeCall_(expr, #expr, __FILE__, __LINE__)
inline void safeCall_(int code, const char* expr, const char* file, int line)
{
    if (code != CUDA_SUCCESS)
    {
        std::cout << "CUDA driver API error: code " << code << ", expr " << expr
        << ", file " << file << ", line " << line << endl;
        exit(-1);
    }
}

struct Worker: public ParallelLoopBody
{
    Worker(int num_devices)
    {
        count = num_devices;
        contexts = new CUcontext[num_devices];
        for (int device_id = 0; device_id < num_devices; device_id++)
        {
            CUdevice device;
            safeCall(cuDeviceGet(&device, device_id));
            safeCall(cuCtxCreate(&contexts[device_id], 0, device));
        }
    }

    virtual void operator() (const Range& range) const
    {
        for (int device_id = range.start; device_id != range.end; ++device_id)
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
    }

    ~Worker()
    {
        if ((contexts != NULL) && count != 0)
        {
            for (int device_id = 0; device_id < count; device_id++)
            {
                safeCall(cuCtxDestroy(contexts[device_id]));
            }

            delete[] contexts;
        }
    }

    CUcontext* contexts;
    int count;
};

int main()
{
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

    // Execute calculation
    parallel_for_(cv::Range(0, num_devices), Worker(num_devices));

    return 0;
}

#endif
