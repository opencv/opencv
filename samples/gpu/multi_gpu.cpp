// Disable some warnings which are caused with CUDA headers
#pragma warning(disable: 4201 4408 4100)

#include <iostream>
#include <cvconfig.h>
#include "opencv2/core/core.hpp"
#include <opencv2/gpu/gpu.hpp>

#if !defined(HAVE_CUDA) || !defined(HAVE_TBB)

int main()
{
#if !defined(HAVE_CUDA)
    cout << "CUDA support is required (CMake key 'WITH_CUDA' must be true).\n";
#endif

#if !defined(HAVE_TBB)
    cout << "TBB support is required (CMake key 'WITH_TBB' must be true).\n";
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

#define cuSafeCall(code) if (code != CUDA_SUCCESS) { \
    cout << "CUDA driver API error: code " << code \
        << ", file " << __FILE__ << ", line " << __LINE__ << endl; \
    destroyContexts(); \
    exit(-1); \
}


// Each GPU is associated with its own context
CUcontext contexts[2];


int main()
{
    if (getCudaEnabledDeviceCount() < 2)
    {
        cout << "Two or more GPUs are required\n";
        return -1;
    }

    cuSafeCall(cuInit(0));

    // Create context for the first GPU
    CUdevice device;
    cuSafeCall(cuDeviceGet(&device, 0));
    cuSafeCall(cuCtxCreate(&contexts[0], 0, device));

    CUcontext prev_context;
    cuSafeCall(cuCtxPopCurrent(&prev_context));

    // Create context for the second GPU
    cuSafeCall(cuDeviceGet(&device, 1));
    cuSafeCall(cuCtxCreate(&contexts[1], 1, device));

    cuSafeCall(cuCtxPopCurrent(&prev_context));

    // Execute calculation in two threads using two GPUs
    int devices[] = {0, 1};
    parallel_do(devices, devices + 2, Worker());

    destroyContexts();
    return 0;
}


void Worker::operator()(int device_id) const
{
    // Set proper context
    cuSafeCall(cuCtxPushCurrent(contexts[device_id]));

    Mat src(1000, 1000, CV_32F);
    Mat dst;

    RNG rng(0);
    rng.fill(src, RNG::UNIFORM, 0, 1);

    // CPU works
    transpose(src, dst);

    GpuMat d_src(src);
    GpuMat d_dst;

    // GPU works
    transpose(d_src, d_dst);

    // Check results
    bool passed = norm(dst - Mat(d_dst), NORM_INF) < 1e-3;
    cout << "GPU #" << device_id << ": "<< (passed ? "passed" : "FAILED") << endl;

    // Deallocate data here, otherwise deallocation will be performed
    // after context is extracted from the stack
    d_src.release();
    d_dst.release();

    CUcontext prev_context;
    cuSafeCall(cuCtxPopCurrent(&prev_context));
}


void destroyContexts()
{
    cuSafeCall(cuCtxDestroy(contexts[0]));
    cuSafeCall(cuCtxDestroy(contexts[1]));
}

#endif