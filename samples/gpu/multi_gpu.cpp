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


void cuSafeCall(int code);
struct Worker { void operator()(int device_id) const; };
void destroy();


// Each GPU is associated with its own context
CUcontext contexts[2];

// Auxiliary variable, stores previusly used context
CUcontext prev_context;


int main()
{
    if (getCudaEnabledDeviceCount() < 2)
    {
        cout << "Two or more GPUs are required\n";
        return -1;
    }

    // Save the default context
    cuSafeCall(cuCtxAttach(&contexts[0], 0));
    cuSafeCall(cuCtxDetach(contexts[0]));

    // Create new context for the second GPU
    CUdevice device;
    cuSafeCall(cuDeviceGet(&device, 1));
    cuSafeCall(cuCtxCreate(&contexts[1], 0, device));

    // Restore the first GPU context
    cuSafeCall(cuCtxPopCurrent(&prev_context));

    // Run 
    int devices[] = {0, 1};
    parallel_do(devices, devices + 2, Worker());

    // Destroy context of the second GPU
    destroy();

    return 0;
}


void Worker::operator()(int device_id) const
{
    cout << device_id << endl;
}


void cuSafeCall(int code)
{
    if (code != CUDA_SUCCESS) 
    {
        cout << "CUDA driver API error: code " << code 
            << ", file " << __FILE__ 
            << ", line " << __LINE__ << endl;
        destroy();
        exit(-1);
    }
}


void destroy() 
{
    cuCtxDestroy(contexts[1]);
}

#endif