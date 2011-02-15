/* This sample demonstrates the way you can perform independed tasks 
   on the different GPUs */

// Disable some warnings which are caused with CUDA headers
#if defined(_MSC_VER)
#pragma warning(disable: 4201 4408 4100)
#endif

#include <iostream>
#include <cvconfig.h>
#include <opencv2/core/core.hpp>
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

#include "opencv2/core/internal.hpp" // For TBB wrappers

using namespace std;
using namespace cv;
using namespace cv::gpu;

struct Worker { void operator()(int device_id) const; };

MultiGpuManager multi_gpu_mgr;

int main()
{
    int num_devices = getCudaEnabledDeviceCount();
    if (num_devices < 2)
    {
        cout << "Two or more GPUs are required\n";
        return -1;
    }
    for (int i = 0; i < num_devices; ++i)
    {
        DeviceInfo dev_info(i);
        if (!dev_info.isCompatible())
        {
            cout << "GPU module isn't built for GPU #" << i << " ("
                 << dev_info.name() << ", CC " << dev_info.majorVersion()
                 << dev_info.minorVersion() << "\n";
            return -1;
        }
    }

    multi_gpu_mgr.init();

    // Execute calculation in two threads using two GPUs
    int devices[] = {0, 2};
    parallel_do(devices, devices + 2, Worker());

    return 0;
}


void Worker::operator()(int device_id) const
{
    multi_gpu_mgr.gpuOn(device_id);

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
    cout << "GPU #" << device_id << " (" << DeviceInfo().name() << "): "
        << (passed ? "passed" : "FAILED") << endl;

    // Deallocate data here, otherwise deallocation will be performed
    // after context is extracted from the stack
    d_src.release();
    d_dst.release();

    multi_gpu_mgr.gpuOff();
}

#endif
