/* This sample demonstrates the way you can perform independed tasks
   on the different GPUs */

// Disable some warnings which are caused with CUDA headers
#if defined(_MSC_VER)
#pragma warning(disable: 4201 4408 4100)
#endif

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

struct Worker: public ParallelLoopBody
{
    virtual void operator() (const Range& range) const
    {
        for (int device_id = range.start; device_id != range.end; ++device_id)
        {
            setDevice(device_id);

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
        }
    }
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

    // Execute calculation in several threads, 1 GPU per thread
    parallel_for_(cv::Range(0, num_devices), Worker());

    return 0;
}
