/* This sample demonstrates working on one piece of data using two GPUs.
   It splits input into two parts and processes them separately on different
   GPUs. */

// Disable some warnings which are caused with CUDA headers
#if defined(_MSC_VER)
#pragma warning(disable: 4201 4408 4100)
#endif

#include <iostream>
#include <cvconfig.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

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

#include "opencv2/core/internal.hpp" // For TBB wrappers

using namespace std;
using namespace cv;
using namespace cv::gpu;

struct Worker { void operator()(int device_id) const; };

MultiGpuManager multi_gpu_mgr;

// GPUs data
GpuMat d_left[2];
GpuMat d_right[2];
StereoBM_GPU* bm[2];
GpuMat d_result[2];

// CPU result
Mat result;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: stereo_multi_gpu <left_image> <right_image>\n";
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
        DeviceInfo dev_info(i);
        if (!dev_info.isCompatible())
        {
            std::cout << "GPU module isn't built for GPU #" << i << " ("
                 << dev_info.name() << ", CC " << dev_info.majorVersion()
                 << dev_info.minorVersion() << "\n";
            return -1;
        }
    }

    // Load input data
    Mat left = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat right = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if (left.empty())
    {
        std::cout << "Cannot open '" << argv[1] << "'\n";
        return -1;
    }
    if (right.empty())
    {
        std::cout << "Cannot open '" << argv[2] << "'\n";
        return -1;
    }

    multi_gpu_mgr.init();

    // Split source images for processing on the GPU #0
    multi_gpu_mgr.gpuOn(0);
    d_left[0].upload(left.rowRange(0, left.rows / 2));
    d_right[0].upload(right.rowRange(0, right.rows / 2));
    bm[0] = new StereoBM_GPU();
    multi_gpu_mgr.gpuOff();

    // Split source images for processing on the GPU #1
    multi_gpu_mgr.gpuOn(1);
    d_left[1].upload(left.rowRange(left.rows / 2, left.rows));
    d_right[1].upload(right.rowRange(right.rows / 2, right.rows));
    bm[1] = new StereoBM_GPU();
    multi_gpu_mgr.gpuOff();

    // Execute calculation in two threads using two GPUs
    int devices[] = {0, 1};
    parallel_do(devices, devices + 2, Worker());

    // Release the first GPU resources
    multi_gpu_mgr.gpuOn(0);
    imshow("GPU #0 result", Mat(d_result[0]));
    d_left[0].release();
    d_right[0].release();
    d_result[0].release();
    delete bm[0];
    multi_gpu_mgr.gpuOff();

    // Release the second GPU resources
    multi_gpu_mgr.gpuOn(1);
    imshow("GPU #1 result", Mat(d_result[1]));
    d_left[1].release();
    d_right[1].release();
    d_result[1].release();
    delete bm[1];
    multi_gpu_mgr.gpuOff();

    waitKey();
    return 0;
}


void Worker::operator()(int device_id) const
{
    multi_gpu_mgr.gpuOn(device_id);

    bm[device_id]->operator()(d_left[device_id], d_right[device_id],
                              d_result[device_id]);

    std::cout << "GPU #" << device_id << " (" << DeviceInfo().name()
        << "): finished\n";

    multi_gpu_mgr.gpuOff();
}

#endif
