#include "../../precomp.hpp"
#include "benchmark.hpp"
#include "common.hpp"
#include "opencl_kernels_dnn.hpp"

using namespace cv;
namespace greentea
{

#ifdef HAVE_OPENCL
Timer::Timer()
    : initted_(false)
    , running_(false)
    , has_run_at_least_once_(false)
{
    Init();
}

Timer::~Timer()
{
    clWaitForEvents(1, &start_gpu_cl_);
    clWaitForEvents(1, &stop_gpu_cl_);
    clReleaseEvent(start_gpu_cl_);
    clReleaseEvent(stop_gpu_cl_);
}

void Timer::Start()
{
    if (!running())
    {
        clWaitForEvents(1, &start_gpu_cl_);
        clReleaseEvent(start_gpu_cl_);
        ocl::Queue queue = ocl::Queue::getDefault();
        ocl::Kernel kernel("null_kernel_float", cv::ocl::dnn::benchmark_oclsrc);
        // TODO(naibaf7): compiler shows deprecated declaration
        // use `clEnqueueNDRangeKernel` instead
        // https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clEnqueueTask.html
        float arg = 0;
        clSetKernelArg((cl_kernel)kernel.ptr(), 0, sizeof(arg), &arg);
        clEnqueueTask((cl_command_queue)queue.ptr(), (cl_kernel)kernel.ptr(), 0,
                      NULL, &start_gpu_cl_);
        clFinish((cl_command_queue)queue.ptr());
        running_ = true;
        has_run_at_least_once_ = true;
    }
}

void Timer::Stop()
{
    if (running())
    {
        clWaitForEvents(1, &stop_gpu_cl_);
        clReleaseEvent(stop_gpu_cl_);
        ocl::Queue queue = ocl::Queue::getDefault();
        ocl::Kernel kernel("null_kernel_float", cv::ocl::dnn::benchmark_oclsrc);
        float arg = 0;
        clSetKernelArg((cl_kernel)kernel.ptr(), 0, sizeof(arg), &arg);
        clEnqueueTask((cl_command_queue)queue.ptr(), (cl_kernel)kernel.ptr(), 0,
                      NULL, &stop_gpu_cl_);
        clFinish((cl_command_queue)queue.ptr());
        running_ = false;
    }
}

float Timer::MicroSeconds()
{
    if (!has_run_at_least_once())
    {
        return 0;
    }
    if (running())
    {
        Stop();
    }
    cl_ulong startTime, stopTime;
    clWaitForEvents(1, &stop_gpu_cl_);
    clGetEventProfilingInfo(start_gpu_cl_, CL_PROFILING_COMMAND_END,
                            sizeof startTime, &startTime, NULL);
    clGetEventProfilingInfo(stop_gpu_cl_, CL_PROFILING_COMMAND_START,
                            sizeof stopTime, &stopTime, NULL);
    double us = static_cast<double>(stopTime - startTime) / 1000.0;
    elapsed_microseconds_ = static_cast<float>(us);
    return elapsed_microseconds_;
}

float Timer::MilliSeconds()
{
    if (!has_run_at_least_once())
    {
        return 0;
    }
    if (running())
    {
        Stop();
    }
    cl_ulong startTime = 0, stopTime = 0;
    clGetEventProfilingInfo(start_gpu_cl_, CL_PROFILING_COMMAND_END,
                            sizeof startTime, &startTime, NULL);
    clGetEventProfilingInfo(stop_gpu_cl_, CL_PROFILING_COMMAND_START,
                            sizeof stopTime, &stopTime, NULL);
    double ms = static_cast<double>(stopTime - startTime) / 1000000.0;
    elapsed_milliseconds_ = static_cast<float>(ms);
    return elapsed_milliseconds_;
}

float Timer::Seconds()
{
    return MilliSeconds() / 1000.;
}

void Timer::Init()
{
    if (!initted())
    {
        start_gpu_cl_ = 0;
        stop_gpu_cl_ = 0;
        initted_ = true;
    }
}
#endif // HAVE_OPENCL

}  // namespace greentea
