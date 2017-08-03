/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (c) 2016-2017 Fabian David Tschopp, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../../precomp.hpp"
#include "benchmark.hpp"
#include "common.hpp"
#include "opencl_kernels_dnn.hpp"

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
