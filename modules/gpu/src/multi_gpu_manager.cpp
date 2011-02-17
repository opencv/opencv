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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"

#if !defined(HAVE_CUDA)

namespace cv { namespace gpu {

class MultiGpuManager::Impl {};
MultiGpuManager::MultiGpuManager() { throw_nogpu(); }
MultiGpuManager::~MultiGpuManager() {}
void MultiGpuManager::init() { throw_nogpu(); }
void MultiGpuManager::gpuOn(int) { throw_nogpu(); }
void MultiGpuManager::gpuOff() { throw_nogpu(); }

}}

#else

#include <vector>
#include <cuda.h>

#define cuSafeCall(expr) safeCall(expr, #expr, __FILE__, __LINE__)

using namespace std;

namespace cv { namespace gpu {


class MultiGpuManager::Impl
{
public:
    Impl();

    ~Impl()
    {
        for (int i = 0; i < num_devices_; ++i)
            cuSafeCall(cuCtxDestroy(contexts_[i]));
    }

    void gpuOn(int gpu_id)
    {
        if (gpu_id < 0 || gpu_id >= num_devices_)
            CV_Error(CV_StsBadArg, "MultiGpuManager::gpuOn: GPU ID is out of range");
        cuSafeCall(cuCtxPushCurrent(contexts_[gpu_id]));
    }

    void gpuOff()
    {
        CUcontext prev_context;
        cuSafeCall(cuCtxPopCurrent(&prev_context));
    }

private:
    void safeCall(CUresult code, const char* expr, const char* file, int line)
    {
        if (code != CUDA_SUCCESS)
            error(expr, file, line, "");
    }

    int num_devices_;
    vector<CUcontext> contexts_;
};


MultiGpuManager::Impl::Impl(): num_devices_(0)
{
    num_devices_ = getCudaEnabledDeviceCount();
    contexts_.resize(num_devices_);

    cuSafeCall(cuInit(0));

    CUdevice device;
    CUcontext prev_context;
    for (int i = 0; i < num_devices_; ++i)
    {
        cuSafeCall(cuDeviceGet(&device, i));
        cuSafeCall(cuCtxCreate(&contexts_[i], 0, device));
        cuSafeCall(cuCtxPopCurrent(&prev_context));
    }
}


MultiGpuManager::MultiGpuManager() {}
MultiGpuManager::~MultiGpuManager() {}


void MultiGpuManager::init()
{
    impl_ = Ptr<Impl>(new Impl());
}


void MultiGpuManager::gpuOn(int gpu_id)
{
    if (impl_.empty())
        CV_Error(CV_StsNullPtr, "MultiGpuManager::gpuOn: must be initialized before any calls");
    impl_->gpuOn(gpu_id);
}


void MultiGpuManager::gpuOff()
{
    if (impl_.empty())
        CV_Error(CV_StsNullPtr, "MultiGpuManager::gpuOff: must be initialized before any calls");
    impl_->gpuOff();
}

}}

#endif
