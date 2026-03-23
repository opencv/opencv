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
// Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
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
//     and / or other materials provided with the distribution.
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
#include "external_umat_allocator.hpp"

#ifdef HAVE_OPENCL
#include "opencv2/core/ocl.hpp"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

#include <string>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <dlfcn.h>
#endif

namespace cv {

struct ExternalDmaBufDesc
{
    int fd;
    size_t size;
};

#ifdef HAVE_OPENCL

#if defined(__linux__)

// Dynamically loaded OpenCL functions
struct OpenCLFunctions
{
    cl_int (*clGetContextInfo)(cl_context, cl_context_info, size_t, void*, size_t*);
    cl_int (*clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
    void* (*clGetExtensionFunctionAddressForPlatform)(cl_platform_id, const char*);
    cl_int (*clReleaseMemObject)(cl_mem);

    bool load()
    {
        // Try to open libOpenCL.so; fallback to RTLD_DEFAULT if already loaded
        void* handle = dlopen("libOpenCL.so", RTLD_LAZY | RTLD_GLOBAL);
        if (!handle)
            handle = RTLD_DEFAULT;

#define LOAD_FUNC(name) \
        name = (decltype(name))dlsym(handle, #name); \
        if (!name) return false

        LOAD_FUNC(clGetContextInfo);
        LOAD_FUNC(clGetDeviceInfo);
        LOAD_FUNC(clGetExtensionFunctionAddressForPlatform);
        LOAD_FUNC(clReleaseMemObject);

#undef LOAD_FUNC
        return true;
    }
};

static OpenCLFunctions& getOpenCL()
{
    static OpenCLFunctions funcs;
    static bool loaded = funcs.load();
    (void)loaded;  // ignore if load fails, we'll check function pointers later
    return funcs;
}

namespace {
    using PFN_clCreateBufferWithProperties = cl_mem (CL_API_CALL*)(
        cl_context context,
        const cl_mem_properties* properties,
        cl_mem_flags flags,
        size_t size,
        void* host_ptr,
        cl_int* errcode_ret);

    using PFN_clImportMemoryARM = cl_mem (CL_API_CALL*)(
        cl_context context,
        cl_mem_flags flags,
        const cl_import_properties_arm* properties,
        void* memory,
        size_t size,
        cl_int* errorcode_ret);

    static bool hasExt(const std::string& exts, const char* name)
    {
        return exts.find(name) != std::string::npos;
    }

    static std::string getDeviceExtensions(cl_device_id did)
    {
        size_t n = 0;
        getOpenCL().clGetDeviceInfo(did, CL_DEVICE_EXTENSIONS, 0, nullptr, &n);
        std::vector<char> buf(n + 1, '\0');
        getOpenCL().clGetDeviceInfo(did, CL_DEVICE_EXTENSIONS, n, buf.data(), nullptr);
        return std::string(buf.data());
    }

    static bool getDefaultDeviceAndPlatform(cl_context ctx,
                                            cl_device_id& device,
                                            cl_platform_id& platform)
    {
        size_t n = 0;
        if (getOpenCL().clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, nullptr, &n) != CL_SUCCESS || n == 0)
            return false;

        std::vector<cl_device_id> devices(n / sizeof(cl_device_id));
        if (getOpenCL().clGetContextInfo(ctx, CL_CONTEXT_DEVICES, n, devices.data(), nullptr) != CL_SUCCESS)
            return false;

        device = devices.front();
        if (getOpenCL().clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr) != CL_SUCCESS)
            return false;

        return true;
    }

    static PFN_clCreateBufferWithProperties loadCreateBufferWithProperties(cl_platform_id platform)
    {
        const char* names[] = { "clCreateBufferWithProperties", "clCreateBufferWithPropertiesKHR" };
        for (const char* name : names)
        {
            auto pfn = reinterpret_cast<PFN_clCreateBufferWithProperties>(
                getOpenCL().clGetExtensionFunctionAddressForPlatform(platform, name));
            if (pfn)
                return pfn;
        }
        return nullptr;
    }

    static PFN_clImportMemoryARM loadImportMemoryARM(cl_platform_id platform)
    {
        return reinterpret_cast<PFN_clImportMemoryARM>(
            getOpenCL().clGetExtensionFunctionAddressForPlatform(platform, "clImportMemoryARM"));
    }

    static cl_mem importViaKhrDmaBuf(cl_context ctx, cl_platform_id platform, int fd, size_t size)
    {
        auto pfnCreate = loadCreateBufferWithProperties(platform);
        if (!pfnCreate)
            return nullptr;

        const cl_mem_properties props[] = {
            CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR, (cl_mem_properties)fd,
            0
        };

        cl_int err = CL_SUCCESS;
        cl_mem mem = pfnCreate(ctx, props, CL_MEM_READ_WRITE, size, nullptr, &err);
        if (err != CL_SUCCESS)
            return nullptr;

        return mem;
    }

    static cl_mem importViaArmDmaBuf(cl_context ctx, cl_platform_id platform, int fd, size_t size)
    {
        auto pfnImport = loadImportMemoryARM(platform);
        if (!pfnImport)
            return nullptr;

        const cl_import_properties_arm props[] = {
            CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM,
            0
        };

        cl_int err = CL_SUCCESS;
        cl_mem mem = pfnImport(ctx, CL_MEM_READ_WRITE, props, &fd, size, &err);
        if (err != CL_SUCCESS)
            return nullptr;

        return mem;
    }
} // namespace
#endif // __linux__
#endif // HAVE_OPENCL

// Helper structure to hold imported OpenCL memory and associated dma-buf fd
struct ExternalMemHandle
{
    cl_mem mem;
    int fd;
    size_t size;
    void* mapped_ptr;   // pointer to mmap-ed region, if any
};

UMatData* ExternalUMatAllocator::allocate(int dims, const int* sizes, int type, void* data0,
                                          size_t* steps, AccessFlag /*accessFlags*/,
                                          UMatUsageFlags /*usageFlags*/) const
{
    CV_Assert(data0 != nullptr);

    CV_UNUSED(dims);
    CV_UNUSED(sizes);
    CV_UNUSED(type);
    CV_UNUSED(steps);

    auto* desc = static_cast<ExternalDmaBufDesc*>(data0);
    CV_Assert(desc->fd >= 0);
    CV_Assert(desc->size > 0);

    UMatData* u = new UMatData(this);
    u->data = u->origdata = nullptr;
    u->size = desc->size;
    u->currAllocator = const_cast<ExternalUMatAllocator*>(this);
    u->urefcount = 0;
    u->refcount = 0;
    u->mapcount = 0;
    u->handle = NULL;
    u->flags = UMatData::COPY_ON_MAP;

#if defined(HAVE_OPENCL) && defined(__linux__)
    if (ocl::useOpenCL())
    {
        OpenCLFunctions& cl = getOpenCL();
        if (cl.clGetContextInfo) // OpenCL functions available
        {
            ocl::Context oclCtx = ocl::Context::getDefault();
            cl_context ctx = (cl_context)oclCtx.ptr();

            if (ctx)
            {
                cl_device_id dev = nullptr;
                cl_platform_id platform = nullptr;

                if (getDefaultDeviceAndPlatform(ctx, dev, platform))
                {
                    const std::string devExts = getDeviceExtensions(dev);

                    cl_mem mem = nullptr;
                    // Prefer KHR extension first
                    if (hasExt(devExts, "cl_khr_external_memory_dma_buf"))
                    {
                        mem = importViaKhrDmaBuf(ctx, platform, desc->fd, desc->size);
                    }
                    // Fallback to ARM extension if KHR not available or failed
                    if (!mem && hasExt(devExts, "cl_arm_import_memory_dma_buf"))
                    {
                        mem = importViaArmDmaBuf(ctx, platform, desc->fd, desc->size);
                    }

                    if (mem)
                    {
                        ExternalMemHandle* handle = new ExternalMemHandle;
                        handle->mem = mem;
                        handle->fd = desc->fd;
                        handle->size = desc->size;
                        handle->mapped_ptr = nullptr;
                        u->handle = (void*)handle;
                        return u;
                    }
                }
            }
        }
    }
#endif

    // Fallback: allocate regular CPU memory
    u->data = u->origdata = (uchar*)cv::fastMalloc(desc->size);
    u->flags = UMatData::HOST_COPY_OBSOLETE;
    return u;
}

bool ExternalUMatAllocator::allocate(UMatData* u, AccessFlag /*accessFlags*/,
                                     UMatUsageFlags /*usageFlags*/) const
{
    u->currAllocator = const_cast<ExternalUMatAllocator*>(this);
    u->urefcount = 0;
    u->refcount = 0;
    u->mapcount = 0;
    u->flags = UMatData::COPY_ON_MAP;
    u->handle = NULL;
    return true;
}

void ExternalUMatAllocator::deallocate(UMatData* u) const
{
    if (!u) return;

#if defined(HAVE_OPENCL) && defined(__linux__)
    if (u->handle)
    {
        ExternalMemHandle* handle = (ExternalMemHandle*)u->handle;
        OpenCLFunctions& cl = getOpenCL();
        if (handle->mem && cl.clReleaseMemObject)
            cl.clReleaseMemObject(handle->mem);
        if (handle->mapped_ptr)
            munmap(handle->mapped_ptr, handle->size);
        delete handle;
        u->handle = NULL;
    }
#endif

    if (u->data && (u->flags & UMatData::HOST_COPY_OBSOLETE))
    {
        cv::fastFree(u->data);
        u->data = u->origdata = nullptr;
    }

    delete u;
}

void ExternalUMatAllocator::map(UMatData* u, AccessFlag /*accessFlags*/) const
{
#if defined(HAVE_OPENCL) && defined(__linux__)
    if (!u->handle)
        return;
    if (u->data)
        return; // already mapped

    ExternalMemHandle* handle = (ExternalMemHandle*)u->handle;
    if (handle->fd >= 0)
    {
        void* ptr = mmap(NULL, handle->size, PROT_READ | PROT_WRITE, MAP_SHARED, handle->fd, 0);
        if (ptr != MAP_FAILED)
        {
            handle->mapped_ptr = ptr;
            u->data = u->origdata = (uchar*)ptr;
        }
        else
        {
            CV_Error(Error::StsError, "Failed to mmap dma-buf");
        }
    }
#else
    CV_UNUSED(u);
#endif
}

void ExternalUMatAllocator::unmap(UMatData* u) const
{
#if defined(HAVE_OPENCL) && defined(__linux__)
    if (!u->handle)
        return;
    if (!u->data)
        return;

    ExternalMemHandle* handle = (ExternalMemHandle*)u->handle;
    if (handle->mapped_ptr)
    {
        munmap(handle->mapped_ptr, handle->size);
        handle->mapped_ptr = nullptr;
        u->data = u->origdata = nullptr;
    }
#else
    CV_UNUSED(u);
#endif
}

void ExternalUMatAllocator::copy(UMatData* srcU, UMatData* dstU, int dims,
                                 const size_t sz[], const size_t srcOfs[], const size_t srcStep[],
                                 const size_t dstOfs[], const size_t dstStep[], bool sync) const
{
    if (srcU->currAllocator != this || dstU->currAllocator != this)
    {
        CV_Error(Error::StsNotImplemented,
                 "Copying between external UMat and non-external UMat is not allowed");
    }
    MatAllocator::copy(srcU, dstU, dims, sz, srcOfs, srcStep, dstOfs, dstStep, sync);
}

static ExternalUMatAllocator g_external_allocator;

MatAllocator* getExternalUMatAllocator()
{
    return &g_external_allocator;
}

} // namespace cv