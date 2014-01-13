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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#if defined(HAVE_OPENCL) && !defined(HAVE_OPENCL_STATIC)

#include "opencv2/core.hpp" // CV_Error

#include "opencv2/core/opencl/runtime/opencl_core.hpp"

static const char* funcToCheckOpenCL1_1 = "clEnqueueReadBufferRect";
#define ERROR_MSG_CANT_LOAD "Failed to load OpenCL runtime\n"
#define ERROR_MSG_INVALID_VERSION "Failed to load OpenCL runtime (expected version 1.1+)\n"

#if defined(__APPLE__)
#include <dlfcn.h>

static void* AppleCLGetProcAddress(const char* name)
{
    static bool initialized = false;
    static void* handle = NULL;
    if (!handle)
    {
        if(!initialized)
        {
            initialized = true;
            const char* path = "/System/Library/Frameworks/OpenCL.framework/Versions/Current/OpenCL";
            const char* envPath = getenv("OPENCV_OPENCL_RUNTIME");
            if (envPath)
                path = envPath;
            handle = dlopen(oclpath, RTLD_LAZY | RTLD_GLOBAL);
            if (handle == NULL)
            {
                fprintf(stderr, ERROR_MSG_CANT_LOAD);
            }
            else if (dlsym(handle, funcToCheckOpenCL1_1) == NULL)
            {
                fprintf(stderr, ERROR_MSG_INVALID_VERSION);
                handle = NULL;
            }
        }
        if (!handle)
            return NULL;
    }
    return dlsym(handle, name);
}
#define CV_CL_GET_PROC_ADDRESS(name) AppleCLGetProcAddress(name)
#endif // __APPLE__

#if defined(_WIN32)
#include <windows.h>

static void* WinGetProcAddress(const char* name)
{
    static bool initialized = false;
    static HMODULE handle = NULL;
    if (!handle)
    {
        if(!initialized)
        {
            initialized = true;
            handle = GetModuleHandleA("OpenCL.dll");
            if (!handle)
            {
                const char* path = "OpenCL.dll";
                const char* envPath = getenv("OPENCV_OPENCL_RUNTIME");
                if (envPath)
                    path = envPath;
                handle = LoadLibraryA(path);
                if (!handle)
                {
                    fprintf(stderr, ERROR_MSG_CANT_LOAD);
                }
                else if (GetProcAddress(handle, funcToCheckOpenCL1_1) == NULL)
                {
                    fprintf(stderr, ERROR_MSG_INVALID_VERSION);
                    handle = NULL;
                }
            }
        }
        if (!handle)
            return NULL;
    }
    return (void*)GetProcAddress(handle, name);
}
#define CV_CL_GET_PROC_ADDRESS(name) WinGetProcAddress(name)
#endif // _WIN32

#if defined(linux)
#include <dlfcn.h>
#include <stdio.h>

static void* GetProcAddress(const char* name)
{
    static bool initialized = false;
    static void* handle = NULL;
    if (!handle)
    {
        if(!initialized)
        {
            initialized = true;
            const char* path = "libOpenCL.so";
            const char* envPath = getenv("OPENCV_OPENCL_RUNTIME");
            if (envPath)
                path = envPath;
            handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
            if (handle == NULL)
            {
                fprintf(stderr, ERROR_MSG_CANT_LOAD);
            }
            else if (dlsym(handle, funcToCheckOpenCL1_1) == NULL)
            {
                fprintf(stderr, ERROR_MSG_INVALID_VERSION);
                handle = NULL;
            }
        }
        if (!handle)
            return NULL;
    }
    return dlsym(handle, name);
}
#define CV_CL_GET_PROC_ADDRESS(name) GetProcAddress(name)
#endif

#ifndef CV_CL_GET_PROC_ADDRESS
#define CV_CL_GET_PROC_ADDRESS(name) NULL
#endif

static void* opencl_check_fn(int ID);

#include "runtime_common.hpp"

#include "autogenerated/opencl_core_impl.hpp"

//
// BEGIN OF CUSTOM FUNCTIONS
//

#define CUSTOM_FUNCTION_ID 1000

//
// END OF CUSTOM FUNCTIONS HERE
//

static void* opencl_check_fn(int ID)
{
    const struct DynamicFnEntry* e = NULL;
    if (ID < CUSTOM_FUNCTION_ID)
    {
        assert(ID >= 0 && ID < (int)(sizeof(opencl_fn_list)/sizeof(opencl_fn_list[0])));
        e = opencl_fn_list[ID];
    }
    else
    {
        CV_ErrorNoReturn(cv::Error::StsBadArg, "Invalid function ID");
    }
    void* func = CV_CL_GET_PROC_ADDRESS(e->fnName);
    if (!func)
    {
        CV_Error(cv::Error::OpenCLApiCallError, cv::format("OpenCL function is not available: [%s]", e->fnName));
    }
    *(e->ppFn) = func;
    return func;
}

#endif
