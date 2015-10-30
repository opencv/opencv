#include "precomp.hpp"

#if defined(HAVE_OPENCL) && !defined(HAVE_OPENCL_STATIC)

#include "opencv2/ocl/cl_runtime/cl_runtime.hpp"

#if defined(__APPLE__)
    #include <dlfcn.h>

    static void* AppleCLGetProcAddress(const char* name)
    {
        static void * image = NULL;
        if (!image)
        {
            image = dlopen("/System/Library/Frameworks/OpenCL.framework/Versions/Current/OpenCL", RTLD_LAZY | RTLD_GLOBAL);
            if (!image)
                return NULL;
        }

        return dlsym(image, name);
    }
    #define CV_CL_GET_PROC_ADDRESS(name) AppleCLGetProcAddress(name)
#endif // __APPLE__

#if defined(_WIN32)
    static void* WinGetProcAddress(const char* name)
    {
        static HMODULE opencl_module = NULL;
        if (!opencl_module)
        {
            opencl_module = GetModuleHandleA("OpenCL.dll");
            if (!opencl_module)
            {
                const char* libName = "OpenCL.dll";
                const char* envOpenCLBinary = getenv("OPENCV_OPENCL_BINARY");
                if (envOpenCLBinary)
                    libName = envOpenCLBinary;
                opencl_module = LoadLibraryA(libName);
                if (!opencl_module)
                    return NULL;
            }
        }
        return (void*)GetProcAddress(opencl_module, name);
    }
    #define CV_CL_GET_PROC_ADDRESS(name) WinGetProcAddress(name)
#endif // _WIN32

#if defined(__linux__)
    #include <dlfcn.h>
    #include <stdio.h>

    static void* GetProcAddress (const char* name)
    {
        static void* h = NULL;
        if (!h)
        {
            const char* name = "libOpenCL.so";
            const char* envOpenCLBinary = getenv("OPENCV_OPENCL_BINARY");
            if (envOpenCLBinary)
                name = envOpenCLBinary;
            h = dlopen(name, RTLD_LAZY | RTLD_GLOBAL);
            if (!h)
                return NULL;
        }

        return dlsym(h, name);
    }
    #define CV_CL_GET_PROC_ADDRESS(name) GetProcAddress(name)
#endif

#ifndef CV_CL_GET_PROC_ADDRESS
#define CV_CL_GET_PROC_ADDRESS(name) NULL
#endif

static void* opencl_check_fn(int ID)
{
    extern const char* opencl_fn_names[];
    void* func = CV_CL_GET_PROC_ADDRESS(opencl_fn_names[ID]);
    if (!func)
    {
        std::ostringstream msg;
        msg << "OpenCL function is not available: [" << opencl_fn_names[ID] << "]";
        CV_Error(CV_StsBadFunc, msg.str());
    }
    extern void* opencl_fn_ptrs[];
    *(void**)(opencl_fn_ptrs[ID]) = func;
    return func;
}

#include "cl_runtime_opencl_impl.hpp"

#endif
