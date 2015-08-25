// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2015, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

#ifdef HAVE_VAAPI
#else // HAVE_VAAPI
#  define NO_VAAPI_SUPPORT_ERROR CV_ErrorNoReturn(cv::Error::StsBadFunc, "OpenCV was build without VA-API support")
#endif // HAVE_VAAPI

using namespace cv;

////////////////////////////////////////////////////////////////////////
// CL-VA Interoperability

#ifdef HAVE_OPENCL
#  include "opencv2/core/opencl/runtime/opencl_core.hpp"
#  include "opencv2/core.hpp"
#  include "opencv2/core/ocl.hpp"
#  include "opencl_kernels_core.hpp"
#else // HAVE_OPENCL
#  define NO_OPENCL_SUPPORT_ERROR CV_ErrorNoReturn(cv::Error::StsBadFunc, "OpenCV was build without OpenCL support")
#endif // HAVE_OPENCL

#if defined(HAVE_VAAPI) && defined(HAVE_OPENCL)
#  include <CL/va_ext.h>
#endif // HAVE_VAAPI && HAVE_OPENCL

namespace cv { namespace vaapi {

#if defined(HAVE_VAAPI) && defined(HAVE_OPENCL)

static clGetDeviceIDsFromVA_APIMediaAdapterINTEL_fn clGetDeviceIDsFromVA_APIMediaAdapterINTEL = NULL;
static clCreateFromVA_APIMediaSurfaceINTEL_fn       clCreateFromVA_APIMediaSurfaceINTEL       = NULL;
static clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn  clEnqueueAcquireVA_APIMediaSurfacesINTEL  = NULL;
static clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn  clEnqueueReleaseVA_APIMediaSurfacesINTEL  = NULL;

static bool contextInitialized = false;

#endif // HAVE_VAAPI && HAVE_OPENCL

namespace ocl {

Context& initializeContextFromVA(VADisplay display)
{
    (void)display;
#if !defined(HAVE_VAAPI)
    NO_VAAPI_SUPPORT_ERROR;
#elif !defined(HAVE_OPENCL)
    NO_OPENCL_SUPPORT_ERROR;
#else
    contextInitialized = false;

    cl_uint numPlatforms;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get number of platforms");
    if (numPlatforms == 0)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: No available platforms");

    std::vector<cl_platform_id> platforms(numPlatforms);
    status = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get platform Id list");

    // For CL-VA interop, we must find platform/device with "cl_intel_va_api_media_sharing" extension.
    // With standard initialization procedure, we should examine platform extension string for that.
    // But in practice, the platform ext string doesn't contain it, while device ext string does.
    // Follow Intel procedure (see tutorial), we should obtain device IDs by extension call.
    // Note that we must obtain function pointers using specific platform ID, and can't provide pointers in advance.
    // So, we iterate and select the first platform, for which we got non-NULL pointers, device, and CL context.

    int found = -1;
    cl_context context = 0;
    cl_device_id device = 0;

    for (int i = 0; i < (int)numPlatforms; ++i)
    {
        // Get extension function pointers

        clGetDeviceIDsFromVA_APIMediaAdapterINTEL = (clGetDeviceIDsFromVA_APIMediaAdapterINTEL_fn)
            clGetExtensionFunctionAddressForPlatform(platforms[i], "clGetDeviceIDsFromVA_APIMediaAdapterINTEL");
        clCreateFromVA_APIMediaSurfaceINTEL       = (clCreateFromVA_APIMediaSurfaceINTEL_fn)
            clGetExtensionFunctionAddressForPlatform(platforms[i], "clCreateFromVA_APIMediaSurfaceINTEL");
        clEnqueueAcquireVA_APIMediaSurfacesINTEL  = (clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn)
            clGetExtensionFunctionAddressForPlatform(platforms[i], "clEnqueueAcquireVA_APIMediaSurfacesINTEL");
        clEnqueueReleaseVA_APIMediaSurfacesINTEL  = (clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn)
            clGetExtensionFunctionAddressForPlatform(platforms[i], "clEnqueueReleaseVA_APIMediaSurfacesINTEL");

        if (((void*)clGetDeviceIDsFromVA_APIMediaAdapterINTEL == NULL) ||
            ((void*)clCreateFromVA_APIMediaSurfaceINTEL == NULL) ||
            ((void*)clEnqueueAcquireVA_APIMediaSurfacesINTEL == NULL) ||
            ((void*)clEnqueueReleaseVA_APIMediaSurfacesINTEL == NULL))
        {
            continue;
        }

        // Query device list

        cl_uint numDevices = 0;

        status = clGetDeviceIDsFromVA_APIMediaAdapterINTEL(platforms[i], CL_VA_API_DISPLAY_INTEL, display,
                                                           CL_PREFERRED_DEVICES_FOR_VA_API_INTEL, 0, NULL, &numDevices);
        if ((status != CL_SUCCESS) || !(numDevices > 0))
            continue;
        numDevices = 1; // initializeContextFromHandle() expects only 1 device
        status = clGetDeviceIDsFromVA_APIMediaAdapterINTEL(platforms[i], CL_VA_API_DISPLAY_INTEL, display,
                                                           CL_PREFERRED_DEVICES_FOR_VA_API_INTEL, numDevices, &device, NULL);
        if (status != CL_SUCCESS)
            continue;

        // Creating CL-VA media sharing OpenCL context

        cl_context_properties props[] = {
            CL_CONTEXT_VA_API_DISPLAY_INTEL, (cl_context_properties) display,
            CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE, // no explicit sync required
            0
        };

        context = clCreateContext(props, numDevices, &device, NULL, NULL, &status);
        if (status != CL_SUCCESS)
        {
            clReleaseDevice(device);
        }
        else
        {
            found = i;
            break;
        }
    }

    if (found < 0)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't create context for VA-API interop");

    Context& ctx = Context::getDefault(false);
    initializeContextFromHandle(ctx, platforms[found], context, device);
    contextInitialized = true;
    return ctx;
#endif
}

#if defined(HAVE_VAAPI) && defined(HAVE_OPENCL)
static bool ocl_convert_nv12_to_bgr(cl_mem clImageY, cl_mem clImageUV, cl_mem clBuffer, int step, int cols, int rows)
{
    ocl::Kernel k;
    k.create("YUV2BGR_NV12_8u", cv::ocl::core::cvtclr_dx_oclsrc, "");
    if (k.empty())
        return false;

    k.args(clImageY, clImageUV, clBuffer, step, cols, rows);

    size_t globalsize[] = { cols, rows };
    return k.run(2, globalsize, 0, false);
}

static bool ocl_convert_bgr_to_nv12(cl_mem clBuffer, int step, int cols, int rows, cl_mem clImageY, cl_mem clImageUV)
{
    ocl::Kernel k;
    k.create("BGR2YUV_NV12_8u", cv::ocl::core::cvtclr_dx_oclsrc, "");
    if (k.empty())
        return false;

    k.args(clBuffer, step, cols, rows, clImageY, clImageUV);

    size_t globalsize[] = { cols, rows };
    return k.run(2, globalsize, 0, false);
}
#endif // HAVE_VAAPI && HAVE_OPENCL

} // namespace cv::vaapi::ocl

void convertToVASurface(InputArray src, VASurfaceID surface, Size size)
{
    (void)src; (void)surface; (void)size;
#if !defined(HAVE_VAAPI)
    NO_VAAPI_SUPPORT_ERROR;
#elif !defined(HAVE_OPENCL)
    NO_OPENCL_SUPPORT_ERROR;
#else
    if (!contextInitialized)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Context for VA-API interop hasn't been created");

    const int stype = CV_8UC4;

    int srcType = src.type();
    CV_Assert(srcType == stype);

    Size srcSize = src.size();
    CV_Assert(srcSize.width == size.width && srcSize.height == size.height);

    UMat u = src.getUMat();

    // TODO Add support for roi
    CV_Assert(u.offset == 0);
    CV_Assert(u.isContinuous());

    cl_mem clBuffer = (cl_mem)u.handle(ACCESS_READ);

    using namespace cv::ocl;
    Context& ctx = Context::getDefault();
    cl_context context = (cl_context)ctx.ptr();

    cl_int status = 0;

    cl_mem clImageY = clCreateFromVA_APIMediaSurfaceINTEL(context, CL_MEM_WRITE_ONLY, &surface, 0, &status);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromVA_APIMediaSurfaceINTEL failed (Y plane)");
    cl_mem clImageUV = clCreateFromVA_APIMediaSurfaceINTEL(context, CL_MEM_WRITE_ONLY, &surface, 1, &status);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromVA_APIMediaSurfaceINTEL failed (UV plane)");

    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

    cl_mem images[2] = { clImageY, clImageUV };
    status = clEnqueueAcquireVA_APIMediaSurfacesINTEL(q, 2, images, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireVA_APIMediaSurfacesINTEL failed");
    if (!ocl::ocl_convert_bgr_to_nv12(clBuffer, (int)u.step[0], u.cols, u.rows, clImageY, clImageUV))
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: ocl_convert_bgr_to_nv12 failed");
    clEnqueueReleaseVA_APIMediaSurfacesINTEL(q, 2, images, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseVA_APIMediaSurfacesINTEL failed");

    status = clFinish(q); // TODO Use events
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clFinish failed");

    status = clReleaseMemObject(clImageY); // TODO RAII
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed (Y plane)");
    status = clReleaseMemObject(clImageUV);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed (UV plane)");
#endif
}

void convertFromVASurface(VASurfaceID surface, Size size, OutputArray dst)
{
    (void)surface; (void)dst; (void)size;
#if !defined(HAVE_VAAPI)
    NO_VAAPI_SUPPORT_ERROR;
#elif !defined(HAVE_OPENCL)
    NO_OPENCL_SUPPORT_ERROR;
#else
    if (!contextInitialized)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Context for VA-API interop hasn't been created");

    const int dtype = CV_8UC4;

    // TODO Need to specify ACCESS_WRITE here somehow to prevent useless data copying!
    dst.create(size, dtype);
    UMat u = dst.getUMat();

    // TODO Add support for roi
    CV_Assert(u.offset == 0);
    CV_Assert(u.isContinuous());

    cl_mem clBuffer = (cl_mem)u.handle(ACCESS_WRITE);

    using namespace cv::ocl;
    Context& ctx = Context::getDefault();
    cl_context context = (cl_context)ctx.ptr();

    cl_int status = 0;

    cl_mem clImageY = clCreateFromVA_APIMediaSurfaceINTEL(context, CL_MEM_READ_ONLY, &surface, 0, &status);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromVA_APIMediaSurfaceINTEL failed (Y plane)");
    cl_mem clImageUV = clCreateFromVA_APIMediaSurfaceINTEL(context, CL_MEM_READ_ONLY, &surface, 1, &status);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromVA_APIMediaSurfaceINTEL failed (UV plane)");

    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

    cl_mem images[2] = { clImageY, clImageUV };
    status = clEnqueueAcquireVA_APIMediaSurfacesINTEL(q, 2, images, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireVA_APIMediaSurfacesINTEL failed");
    if (!ocl::ocl_convert_nv12_to_bgr(clImageY, clImageUV, clBuffer, (int)u.step[0], u.cols, u.rows))
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: ocl_convert_nv12_to_bgr failed");
    status = clEnqueueReleaseVA_APIMediaSurfacesINTEL(q, 2, images, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseVA_APIMediaSurfacesINTEL failed");

    status = clFinish(q); // TODO Use events
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clFinish failed");

    status = clReleaseMemObject(clImageY); // TODO RAII
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed (Y plane)");
    status = clReleaseMemObject(clImageUV);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed (UV plane)");
#endif
}

}} // namespace cv::vaapi
