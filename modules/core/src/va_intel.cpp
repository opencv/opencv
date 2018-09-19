// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2015, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

#ifdef HAVE_VA
#  include <va/va.h>
#else  // HAVE_VA
#  define NO_VA_SUPPORT_ERROR CV_Error(cv::Error::StsBadFunc, "OpenCV was build without VA support (libva)")
#endif // HAVE_VA

using namespace cv;

////////////////////////////////////////////////////////////////////////
// CL-VA Interoperability

#ifdef HAVE_OPENCL
#  include "opencv2/core/opencl/runtime/opencl_core.hpp"
#  include "opencv2/core.hpp"
#  include "opencv2/core/ocl.hpp"
#  include "opencl_kernels_core.hpp"
#endif // HAVE_OPENCL

#if defined(HAVE_VA_INTEL) && defined(HAVE_OPENCL)
#  include <CL/va_ext.h>
#endif // HAVE_VA_INTEL && HAVE_OPENCL

namespace cv { namespace va_intel {

#if defined(HAVE_VA_INTEL) && defined(HAVE_OPENCL)

static clGetDeviceIDsFromVA_APIMediaAdapterINTEL_fn clGetDeviceIDsFromVA_APIMediaAdapterINTEL = NULL;
static clCreateFromVA_APIMediaSurfaceINTEL_fn       clCreateFromVA_APIMediaSurfaceINTEL       = NULL;
static clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn  clEnqueueAcquireVA_APIMediaSurfacesINTEL  = NULL;
static clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn  clEnqueueReleaseVA_APIMediaSurfacesINTEL  = NULL;

static bool contextInitialized = false;

#endif // HAVE_VA_INTEL && HAVE_OPENCL

namespace ocl {

Context& initializeContextFromVA(VADisplay display, bool tryInterop)
{
    CV_UNUSED(display); CV_UNUSED(tryInterop);
#if !defined(HAVE_VA)
    NO_VA_SUPPORT_ERROR;
#else  // !HAVE_VA
# if (defined(HAVE_VA_INTEL) && defined(HAVE_OPENCL))
    contextInitialized = false;
    if (tryInterop)
    {
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

        if (found >= 0)
        {
            contextInitialized = true;
            Context& ctx = Context::getDefault(false);
            initializeContextFromHandle(ctx, platforms[found], context, device);
            return ctx;
        }
    }
# endif // HAVE_VA_INTEL && HAVE_OPENCL
    {
        Context& ctx = Context::getDefault(true);
        return ctx;
    }
#endif  // !HAVE_VA
}

#if defined(HAVE_VA_INTEL) && defined(HAVE_OPENCL)
static bool ocl_convert_nv12_to_bgr(cl_mem clImageY, cl_mem clImageUV, cl_mem clBuffer, int step, int cols, int rows)
{
    ocl::Kernel k;
    k.create("YUV2BGR_NV12_8u", cv::ocl::core::cvtclr_dx_oclsrc, "");
    if (k.empty())
        return false;

    k.args(clImageY, clImageUV, clBuffer, step, cols, rows);

    size_t globalsize[] = { (size_t)cols, (size_t)rows };
    return k.run(2, globalsize, 0, false);
}

static bool ocl_convert_bgr_to_nv12(cl_mem clBuffer, int step, int cols, int rows, cl_mem clImageY, cl_mem clImageUV)
{
    ocl::Kernel k;
    k.create("BGR2YUV_NV12_8u", cv::ocl::core::cvtclr_dx_oclsrc, "");
    if (k.empty())
        return false;

    k.args(clBuffer, step, cols, rows, clImageY, clImageUV);

    size_t globalsize[] = { (size_t)cols, (size_t)rows };
    return k.run(2, globalsize, 0, false);
}
#endif // HAVE_VA_INTEL && HAVE_OPENCL

} // namespace cv::va_intel::ocl

#if defined(HAVE_VA)
const int NCHANNELS = 3;

static void copy_convert_nv12_to_bgr(const VAImage& image, const unsigned char* buffer, Mat& bgr)
{
    const float d1 = 16.0f;
    const float d2 = 128.0f;

    static const float coeffs[5] =
        {
            1.163999557f,
            2.017999649f,
            -0.390999794f,
            -0.812999725f,
            1.5959997177f
        };

    const size_t srcOffsetY = image.offsets[0];
    const size_t srcOffsetUV = image.offsets[1];

    const size_t srcStepY = image.pitches[0];
    const size_t srcStepUV = image.pitches[1];

    const size_t dstStep = bgr.step;

    const unsigned char* srcY0 = buffer + srcOffsetY;
    const unsigned char* srcUV = buffer + srcOffsetUV;

    unsigned char* dst0 = bgr.data;

    for (int y = 0; y < bgr.rows; y += 2)
    {
        const unsigned char* srcY1 = srcY0 + srcStepY;
        unsigned char *dst1 = dst0 + dstStep;

        for (int x = 0; x < bgr.cols; x += 2)
        {
            float Y0 = float(srcY0[x+0]);
            float Y1 = float(srcY0[x+1]);
            float Y2 = float(srcY1[x+0]);
            float Y3 = float(srcY1[x+1]);

            float U = float(srcUV[2*(x/2)+0]) - d2;
            float V = float(srcUV[2*(x/2)+1]) - d2;

            Y0 = std::max(0.0f, Y0 - d1) * coeffs[0];
            Y1 = std::max(0.0f, Y1 - d1) * coeffs[0];
            Y2 = std::max(0.0f, Y2 - d1) * coeffs[0];
            Y3 = std::max(0.0f, Y3 - d1) * coeffs[0];

            float ruv = coeffs[4]*V;
            float guv = coeffs[3]*V + coeffs[2]*U;
            float buv = coeffs[1]*U;

            dst0[(x+0)*NCHANNELS+0] = saturate_cast<unsigned char>(Y0 + buv);
            dst0[(x+0)*NCHANNELS+1] = saturate_cast<unsigned char>(Y0 + guv);
            dst0[(x+0)*NCHANNELS+2] = saturate_cast<unsigned char>(Y0 + ruv);

            dst0[(x+1)*NCHANNELS+0] = saturate_cast<unsigned char>(Y1 + buv);
            dst0[(x+1)*NCHANNELS+1] = saturate_cast<unsigned char>(Y1 + guv);
            dst0[(x+1)*NCHANNELS+2] = saturate_cast<unsigned char>(Y1 + ruv);

            dst1[(x+0)*NCHANNELS+0] = saturate_cast<unsigned char>(Y2 + buv);
            dst1[(x+0)*NCHANNELS+1] = saturate_cast<unsigned char>(Y2 + guv);
            dst1[(x+0)*NCHANNELS+2] = saturate_cast<unsigned char>(Y2 + ruv);

            dst1[(x+1)*NCHANNELS+0] = saturate_cast<unsigned char>(Y3 + buv);
            dst1[(x+1)*NCHANNELS+1] = saturate_cast<unsigned char>(Y3 + guv);
            dst1[(x+1)*NCHANNELS+2] = saturate_cast<unsigned char>(Y3 + ruv);
        }

        srcY0 = srcY1 + srcStepY;
        srcUV += srcStepUV;
        dst0 = dst1 + dstStep;
    }
}

static void copy_convert_bgr_to_nv12(const VAImage& image, const Mat& bgr, unsigned char* buffer)
{
    const float d1 = 16.0f;
    const float d2 = 128.0f;

    static const float coeffs[8] =
        {
            0.256999969f,  0.50399971f,   0.09799957f,   -0.1479988098f,
            -0.2909994125f, 0.438999176f, -0.3679990768f, -0.0709991455f
        };

    const size_t dstOffsetY = image.offsets[0];
    const size_t dstOffsetUV = image.offsets[1];

    const size_t dstStepY = image.pitches[0];
    const size_t dstStepUV = image.pitches[1];

    const size_t srcStep = bgr.step;

    const unsigned char* src0 = bgr.data;

    unsigned char* dstY0 = buffer + dstOffsetY;
    unsigned char* dstUV = buffer + dstOffsetUV;

    for (int y = 0; y < bgr.rows; y += 2)
    {
        const unsigned char *src1 = src0 + srcStep;
        unsigned char* dstY1 = dstY0 + dstStepY;

        for (int x = 0; x < bgr.cols; x += 2)
        {
            float B0 = float(src0[(x+0)*NCHANNELS+0]);
            float G0 = float(src0[(x+0)*NCHANNELS+1]);
            float R0 = float(src0[(x+0)*NCHANNELS+2]);

            float B1 = float(src0[(x+1)*NCHANNELS+0]);
            float G1 = float(src0[(x+1)*NCHANNELS+1]);
            float R1 = float(src0[(x+1)*NCHANNELS+2]);

            float B2 = float(src1[(x+0)*NCHANNELS+0]);
            float G2 = float(src1[(x+0)*NCHANNELS+1]);
            float R2 = float(src1[(x+0)*NCHANNELS+2]);

            float B3 = float(src1[(x+1)*NCHANNELS+0]);
            float G3 = float(src1[(x+1)*NCHANNELS+1]);
            float R3 = float(src1[(x+1)*NCHANNELS+2]);

            float Y0 = coeffs[0]*R0 + coeffs[1]*G0 + coeffs[2]*B0 + d1;
            float Y1 = coeffs[0]*R1 + coeffs[1]*G1 + coeffs[2]*B1 + d1;
            float Y2 = coeffs[0]*R2 + coeffs[1]*G2 + coeffs[2]*B2 + d1;
            float Y3 = coeffs[0]*R3 + coeffs[1]*G3 + coeffs[2]*B3 + d1;

            float U = coeffs[3]*R0 + coeffs[4]*G0 + coeffs[5]*B0 + d2;
            float V = coeffs[5]*R0 + coeffs[6]*G0 + coeffs[7]*B0 + d2;

            dstY0[x+0] = saturate_cast<unsigned char>(Y0);
            dstY0[x+1] = saturate_cast<unsigned char>(Y1);
            dstY1[x+0] = saturate_cast<unsigned char>(Y2);
            dstY1[x+1] = saturate_cast<unsigned char>(Y3);

            dstUV[2*(x/2)+0] = saturate_cast<unsigned char>(U);
            dstUV[2*(x/2)+1] = saturate_cast<unsigned char>(V);
        }

        src0 = src1 + srcStep;
        dstY0 = dstY1 + dstStepY;
        dstUV += dstStepUV;
    }
}


static void copy_convert_yv12_to_bgr(const VAImage& image, const unsigned char* buffer, Mat& bgr)
{
    const float d1 = 16.0f;
    const float d2 = 128.0f;

    static const float coeffs[5] =
        {
            1.163999557f,
            2.017999649f,
            -0.390999794f,
            -0.812999725f,
            1.5959997177f
        };

    CV_CheckEQ(image.format.fourcc, VA_FOURCC_YV12, "Unexpected image format");
    CV_CheckEQ(image.num_planes, 3, "");

    const size_t srcOffsetY = image.offsets[0];
    const size_t srcOffsetV = image.offsets[1];
    const size_t srcOffsetU = image.offsets[2];

    const size_t srcStepY = image.pitches[0];
    const size_t srcStepU = image.pitches[1];
    const size_t srcStepV = image.pitches[2];

    const size_t dstStep = bgr.step;

    const unsigned char* srcY_ = buffer + srcOffsetY;
    const unsigned char* srcV_ = buffer + srcOffsetV;
    const unsigned char* srcU_ = buffer + srcOffsetU;

    for (int y = 0; y < bgr.rows; y += 2)
    {
        const unsigned char* srcY0 = srcY_ + (srcStepY) * y;
        const unsigned char* srcY1 = srcY0 + srcStepY;

        const unsigned char* srcV = srcV_ + (srcStepV) * y / 2;
        const unsigned char* srcU = srcU_ + (srcStepU) * y / 2;

        unsigned char* dst0 = bgr.data + (dstStep) * y;
        unsigned char* dst1 = dst0 + dstStep;

        for (int x = 0; x < bgr.cols; x += 2)
        {
            float Y0 = float(srcY0[x+0]);
            float Y1 = float(srcY0[x+1]);
            float Y2 = float(srcY1[x+0]);
            float Y3 = float(srcY1[x+1]);

            float U = float(srcU[x/2]) - d2;
            float V = float(srcV[x/2]) - d2;

            Y0 = std::max(0.0f, Y0 - d1) * coeffs[0];
            Y1 = std::max(0.0f, Y1 - d1) * coeffs[0];
            Y2 = std::max(0.0f, Y2 - d1) * coeffs[0];
            Y3 = std::max(0.0f, Y3 - d1) * coeffs[0];

            float ruv = coeffs[4]*V;
            float guv = coeffs[3]*V + coeffs[2]*U;
            float buv = coeffs[1]*U;

            dst0[(x+0)*NCHANNELS+0] = saturate_cast<unsigned char>(Y0 + buv);
            dst0[(x+0)*NCHANNELS+1] = saturate_cast<unsigned char>(Y0 + guv);
            dst0[(x+0)*NCHANNELS+2] = saturate_cast<unsigned char>(Y0 + ruv);

            dst0[(x+1)*NCHANNELS+0] = saturate_cast<unsigned char>(Y1 + buv);
            dst0[(x+1)*NCHANNELS+1] = saturate_cast<unsigned char>(Y1 + guv);
            dst0[(x+1)*NCHANNELS+2] = saturate_cast<unsigned char>(Y1 + ruv);

            dst1[(x+0)*NCHANNELS+0] = saturate_cast<unsigned char>(Y2 + buv);
            dst1[(x+0)*NCHANNELS+1] = saturate_cast<unsigned char>(Y2 + guv);
            dst1[(x+0)*NCHANNELS+2] = saturate_cast<unsigned char>(Y2 + ruv);

            dst1[(x+1)*NCHANNELS+0] = saturate_cast<unsigned char>(Y3 + buv);
            dst1[(x+1)*NCHANNELS+1] = saturate_cast<unsigned char>(Y3 + guv);
            dst1[(x+1)*NCHANNELS+2] = saturate_cast<unsigned char>(Y3 + ruv);
        }
    }
}

static void copy_convert_bgr_to_yv12(const VAImage& image, const Mat& bgr, unsigned char* buffer)
{
    const float d1 = 16.0f;
    const float d2 = 128.0f;

    static const float coeffs[8] =
        {
            0.256999969f,  0.50399971f,   0.09799957f,   -0.1479988098f,
            -0.2909994125f, 0.438999176f, -0.3679990768f, -0.0709991455f
        };

    CV_CheckEQ(image.format.fourcc, VA_FOURCC_YV12, "Unexpected image format");
    CV_CheckEQ(image.num_planes, 3, "");

    const size_t dstOffsetY = image.offsets[0];
    const size_t dstOffsetV = image.offsets[1];
    const size_t dstOffsetU = image.offsets[2];

    const size_t dstStepY = image.pitches[0];
    const size_t dstStepU = image.pitches[1];
    const size_t dstStepV = image.pitches[2];

    unsigned char* dstY_ = buffer + dstOffsetY;
    unsigned char* dstV_ = buffer + dstOffsetV;
    unsigned char* dstU_ = buffer + dstOffsetU;

    const size_t srcStep = bgr.step;

    for (int y = 0; y < bgr.rows; y += 2)
    {
        unsigned char* dstY0 = dstY_ + (dstStepY) * y;
        unsigned char* dstY1 = dstY0 + dstStepY;

        unsigned char* dstV = dstV_ + (dstStepV) * y / 2;
        unsigned char* dstU = dstU_ + (dstStepU) * y / 2;

        const unsigned char* src0 = bgr.data + (srcStep) * y;
        const unsigned char* src1 = src0 + srcStep;

        for (int x = 0; x < bgr.cols; x += 2)
        {
            float B0 = float(src0[(x+0)*NCHANNELS+0]);
            float G0 = float(src0[(x+0)*NCHANNELS+1]);
            float R0 = float(src0[(x+0)*NCHANNELS+2]);

            float B1 = float(src0[(x+1)*NCHANNELS+0]);
            float G1 = float(src0[(x+1)*NCHANNELS+1]);
            float R1 = float(src0[(x+1)*NCHANNELS+2]);

            float B2 = float(src1[(x+0)*NCHANNELS+0]);
            float G2 = float(src1[(x+0)*NCHANNELS+1]);
            float R2 = float(src1[(x+0)*NCHANNELS+2]);

            float B3 = float(src1[(x+1)*NCHANNELS+0]);
            float G3 = float(src1[(x+1)*NCHANNELS+1]);
            float R3 = float(src1[(x+1)*NCHANNELS+2]);

            float Y0 = coeffs[0]*R0 + coeffs[1]*G0 + coeffs[2]*B0 + d1;
            float Y1 = coeffs[0]*R1 + coeffs[1]*G1 + coeffs[2]*B1 + d1;
            float Y2 = coeffs[0]*R2 + coeffs[1]*G2 + coeffs[2]*B2 + d1;
            float Y3 = coeffs[0]*R3 + coeffs[1]*G3 + coeffs[2]*B3 + d1;

            float U = coeffs[3]*R0 + coeffs[4]*G0 + coeffs[5]*B0 + d2;
            float V = coeffs[5]*R0 + coeffs[6]*G0 + coeffs[7]*B0 + d2;

            dstY0[x+0] = saturate_cast<unsigned char>(Y0);
            dstY0[x+1] = saturate_cast<unsigned char>(Y1);
            dstY1[x+0] = saturate_cast<unsigned char>(Y2);
            dstY1[x+1] = saturate_cast<unsigned char>(Y3);

            dstU[x/2] = saturate_cast<unsigned char>(U);
            dstV[x/2] = saturate_cast<unsigned char>(V);
        }
    }
}
#endif // HAVE_VA

void convertToVASurface(VADisplay display, InputArray src, VASurfaceID surface, Size size)
{
    CV_UNUSED(display); CV_UNUSED(src); CV_UNUSED(surface); CV_UNUSED(size);
#if !defined(HAVE_VA)
    NO_VA_SUPPORT_ERROR;
#else  // !HAVE_VA
    const int stype = CV_8UC3;

    int srcType = src.type();
    CV_Assert(srcType == stype);

    Size srcSize = src.size();
    CV_Assert(srcSize.width == size.width && srcSize.height == size.height);

# if (defined(HAVE_VA_INTEL) && defined(HAVE_OPENCL))
    if (contextInitialized)
    {
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
    }
    else
# endif // HAVE_VA_INTEL && HAVE_OPENCL
    {
        Mat m = src.getMat();

        // TODO Add support for roi
        CV_Assert(m.data == m.datastart);
        CV_Assert(m.isContinuous());

        VAStatus status = 0;

        status = vaSyncSurface(display, surface);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaSyncSurface failed");

        VAImage image;
        status = vaDeriveImage(display, surface, &image);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaDeriveImage failed");

        unsigned char* buffer = 0;
        status = vaMapBuffer(display, image.buf, (void **)&buffer);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaMapBuffer failed");

        if (image.format.fourcc == VA_FOURCC_NV12)
            copy_convert_bgr_to_nv12(image, m, buffer);
        if (image.format.fourcc == VA_FOURCC_YV12)
            copy_convert_bgr_to_yv12(image, m, buffer);
        else
            CV_Check((int)image.format.fourcc, image.format.fourcc == VA_FOURCC_NV12 || image.format.fourcc == VA_FOURCC_YV12, "Unexpected image format");

        status = vaUnmapBuffer(display, image.buf);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaUnmapBuffer failed");

        status = vaDestroyImage(display, image.image_id);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaDestroyImage failed");
    }
#endif  // !HAVE_VA
}

void convertFromVASurface(VADisplay display, VASurfaceID surface, Size size, OutputArray dst)
{
    CV_UNUSED(display); CV_UNUSED(surface); CV_UNUSED(dst); CV_UNUSED(size);
#if !defined(HAVE_VA)
    NO_VA_SUPPORT_ERROR;
#else  // !HAVE_VA
    const int dtype = CV_8UC3;

    // TODO Need to specify ACCESS_WRITE here somehow to prevent useless data copying!
    dst.create(size, dtype);

# if (defined(HAVE_VA_INTEL) && defined(HAVE_OPENCL))
    if (contextInitialized)
    {
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
    }
    else
# endif // HAVE_VA_INTEL && HAVE_OPENCL
    {
        Mat m = dst.getMat();

        // TODO Add support for roi
        CV_Assert(m.data == m.datastart);
        CV_Assert(m.isContinuous());

        VAStatus status = 0;

        status = vaSyncSurface(display, surface);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaSyncSurface failed");

        VAImage image;
        status = vaDeriveImage(display, surface, &image);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaDeriveImage failed");

        unsigned char* buffer = 0;
        status = vaMapBuffer(display, image.buf, (void **)&buffer);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaMapBuffer failed");

        if (image.format.fourcc == VA_FOURCC_NV12)
            copy_convert_nv12_to_bgr(image, buffer, m);
        if (image.format.fourcc == VA_FOURCC_YV12)
            copy_convert_yv12_to_bgr(image, buffer, m);
        else
            CV_Check((int)image.format.fourcc, image.format.fourcc == VA_FOURCC_NV12 || image.format.fourcc == VA_FOURCC_YV12, "Unexpected image format");

        status = vaUnmapBuffer(display, image.buf);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaUnmapBuffer failed");

        status = vaDestroyImage(display, image.image_id);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaDestroyImage failed");
    }
#endif  // !HAVE_VA
}

}} // namespace cv::va_intel
