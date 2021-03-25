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
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/directx.hpp"
#include "opencl_kernels_core.hpp"

#ifdef HAVE_DIRECTX
#include <vector>
#include "directx.inc.hpp"
#else // HAVE_DIRECTX
#define NO_DIRECTX_SUPPORT_ERROR CV_Error(cv::Error::StsBadFunc, "OpenCV was build without DirectX support")
#endif

#ifndef HAVE_OPENCL
#define NO_OPENCL_SUPPORT_ERROR CV_Error(cv::Error::StsBadFunc, "OpenCV was build without OpenCL support")
#endif // HAVE_OPENCL

namespace cv { namespace directx {

int getTypeFromDXGI_FORMAT(const int iDXGI_FORMAT)
{
    CV_UNUSED(iDXGI_FORMAT);
#if !defined(HAVE_DIRECTX)
    NO_DIRECTX_SUPPORT_ERROR;
#else
    const int errorType = -1;
    switch ((enum DXGI_FORMAT)iDXGI_FORMAT)
    {
    //case DXGI_FORMAT_UNKNOWN:
    //case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return CV_32FC4;
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:  return CV_32SC4;
    //case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R32G32B32_FLOAT: return CV_32FC3;
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT: return CV_32SC3;
    //case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    //case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT: return CV_16UC4;
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_SINT: return CV_16SC4;
    //case DXGI_FORMAT_R32G32_TYPELESS:
    //case DXGI_FORMAT_R32G32_FLOAT:
    //case DXGI_FORMAT_R32G32_UINT:
    //case DXGI_FORMAT_R32G32_SINT:
    //case DXGI_FORMAT_R32G8X24_TYPELESS:
    //case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    //case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    //case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
    //case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    //case DXGI_FORMAT_R10G10B10A2_UNORM:
    //case DXGI_FORMAT_R10G10B10A2_UINT:
    //case DXGI_FORMAT_R11G11B10_FLOAT:
    //case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UINT: return CV_8UC4;
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SINT: return CV_8SC4;
    //case DXGI_FORMAT_R16G16_TYPELESS:
    //case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_UINT: return CV_16UC2;
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16_SINT: return CV_16SC2;
    //case DXGI_FORMAT_R32_TYPELESS:
    //case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT: return CV_32FC1;
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT: return CV_32SC1;
    //case DXGI_FORMAT_R24G8_TYPELESS:
    //case DXGI_FORMAT_D24_UNORM_S8_UINT:
    //case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
    //case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
    //case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_UINT: return CV_8UC2;
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8_SINT: return CV_8SC2;
    //case DXGI_FORMAT_R16_TYPELESS:
    //case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT: return CV_16UC1;
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16_SINT: return CV_16SC1;
    //case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_UINT: return CV_8UC1;
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8_SINT: return CV_8SC1;
    case DXGI_FORMAT_A8_UNORM: return CV_8UC1;
    //case DXGI_FORMAT_R1_UNORM:
    //case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
    //case DXGI_FORMAT_R8G8_B8G8_UNORM:
    //case DXGI_FORMAT_G8R8_G8B8_UNORM:
    //case DXGI_FORMAT_BC1_TYPELESS:
    //case DXGI_FORMAT_BC1_UNORM:
    //case DXGI_FORMAT_BC1_UNORM_SRGB:
    //case DXGI_FORMAT_BC2_TYPELESS:
    //case DXGI_FORMAT_BC2_UNORM:
    //case DXGI_FORMAT_BC2_UNORM_SRGB:
    //case DXGI_FORMAT_BC3_TYPELESS:
    //case DXGI_FORMAT_BC3_UNORM:
    //case DXGI_FORMAT_BC3_UNORM_SRGB:
    //case DXGI_FORMAT_BC4_TYPELESS:
    //case DXGI_FORMAT_BC4_UNORM:
    //case DXGI_FORMAT_BC4_SNORM:
    //case DXGI_FORMAT_BC5_TYPELESS:
    //case DXGI_FORMAT_BC5_UNORM:
    //case DXGI_FORMAT_BC5_SNORM:
    //case DXGI_FORMAT_B5G6R5_UNORM:
    //case DXGI_FORMAT_B5G5R5A1_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM: return CV_8UC4;
    //case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
    //case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return CV_8UC4;
    //case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: return CV_8UC4;
    //case DXGI_FORMAT_BC6H_TYPELESS:
    //case DXGI_FORMAT_BC6H_UF16:
    //case DXGI_FORMAT_BC6H_SF16:
    //case DXGI_FORMAT_BC7_TYPELESS:
    //case DXGI_FORMAT_BC7_UNORM:
    //case DXGI_FORMAT_BC7_UNORM_SRGB:
#ifdef HAVE_DIRECTX_NV12 //D3DX11 should support DXGI_FORMAT_NV12.
    case DXGI_FORMAT_NV12: return CV_8UC3;
#endif
    default: break;
    }
    return errorType;
#endif
}

int getTypeFromD3DFORMAT(const int iD3DFORMAT)
{
    CV_UNUSED(iD3DFORMAT);
#if !defined(HAVE_DIRECTX)
    NO_DIRECTX_SUPPORT_ERROR;
#else
    const int errorType = -1;
    switch ((enum _D3DFORMAT)iD3DFORMAT)
    {
    //case D3DFMT_UNKNOWN:
    case D3DFMT_R8G8B8: return CV_8UC3;
    case D3DFMT_A8R8G8B8:
    case D3DFMT_X8R8G8B8: return CV_8UC4;
    //case D3DFMT_R5G6B5:
    //case D3DFMT_X1R5G5B5:
    //case D3DFMT_A1R5G5B5:
    //case D3DFMT_A4R4G4B4:
    //case D3DFMT_R3G3B2:
    case D3DFMT_A8: return CV_8UC1;
    //case D3DFMT_A8R3G3B2:
    //case D3DFMT_X4R4G4B4:
    //case D3DFMT_A2B10G10R10:
    case D3DFMT_A8B8G8R8:
    case D3DFMT_X8B8G8R8: return CV_8UC4;
    //case D3DFMT_G16R16:
    //case D3DFMT_A2R10G10B10:
    //case D3DFMT_A16B16G16R16:

    case D3DFMT_A8P8: return CV_8UC2;
    case D3DFMT_P8: return CV_8UC1;

    case D3DFMT_L8: return CV_8UC1;
    case D3DFMT_A8L8: return CV_8UC2;
    //case D3DFMT_A4L4:

    case D3DFMT_V8U8: return CV_8UC2;
    //case D3DFMT_L6V5U5:
    case D3DFMT_X8L8V8U8:
    case D3DFMT_Q8W8V8U8: return CV_8UC4;
    case D3DFMT_V16U16: return CV_16UC4; // TODO 16SC4 ?
    //case D3DFMT_A2W10V10U10:

    case D3DFMT_D16_LOCKABLE: return CV_16UC1;
    case D3DFMT_D32: return CV_32SC1;
    //case D3DFMT_D15S1:
    //case D3DFMT_D24S8:
    //case D3DFMT_D24X8:
    //case D3DFMT_D24X4S4:
    case D3DFMT_D16: return CV_16UC1;

    case D3DFMT_D32F_LOCKABLE: return CV_32FC1;
    default: break;
    }
    return errorType;
#endif
}

namespace ocl {

#if defined(HAVE_DIRECTX) && defined(HAVE_OPENCL)
static bool g_isDirect3DDevice9Ex = false; // Direct3DDevice9Ex or Direct3DDevice9 was used
#endif

Context& initializeContextFromD3D11Device(ID3D11Device* pD3D11Device)
{
    CV_UNUSED(pD3D11Device);
#if !defined(HAVE_DIRECTX)
    NO_DIRECTX_SUPPORT_ERROR;
#elif !defined(HAVE_OPENCL)
    NO_OPENCL_SUPPORT_ERROR;
#else
    cl_uint numPlatforms;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get number of platforms");
    if (numPlatforms == 0)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: No available platforms");

    std::vector<cl_platform_id> platforms(numPlatforms);
    status = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get platforms");

    // TODO Filter platforms by name from OPENCV_OPENCL_DEVICE

    size_t exts_len;
    cv::AutoBuffer<char> extensions;
    bool is_support_cl_khr_d3d11_sharing = false;
#ifdef HAVE_OPENCL_D3D11_NV
    bool is_support_cl_nv_d3d11_sharing = false;
#endif
    for (int i = 0; i < (int)numPlatforms; i++)
    {
        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &exts_len);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get length of CL_PLATFORM_EXTENSIONS");
        extensions.resize(exts_len);
        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, exts_len, static_cast<void*>(extensions.data()), NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLInitError, "OpenCL: No available CL_PLATFORM_EXTENSIONS");
        if (strstr(extensions.data(), "cl_khr_d3d11_sharing"))
            is_support_cl_khr_d3d11_sharing = true;
#ifdef HAVE_OPENCL_D3D11_NV
        if (strstr(extensions.data(), "cl_nv_d3d11_sharing"))
            is_support_cl_nv_d3d11_sharing = true;
#endif
    }
#ifdef HAVE_OPENCL_D3D11_NV
    if (!is_support_cl_nv_d3d11_sharing && !is_support_cl_khr_d3d11_sharing)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: No supported extensions");
#else
    if (!is_support_cl_khr_d3d11_sharing)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: No supported extensions");
#endif

    int found = -1;
    cl_device_id device = NULL;
    cl_uint numDevices = 0;
    cl_context context = NULL;

#ifdef HAVE_OPENCL_D3D11_NV
    if (is_support_cl_nv_d3d11_sharing)
    {
        // try with CL_PREFERRED_DEVICES_FOR_D3D11_NV
        for (int i = 0; i < (int)numPlatforms; i++)
        {
            clGetDeviceIDsFromD3D11NV_fn clGetDeviceIDsFromD3D11NV = (clGetDeviceIDsFromD3D11NV_fn)
                    clGetExtensionFunctionAddressForPlatform(platforms[i], "clGetDeviceIDsFromD3D11NV");
            if (!clGetDeviceIDsFromD3D11NV)
                continue;

            device = NULL;
            numDevices = 0;
            status = clGetDeviceIDsFromD3D11NV(platforms[i], CL_D3D11_DEVICE_NV, pD3D11Device,
                    CL_PREFERRED_DEVICES_FOR_D3D11_NV, 1, &device, &numDevices);
            if (status != CL_SUCCESS)
                continue;
            if (numDevices > 0)
            {
                cl_context_properties properties[] = {
                        CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i],
                        CL_CONTEXT_D3D11_DEVICE_NV, (cl_context_properties)(pD3D11Device),
                        //CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
                        0
                };

                context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
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
        }
        if (found < 0)
        {
            // try with CL_ALL_DEVICES_FOR_D3D11_NV
            for (int i = 0; i < (int)numPlatforms; i++)
            {
                clGetDeviceIDsFromD3D11NV_fn clGetDeviceIDsFromD3D11NV = (clGetDeviceIDsFromD3D11NV_fn)
                        clGetExtensionFunctionAddressForPlatform(platforms[i], "clGetDeviceIDsFromD3D11NV");
                if (!clGetDeviceIDsFromD3D11NV)
                    continue;

                device = NULL;
                numDevices = 0;
                status = clGetDeviceIDsFromD3D11NV(platforms[i], CL_D3D11_DEVICE_NV, pD3D11Device,
                        CL_ALL_DEVICES_FOR_D3D11_NV, 1, &device, &numDevices);
                if (status != CL_SUCCESS)
                    continue;
                if (numDevices > 0)
                {
                    cl_context_properties properties[] = {
                            CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i],
                            CL_CONTEXT_D3D11_DEVICE_NV, (cl_context_properties)(pD3D11Device),
                            //CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
                            0
                    };
                    context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
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
            }
        }
    }
#endif
    if (is_support_cl_khr_d3d11_sharing)
    {
        if (found < 0)
        {
            // try with CL_PREFERRED_DEVICES_FOR_D3D11_KHR
            for (int i = 0; i < (int)numPlatforms; i++)
            {
                clGetDeviceIDsFromD3D11KHR_fn clGetDeviceIDsFromD3D11KHR = (clGetDeviceIDsFromD3D11KHR_fn)
                        clGetExtensionFunctionAddressForPlatform(platforms[i], "clGetDeviceIDsFromD3D11KHR");
                if (!clGetDeviceIDsFromD3D11KHR)
                    continue;

                device = NULL;
                numDevices = 0;

                status = clGetDeviceIDsFromD3D11KHR(platforms[i], CL_D3D11_DEVICE_KHR, pD3D11Device,
                        CL_PREFERRED_DEVICES_FOR_D3D11_KHR, 1, &device, &numDevices);

                if (status != CL_SUCCESS)
                    continue;
                if (numDevices > 0)
                {
                    cl_context_properties properties[] = {
                            CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i],
                            CL_CONTEXT_D3D11_DEVICE_KHR, (cl_context_properties)(pD3D11Device),
                            CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
                            NULL, NULL
                    };
                    context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
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
            }
        }
        if (found < 0)
        {
            // try with CL_ALL_DEVICES_FOR_D3D11_KHR
            for (int i = 0; i < (int)numPlatforms; i++)
            {
                clGetDeviceIDsFromD3D11KHR_fn clGetDeviceIDsFromD3D11KHR = (clGetDeviceIDsFromD3D11KHR_fn)
                        clGetExtensionFunctionAddressForPlatform(platforms[i], "clGetDeviceIDsFromD3D11KHR");
                if (!clGetDeviceIDsFromD3D11KHR)
                    continue;

                device = NULL;
                numDevices = 0;
                status = clGetDeviceIDsFromD3D11KHR(platforms[i], CL_D3D11_DEVICE_KHR, pD3D11Device,
                        CL_ALL_DEVICES_FOR_D3D11_KHR, 1, &device, &numDevices);
                if (status != CL_SUCCESS)
                    continue;
                if (numDevices > 0)
                {
                    cl_context_properties properties[] = {
                            CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i],
                            CL_CONTEXT_D3D11_DEVICE_KHR, (cl_context_properties)(pD3D11Device),
                            CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
                            NULL, NULL
                    };
                    context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
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
            }
        }
    }
    if (found < 0)
    {
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't create context for DirectX interop");
    }

    Context& ctx = Context::getDefault(false);
    initializeContextFromHandle(ctx, platforms[found], context, device);
    return ctx;
#endif
}

Context& initializeContextFromD3D10Device(ID3D10Device* pD3D10Device)
{
    CV_UNUSED(pD3D10Device);
#if !defined(HAVE_DIRECTX)
    NO_DIRECTX_SUPPORT_ERROR;
#elif !defined(HAVE_OPENCL)
    NO_OPENCL_SUPPORT_ERROR;
#else
    cl_uint numPlatforms;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get number of platforms");
    if (numPlatforms == 0)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: No available platforms");

    std::vector<cl_platform_id> platforms(numPlatforms);
    status = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get platforms");

    // TODO Filter platforms by name from OPENCV_OPENCL_DEVICE

    int found = -1;
    cl_device_id device = NULL;
    cl_uint numDevices = 0;
    cl_context context = NULL;

    // try with CL_PREFERRED_DEVICES_FOR_D3D10_KHR
    for (int i = 0; i < (int)numPlatforms; i++)
    {
        clGetDeviceIDsFromD3D10KHR_fn clGetDeviceIDsFromD3D10KHR = (clGetDeviceIDsFromD3D10KHR_fn)
                clGetExtensionFunctionAddressForPlatform(platforms[i], "clGetDeviceIDsFromD3D10KHR");
        if (!clGetDeviceIDsFromD3D10KHR)
            continue;

        device = NULL;
        numDevices = 0;
        status = clGetDeviceIDsFromD3D10KHR(platforms[i], CL_D3D10_DEVICE_KHR, pD3D10Device,
                CL_PREFERRED_DEVICES_FOR_D3D10_KHR, 1, &device, &numDevices);
        if (status != CL_SUCCESS)
            continue;
        if (numDevices > 0)
        {
            cl_context_properties properties[] = {
                    CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i],
                    CL_CONTEXT_D3D10_DEVICE_KHR, (cl_context_properties)(pD3D10Device),
                    CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
                    NULL, NULL
            };
            context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
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
    }
    if (found < 0)
    {
        // try with CL_ALL_DEVICES_FOR_D3D10_KHR
        for (int i = 0; i < (int)numPlatforms; i++)
        {
            clGetDeviceIDsFromD3D10KHR_fn clGetDeviceIDsFromD3D10KHR = (clGetDeviceIDsFromD3D10KHR_fn)
                    clGetExtensionFunctionAddressForPlatform(platforms[i], "clGetDeviceIDsFromD3D10KHR");
            if (!clGetDeviceIDsFromD3D10KHR)
                continue;

            device = NULL;
            numDevices = 0;
            status = clGetDeviceIDsFromD3D10KHR(platforms[i], CL_D3D10_DEVICE_KHR, pD3D10Device,
                    CL_ALL_DEVICES_FOR_D3D10_KHR, 1, &device, &numDevices);
            if (status != CL_SUCCESS)
                continue;
            if (numDevices > 0)
            {
                cl_context_properties properties[] = {
                        CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i],
                        CL_CONTEXT_D3D10_DEVICE_KHR, (cl_context_properties)(pD3D10Device),
                        CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
                        NULL, NULL
                };
                context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
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
        }
        if (found < 0)
            CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't create context for DirectX interop");
    }


    Context& ctx = Context::getDefault(false);
    initializeContextFromHandle(ctx, platforms[found], context, device);
    return ctx;
#endif
}

Context& initializeContextFromDirect3DDevice9Ex(IDirect3DDevice9Ex* pDirect3DDevice9Ex)
{
    CV_UNUSED(pDirect3DDevice9Ex);
#if !defined(HAVE_DIRECTX)
    NO_DIRECTX_SUPPORT_ERROR;
#elif !defined(HAVE_OPENCL)
    NO_OPENCL_SUPPORT_ERROR;
#else
    cl_uint numPlatforms;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get number of platforms");
    if (numPlatforms == 0)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: No available platforms");

    std::vector<cl_platform_id> platforms(numPlatforms);
    status = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get platforms");

    // TODO Filter platforms by name from OPENCV_OPENCL_DEVICE

    int found = -1;
    cl_device_id device = NULL;
    cl_uint numDevices = 0;
    cl_context context = NULL;

    // try with CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR
    for (int i = 0; i < (int)numPlatforms; i++)
    {
        clGetDeviceIDsFromDX9MediaAdapterKHR_fn clGetDeviceIDsFromDX9MediaAdapterKHR = (clGetDeviceIDsFromDX9MediaAdapterKHR_fn)
                clGetExtensionFunctionAddressForPlatform(platforms[i], "clGetDeviceIDsFromDX9MediaAdapterKHR");
        if (!clGetDeviceIDsFromDX9MediaAdapterKHR)
            continue;

        device = NULL;
        numDevices = 0;
        cl_dx9_media_adapter_type_khr type = CL_ADAPTER_D3D9EX_KHR;
        status = clGetDeviceIDsFromDX9MediaAdapterKHR(platforms[i], 1, &type, &pDirect3DDevice9Ex,
                CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR, 1, &device, &numDevices);
        if (status != CL_SUCCESS)
            continue;
        if (numDevices > 0)
        {
            cl_context_properties properties[] = {
                    CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i],
                    CL_CONTEXT_ADAPTER_D3D9EX_KHR, (cl_context_properties)(pDirect3DDevice9Ex),
                    CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
                    NULL, NULL
            };
            context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
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
    }
    if (found < 0)
    {
        // try with CL_ALL_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR
        for (int i = 0; i < (int)numPlatforms; i++)
        {
            clGetDeviceIDsFromDX9MediaAdapterKHR_fn clGetDeviceIDsFromDX9MediaAdapterKHR = (clGetDeviceIDsFromDX9MediaAdapterKHR_fn)
                    clGetExtensionFunctionAddressForPlatform(platforms[i], "clGetDeviceIDsFromDX9MediaAdapterKHR");
            if (!clGetDeviceIDsFromDX9MediaAdapterKHR)
                continue;

            device = NULL;
            numDevices = 0;
            cl_dx9_media_adapter_type_khr type = CL_ADAPTER_D3D9EX_KHR;
            status = clGetDeviceIDsFromDX9MediaAdapterKHR(platforms[i], 1, &type, &pDirect3DDevice9Ex,
                    CL_ALL_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR, 1, &device, &numDevices);
            if (status != CL_SUCCESS)
                continue;
            if (numDevices > 0)
            {
                cl_context_properties properties[] = {
                        CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i],
                        CL_CONTEXT_ADAPTER_D3D9EX_KHR, (cl_context_properties)(pDirect3DDevice9Ex),
                        CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
                        NULL, NULL
                };
                context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
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
        }
        if (found < 0)
            CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't create context for DirectX interop");
    }

    Context& ctx = Context::getDefault(false);
    initializeContextFromHandle(ctx, platforms[found], context, device);
    g_isDirect3DDevice9Ex = true;
    return ctx;
#endif
}

Context& initializeContextFromDirect3DDevice9(IDirect3DDevice9* pDirect3DDevice9)
{
    CV_UNUSED(pDirect3DDevice9);
#if !defined(HAVE_DIRECTX)
    NO_DIRECTX_SUPPORT_ERROR;
#elif !defined(HAVE_OPENCL)
    NO_OPENCL_SUPPORT_ERROR;
#else
    cl_uint numPlatforms;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get number of platforms");
    if (numPlatforms == 0)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: No available platforms");

    std::vector<cl_platform_id> platforms(numPlatforms);
    status = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get platforms");

    // TODO Filter platforms by name from OPENCV_OPENCL_DEVICE

    int found = -1;
    cl_device_id device = NULL;
    cl_uint numDevices = 0;
    cl_context context = NULL;

    // try with CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR
    for (int i = 0; i < (int)numPlatforms; i++)
    {
        clGetDeviceIDsFromDX9MediaAdapterKHR_fn clGetDeviceIDsFromDX9MediaAdapterKHR = (clGetDeviceIDsFromDX9MediaAdapterKHR_fn)
                clGetExtensionFunctionAddressForPlatform(platforms[i], "clGetDeviceIDsFromDX9MediaAdapterKHR");
        if (!clGetDeviceIDsFromDX9MediaAdapterKHR)
            continue;

        device = NULL;
        numDevices = 0;
        cl_dx9_media_adapter_type_khr type = CL_ADAPTER_D3D9_KHR;
        status = clGetDeviceIDsFromDX9MediaAdapterKHR(platforms[i], 1, &type, &pDirect3DDevice9,
                CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR, 1, &device, &numDevices);
        if (status != CL_SUCCESS)
            continue;
        if (numDevices > 0)
        {
            cl_context_properties properties[] = {
                    CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i],
                    CL_CONTEXT_ADAPTER_D3D9_KHR, (cl_context_properties)(pDirect3DDevice9),
                    CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
                    NULL, NULL
            };
            context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
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
    }
    if (found < 0)
    {
        // try with CL_ALL_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR
        for (int i = 0; i < (int)numPlatforms; i++)
        {
            clGetDeviceIDsFromDX9MediaAdapterKHR_fn clGetDeviceIDsFromDX9MediaAdapterKHR = (clGetDeviceIDsFromDX9MediaAdapterKHR_fn)
                    clGetExtensionFunctionAddressForPlatform(platforms[i], "clGetDeviceIDsFromDX9MediaAdapterKHR");
            if (!clGetDeviceIDsFromDX9MediaAdapterKHR)
                continue;

            device = NULL;
            numDevices = 0;
            cl_dx9_media_adapter_type_khr type = CL_ADAPTER_D3D9_KHR;
            status = clGetDeviceIDsFromDX9MediaAdapterKHR(platforms[i], 1, &type, &pDirect3DDevice9,
                    CL_ALL_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR, 1, &device, &numDevices);
            if (status != CL_SUCCESS)
                continue;
            if (numDevices > 0)
            {
                cl_context_properties properties[] = {
                        CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i],
                        CL_CONTEXT_ADAPTER_D3D9_KHR, (cl_context_properties)(pDirect3DDevice9),
                        CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
                        NULL, NULL
                };
                context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
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
        }
        if (found < 0)
            CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't create context for DirectX interop");
    }

    Context& ctx = Context::getDefault(false);
    initializeContextFromHandle(ctx, platforms[found], context, device);
    g_isDirect3DDevice9Ex = false;
    return ctx;
#endif
}

} // namespace cv::ocl

#if defined(HAVE_DIRECTX) && defined(HAVE_OPENCL)

#ifdef HAVE_OPENCL_D3D11_NV
clCreateFromD3D11Texture2DNV_fn clCreateFromD3D11Texture2DNV = NULL;
clEnqueueAcquireD3D11ObjectsNV_fn clEnqueueAcquireD3D11ObjectsNV = NULL;
clEnqueueReleaseD3D11ObjectsNV_fn clEnqueueReleaseD3D11ObjectsNV = NULL;
#endif
clCreateFromD3D11Texture2DKHR_fn clCreateFromD3D11Texture2DKHR = NULL;
clEnqueueAcquireD3D11ObjectsKHR_fn clEnqueueAcquireD3D11ObjectsKHR = NULL;
clEnqueueReleaseD3D11ObjectsKHR_fn clEnqueueReleaseD3D11ObjectsKHR = NULL;

static bool __OpenCLinitializeD3D11()
{
    using namespace cv::ocl;
    static cl_platform_id initializedPlatform = NULL;
    cl_platform_id platform = (cl_platform_id)Platform::getDefault().ptr();

    bool useCLNVEXT = false;
    size_t exts_len;
    cl_int status = clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 0, NULL, &exts_len);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't get length of CL_PLATFORM_EXTENSIONS");
    cv::AutoBuffer<char> extensions(exts_len);
    status = clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, exts_len, static_cast<void*>(extensions.data()), NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: No available CL_PLATFORM_EXTENSIONS");
    bool is_support_cl_khr_d3d11_sharing = false;
    if (strstr(extensions.data(), "cl_khr_d3d11_sharing"))
        is_support_cl_khr_d3d11_sharing = true;
#ifdef HAVE_OPENCL_D3D11_NV
    bool is_support_cl_nv_d3d11_sharing = false;
    if (strstr(extensions.data(), "cl_nv_d3d11_sharing"))
        is_support_cl_nv_d3d11_sharing = true;
    if (!is_support_cl_nv_d3d11_sharing && !is_support_cl_khr_d3d11_sharing)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: No supported extensions");
#else
    if (!is_support_cl_khr_d3d11_sharing)
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: No supported extensions");
#endif

#ifdef HAVE_OPENCL_D3D11_NV
    if (is_support_cl_nv_d3d11_sharing)
    {
        if (initializedPlatform != platform)
        {
            clCreateFromD3D11Texture2DNV = (clCreateFromD3D11Texture2DNV_fn)
                    clGetExtensionFunctionAddressForPlatform(platform, "clCreateFromD3D11Texture2DNV");
            clEnqueueAcquireD3D11ObjectsNV = (clEnqueueAcquireD3D11ObjectsNV_fn)
                    clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueAcquireD3D11ObjectsNV");
            clEnqueueReleaseD3D11ObjectsNV = (clEnqueueReleaseD3D11ObjectsNV_fn)
                    clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueReleaseD3D11ObjectsNV");
            initializedPlatform = platform;
        }
        if (clCreateFromD3D11Texture2DNV && clEnqueueAcquireD3D11ObjectsNV && clEnqueueReleaseD3D11ObjectsNV)
        {
            useCLNVEXT = true;
        }
    }
    else
#endif
    {
        if (is_support_cl_khr_d3d11_sharing)
        {
            if (initializedPlatform != platform)
            {
                clCreateFromD3D11Texture2DKHR = (clCreateFromD3D11Texture2DKHR_fn)
                        clGetExtensionFunctionAddressForPlatform(platform, "clCreateFromD3D11Texture2DKHR");
                clEnqueueAcquireD3D11ObjectsKHR = (clEnqueueAcquireD3D11ObjectsKHR_fn)
                        clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueAcquireD3D11ObjectsKHR");
                clEnqueueReleaseD3D11ObjectsKHR = (clEnqueueReleaseD3D11ObjectsKHR_fn)
                        clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueReleaseD3D11ObjectsKHR");
                initializedPlatform = platform;
            }
            if (!clCreateFromD3D11Texture2DKHR || !clEnqueueAcquireD3D11ObjectsKHR || !clEnqueueReleaseD3D11ObjectsKHR)
            {
                CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't find functions for D3D11");
            }
        }
    }
    return useCLNVEXT;
}
#endif // defined(HAVE_DIRECTX) && defined(HAVE_OPENCL)

} // namespace directx


namespace ocl {

#if defined(HAVE_DIRECTX) && defined(HAVE_OPENCL)
#ifdef HAVE_DIRECTX_NV12

static
bool ocl_convert_nv12_to_bgr(
    cl_mem clImageY,
    cl_mem clImageUV,
    cl_mem clBuffer,
    int step,
    int cols,
    int rows)
{
    ocl::Kernel k;
    k.create("YUV2BGR_NV12_8u", cv::ocl::core::cvtclr_dx_oclsrc, "");
    if (k.empty())
        return false;

    k.args(clImageY, clImageUV, clBuffer, step, cols, rows);

    size_t globalsize[] = { (size_t)cols/2, (size_t)rows/2 };
    return k.run(2, globalsize, 0, false);
}


static
bool ocl_convert_bgr_to_nv12(
    cl_mem clBuffer,
    int step,
    int cols,
    int rows,
    cl_mem clImageY,
    cl_mem clImageUV)
{
    ocl::Kernel k;
    k.create("BGR2YUV_NV12_8u", cv::ocl::core::cvtclr_dx_oclsrc, "");
    if (k.empty())
        return false;

    k.args(clBuffer, step, cols, rows, clImageY, clImageUV);

    size_t globalsize[] = { (size_t)cols/2, (size_t)rows/2 };
    return k.run(2, globalsize, 0, false);
}

#endif // HAVE_DIRECTX_NV12
#endif // HAVE_DIRECTX && HAVE_OPENCL

} // namespace ocl


namespace directx {

#if defined(HAVE_DIRECTX) && defined(HAVE_OPENCL)
static void __convertToD3D11Texture2DKHR(InputArray src, ID3D11Texture2D* pD3D11Texture2D)
{
    D3D11_TEXTURE2D_DESC desc = { 0 };
    pD3D11Texture2D->GetDesc(&desc);

    int srcType = src.type();
    int textureType = getTypeFromDXGI_FORMAT(desc.Format);
    CV_Assert(textureType == srcType);

    Size srcSize = src.size();
    CV_Assert(srcSize.width == (int)desc.Width && srcSize.height == (int)desc.Height);

    UMat u = src.getUMat();

    // TODO Add support for roi
    CV_Assert(u.offset == 0);
    CV_Assert(u.isContinuous());

    cl_mem clBuffer = (cl_mem)u.handle(ACCESS_READ);

    using namespace cv::ocl;
    Context& ctx = Context::getDefault();
    cl_context context = (cl_context)ctx.ptr();

    cl_int status = 0;
    cl_mem clImage = 0;
#ifdef HAVE_DIRECTX_NV12
    cl_mem clImageUV = 0;
#endif
    clImage = clCreateFromD3D11Texture2DKHR(context, CL_MEM_WRITE_ONLY, pD3D11Texture2D, 0, &status);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromD3D11Texture2DKHR failed");

#ifdef HAVE_DIRECTX_NV12
    if(DXGI_FORMAT_NV12 == desc.Format)
    {
        clImageUV = clCreateFromD3D11Texture2DKHR(context, CL_MEM_WRITE_ONLY, pD3D11Texture2D, 1, &status);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromD3D11Texture2DKHR failed");
    }
#endif

    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

    status = clEnqueueAcquireD3D11ObjectsKHR(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireD3D11ObjectsKHR failed");

#ifdef HAVE_DIRECTX_NV12
    if(DXGI_FORMAT_NV12 == desc.Format)
    {
        status = clEnqueueAcquireD3D11ObjectsKHR(q, 1, &clImageUV, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireD3D11ObjectsKHR failed");

        if(!ocl::ocl_convert_bgr_to_nv12(clBuffer, (int)u.step[0], u.cols, u.rows, clImage, clImageUV))
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: ocl_convert_bgr_to_nv12 failed");

        status = clEnqueueReleaseD3D11ObjectsKHR(q, 1, &clImageUV, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseD3D11ObjectsKHR failed");
    }
    else
#endif
    {
        size_t offset = 0; // TODO
        size_t origin[3] = { 0, 0, 0 };
        size_t region[3] = { (size_t)u.cols, (size_t)u.rows, 1 };

        status = clEnqueueCopyBufferToImage(q, clBuffer, clImage, offset, origin, region, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueCopyBufferToImage failed");
    }

    status = clEnqueueReleaseD3D11ObjectsKHR(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseD3D11ObjectsKHR failed");

    status = clFinish(q); // TODO Use events
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clFinish failed");

    status = clReleaseMemObject(clImage); // TODO RAII
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed");

#ifdef HAVE_DIRECTX_NV12
    if(DXGI_FORMAT_NV12 == desc.Format)
    {
        status = clReleaseMemObject(clImageUV);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed");
    }
#endif
}
#endif

#if defined(HAVE_OPENCL_D3D11_NV)
static void __convertToD3D11Texture2DNV(InputArray src, ID3D11Texture2D* pD3D11Texture2D)
{
    D3D11_TEXTURE2D_DESC desc = { 0 };
    pD3D11Texture2D->GetDesc(&desc);

    int srcType = src.type();
    int textureType = getTypeFromDXGI_FORMAT(desc.Format);
    CV_Assert(textureType == srcType);

    Size srcSize = src.size();
    CV_Assert(srcSize.width == (int)desc.Width && srcSize.height == (int)desc.Height);

    UMat u = src.getUMat();

    // TODO Add support for roi
    CV_Assert(u.offset == 0);
    CV_Assert(u.isContinuous());

    cl_mem clBuffer = (cl_mem)u.handle(ACCESS_READ);

    using namespace cv::ocl;
    Context& ctx = Context::getDefault();
    cl_context context = (cl_context)ctx.ptr();

    cl_int status = 0;
    cl_mem clImage = 0;
#ifdef HAVE_DIRECTX_NV12
    cl_mem clImageUV = 0;
#endif
    clImage = clCreateFromD3D11Texture2DNV(context, CL_MEM_WRITE_ONLY, pD3D11Texture2D, 0, &status);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromD3D11Texture2DNV failed");

#ifdef HAVE_DIRECTX_NV12
    if (DXGI_FORMAT_NV12 == desc.Format)
    {
        clImageUV = clCreateFromD3D11Texture2DNV(context, CL_MEM_WRITE_ONLY, pD3D11Texture2D, 1, &status);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromD3D11Texture2DNV failed");
    }
#endif
    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
    status = clEnqueueAcquireD3D11ObjectsNV(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireD3D11ObjectsNV failed");

#ifdef HAVE_DIRECTX_NV12
    if(DXGI_FORMAT_NV12 == desc.Format)
    {
        status = clEnqueueAcquireD3D11ObjectsNV(q, 1, &clImageUV, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireD3D11ObjectsNV failed");

        if(!ocl::ocl_convert_bgr_to_nv12(clBuffer, (int)u.step[0], u.cols, u.rows, clImage, clImageUV))
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: ocl_convert_bgr_to_nv12 failed");

        status = clEnqueueReleaseD3D11ObjectsNV(q, 1, &clImageUV, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseD3D11ObjectsNV failed");
    }
    else
#endif
    {
        size_t offset = 0; // TODO
        size_t origin[3] = { 0, 0, 0 };
        size_t region[3] = { (size_t)u.cols, (size_t)u.rows, 1 };

        status = clEnqueueCopyBufferToImage(q, clBuffer, clImage, offset, origin, region, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueCopyBufferToImage failed");
    }

    status = clEnqueueReleaseD3D11ObjectsNV(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseD3D11ObjectsNV failed");

    status = clFinish(q); // TODO Use events
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clFinish failed");

    status = clReleaseMemObject(clImage); // TODO RAII
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed");

#ifdef HAVE_DIRECTX_NV12
    if(DXGI_FORMAT_NV12 == desc.Format)
    {
        status = clReleaseMemObject(clImageUV);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed");
    }
#endif
}
#endif

#if defined(HAVE_DIRECTX) && defined(HAVE_OPENCL)
static void __convertFromD3D11Texture2DKHR(ID3D11Texture2D* pD3D11Texture2D, OutputArray dst)
{
    D3D11_TEXTURE2D_DESC desc = { 0 };
    pD3D11Texture2D->GetDesc(&desc);

    int textureType = getTypeFromDXGI_FORMAT(desc.Format);
    CV_Assert(textureType >= 0);

    // TODO Need to specify ACCESS_WRITE here somehow to prevent useless data copying!
    dst.create(Size(desc.Width, desc.Height), textureType);
    UMat u = dst.getUMat();

    // TODO Add support for roi
    CV_Assert(u.offset == 0);
    CV_Assert(u.isContinuous());

    cl_mem clBuffer = (cl_mem)u.handle(ACCESS_READ);

    using namespace cv::ocl;
    Context& ctx = Context::getDefault();
    cl_context context = (cl_context)ctx.ptr();

    cl_int status = 0;
    cl_mem clImage = 0;

    clImage = clCreateFromD3D11Texture2DKHR(context, CL_MEM_READ_ONLY, pD3D11Texture2D, 0, &status);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromD3D11Texture2DKHR failed");

#ifdef HAVE_DIRECTX_NV12
    cl_mem clImageUV = 0;
    if(DXGI_FORMAT_NV12 == desc.Format)
    {
        clImageUV = clCreateFromD3D11Texture2DKHR(context, CL_MEM_READ_ONLY, pD3D11Texture2D, 1, &status);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromD3D11Texture2DKHR failed");
    }
#endif

    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

    status = clEnqueueAcquireD3D11ObjectsKHR(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireD3D11ObjectsKHR failed");

#ifdef HAVE_DIRECTX_NV12
    if(DXGI_FORMAT_NV12 == desc.Format)
    {
        status = clEnqueueAcquireD3D11ObjectsKHR(q, 1, &clImageUV, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireD3D11ObjectsKHR failed");

        if(!ocl::ocl_convert_nv12_to_bgr(clImage, clImageUV, clBuffer, (int)u.step[0], u.cols, u.rows))
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: ocl_convert_nv12_to_bgr failed");

        status = clEnqueueReleaseD3D11ObjectsKHR(q, 1, &clImageUV, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseD3D11ObjectsKHR failed");
    }
    else
#endif
    {
        size_t offset = 0; // TODO
        size_t origin[3] = { 0, 0, 0 };
        size_t region[3] = { (size_t)u.cols, (size_t)u.rows, 1 };

        status = clEnqueueCopyImageToBuffer(q, clImage, clBuffer, origin, region, offset, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueCopyImageToBuffer failed");
    }

    status = clEnqueueReleaseD3D11ObjectsKHR(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseD3D11ObjectsKHR failed");

    status = clFinish(q); // TODO Use events
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clFinish failed");

    status = clReleaseMemObject(clImage); // TODO RAII
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed");

#ifdef HAVE_DIRECTX_NV12
    if(DXGI_FORMAT_NV12 == desc.Format)
    {
        status = clReleaseMemObject(clImageUV);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed");
    }
#endif
}
#endif

#if defined(HAVE_OPENCL_D3D11_NV)
static void __convertFromD3D11Texture2DNV(ID3D11Texture2D* pD3D11Texture2D, OutputArray dst)
{
    D3D11_TEXTURE2D_DESC desc = { 0 };
    pD3D11Texture2D->GetDesc(&desc);

    int textureType = getTypeFromDXGI_FORMAT(desc.Format);
    CV_Assert(textureType >= 0);

    // TODO Need to specify ACCESS_WRITE here somehow to prevent useless data copying!
    dst.create(Size(desc.Width, desc.Height), textureType);
    UMat u = dst.getUMat();

    // TODO Add support for roi
    CV_Assert(u.offset == 0);
    CV_Assert(u.isContinuous());

    cl_mem clBuffer = (cl_mem)u.handle(ACCESS_READ);

    using namespace cv::ocl;
    Context& ctx = Context::getDefault();
    cl_context context = (cl_context)ctx.ptr();

    cl_int status = 0;
    cl_mem clImage = 0;

    clImage = clCreateFromD3D11Texture2DNV(context, CL_MEM_READ_ONLY, pD3D11Texture2D, 0, &status);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromD3D11Texture2DNV failed");

#ifdef HAVE_DIRECTX_NV12
    cl_mem clImageUV = 0;
    if(DXGI_FORMAT_NV12 == desc.Format)
    {
        clImageUV = clCreateFromD3D11Texture2DNV(context, CL_MEM_READ_ONLY, pD3D11Texture2D, 1, &status);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromD3D11Texture2DNV failed");
    }
#endif

    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
    status = clEnqueueAcquireD3D11ObjectsNV(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireD3D11ObjectsNV failed");

#ifdef HAVE_DIRECTX_NV12
    if (DXGI_FORMAT::DXGI_FORMAT_NV12 == desc.Format)
    {
        status = clEnqueueAcquireD3D11ObjectsNV(q, 1, &clImageUV, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireD3D11ObjectsNV failed");

        if (!ocl::ocl_convert_nv12_to_bgr(clImage, clImageUV, clBuffer, (int)u.step[0], u.cols, u.rows))
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: ocl_convert_nv12_to_bgr failed");

        status = clEnqueueReleaseD3D11ObjectsNV(q, 1, &clImageUV, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseD3D11ObjectsNV failed");
    }
    else
#endif
    {
        size_t offset = 0; // TODO
        size_t origin[3] = { 0, 0, 0 };
        size_t region[3] = { (size_t)u.cols, (size_t)u.rows, 1 };

        status = clEnqueueCopyImageToBuffer(q, clImage, clBuffer, origin, region, offset, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueCopyImageToBuffer failed");
    }

    status = clEnqueueReleaseD3D11ObjectsNV(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseD3D11ObjectsNV failed");

    status = clFinish(q); // TODO Use events
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clFinish failed");

    status = clReleaseMemObject(clImage); // TODO RAII
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed");

#ifdef HAVE_DIRECTX_NV12
    if(DXGI_FORMAT_NV12 == desc.Format)
    {
        status = clReleaseMemObject(clImageUV);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed");
    }
#endif
}
#endif

void convertToD3D11Texture2D(InputArray src, ID3D11Texture2D* pD3D11Texture2D)
{
    CV_UNUSED(src); CV_UNUSED(pD3D11Texture2D);
#if !defined(HAVE_DIRECTX)
    NO_DIRECTX_SUPPORT_ERROR;
#elif !defined(HAVE_OPENCL)
    NO_OPENCL_SUPPORT_ERROR;
#else

    bool useCLNVEXT = __OpenCLinitializeD3D11();
    if(!useCLNVEXT){
        __convertToD3D11Texture2DKHR(src,pD3D11Texture2D);
    }
#ifdef HAVE_OPENCL_D3D11_NV
    else
    {
        __convertToD3D11Texture2DNV(src,pD3D11Texture2D);
    }
#endif
#endif
}

void convertFromD3D11Texture2D(ID3D11Texture2D* pD3D11Texture2D, OutputArray dst)
{
    CV_UNUSED(pD3D11Texture2D); CV_UNUSED(dst);
#if !defined(HAVE_DIRECTX)
    NO_DIRECTX_SUPPORT_ERROR;
#elif !defined(HAVE_OPENCL)
    NO_OPENCL_SUPPORT_ERROR;
#else

    bool useCLNVEXT = __OpenCLinitializeD3D11();
    if(!useCLNVEXT){
        __convertFromD3D11Texture2DKHR(pD3D11Texture2D,dst);
    }
#ifdef HAVE_OPENCL_D3D11_NV
    else
    {
        __convertFromD3D11Texture2DNV(pD3D11Texture2D,dst);
    }
#endif
#endif
}

#if defined(HAVE_DIRECTX) && defined(HAVE_OPENCL)
clCreateFromD3D10Texture2DKHR_fn clCreateFromD3D10Texture2DKHR = NULL;
clEnqueueAcquireD3D10ObjectsKHR_fn clEnqueueAcquireD3D10ObjectsKHR = NULL;
clEnqueueReleaseD3D10ObjectsKHR_fn clEnqueueReleaseD3D10ObjectsKHR = NULL;

static void __OpenCLinitializeD3D10()
{
    using namespace cv::ocl;
    static cl_platform_id initializedPlatform = NULL;
    cl_platform_id platform = (cl_platform_id)Platform::getDefault().ptr();
    if (initializedPlatform != platform)
    {
        clCreateFromD3D10Texture2DKHR = (clCreateFromD3D10Texture2DKHR_fn)
                clGetExtensionFunctionAddressForPlatform(platform, "clCreateFromD3D10Texture2DKHR");
        clEnqueueAcquireD3D10ObjectsKHR = (clEnqueueAcquireD3D10ObjectsKHR_fn)
                clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueAcquireD3D10ObjectsKHR");
        clEnqueueReleaseD3D10ObjectsKHR = (clEnqueueReleaseD3D10ObjectsKHR_fn)
                clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueReleaseD3D10ObjectsKHR");
        initializedPlatform = platform;
    }
    if (!clCreateFromD3D10Texture2DKHR || !clEnqueueAcquireD3D10ObjectsKHR || !clEnqueueReleaseD3D10ObjectsKHR)
    {
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't find functions for D3D10");
    }
}
#endif // defined(HAVE_DIRECTX) && defined(HAVE_OPENCL)

void convertToD3D10Texture2D(InputArray src, ID3D10Texture2D* pD3D10Texture2D)
{
    CV_UNUSED(src); CV_UNUSED(pD3D10Texture2D);
#if !defined(HAVE_DIRECTX)
    NO_DIRECTX_SUPPORT_ERROR;
#elif defined(HAVE_OPENCL)
    __OpenCLinitializeD3D10();

    D3D10_TEXTURE2D_DESC desc = { 0 };
    pD3D10Texture2D->GetDesc(&desc);

    int srcType = src.type();
    int textureType = getTypeFromDXGI_FORMAT(desc.Format);
    CV_Assert(textureType == srcType);

    Size srcSize = src.size();
    CV_Assert(srcSize.width == (int)desc.Width && srcSize.height == (int)desc.Height);

    using namespace cv::ocl;
    Context& ctx = Context::getDefault();
    cl_context context = (cl_context)ctx.ptr();

    UMat u = src.getUMat();

    // TODO Add support for roi
    CV_Assert(u.offset == 0);
    CV_Assert(u.isContinuous());

    cl_int status = 0;
    cl_mem clImage = clCreateFromD3D10Texture2DKHR(context, CL_MEM_WRITE_ONLY, pD3D10Texture2D, 0, &status);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromD3D10Texture2DKHR failed");

    cl_mem clBuffer = (cl_mem)u.handle(ACCESS_READ);

    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
    status = clEnqueueAcquireD3D10ObjectsKHR(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireD3D10ObjectsKHR failed");
    size_t offset = 0; // TODO
    size_t dst_origin[3] = {0, 0, 0};
    size_t region[3] = {(size_t)u.cols, (size_t)u.rows, 1};
    status = clEnqueueCopyBufferToImage(q, clBuffer, clImage, offset, dst_origin, region, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueCopyBufferToImage failed");
    status = clEnqueueReleaseD3D10ObjectsKHR(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseD3D10ObjectsKHR failed");

    status = clFinish(q); // TODO Use events
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clFinish failed");

    status = clReleaseMemObject(clImage); // TODO RAII
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed");
#else
    // TODO memcpy
    NO_OPENCL_SUPPORT_ERROR;
#endif
}
void convertFromD3D10Texture2D(ID3D10Texture2D* pD3D10Texture2D, OutputArray dst)
{
    CV_UNUSED(pD3D10Texture2D); CV_UNUSED(dst);
#if !defined(HAVE_DIRECTX)
    NO_DIRECTX_SUPPORT_ERROR;
#elif defined(HAVE_OPENCL)
    __OpenCLinitializeD3D10();

    D3D10_TEXTURE2D_DESC desc = { 0 };
    pD3D10Texture2D->GetDesc(&desc);

    int textureType = getTypeFromDXGI_FORMAT(desc.Format);
    CV_Assert(textureType >= 0);

    using namespace cv::ocl;
    Context& ctx = Context::getDefault();
    cl_context context = (cl_context)ctx.ptr();

    // TODO Need to specify ACCESS_WRITE here somehow to prevent useless data copying!
    dst.create(Size(desc.Width, desc.Height), textureType);
    UMat u = dst.getUMat();

    // TODO Add support for roi
    CV_Assert(u.offset == 0);
    CV_Assert(u.isContinuous());

    cl_int status = 0;
    cl_mem clImage = clCreateFromD3D10Texture2DKHR(context, CL_MEM_READ_ONLY, pD3D10Texture2D, 0, &status);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromD3D10Texture2DKHR failed");

    cl_mem clBuffer = (cl_mem)u.handle(ACCESS_READ);

    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
    status = clEnqueueAcquireD3D10ObjectsKHR(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireD3D10ObjectsKHR failed");
    size_t offset = 0; // TODO
    size_t src_origin[3] = {0, 0, 0};
    size_t region[3] = {(size_t)u.cols, (size_t)u.rows, 1};
    status = clEnqueueCopyImageToBuffer(q, clImage, clBuffer, src_origin, region, offset, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueCopyImageToBuffer failed");
    status = clEnqueueReleaseD3D10ObjectsKHR(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseD3D10ObjectsKHR failed");

    status = clFinish(q); // TODO Use events
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clFinish failed");

    status = clReleaseMemObject(clImage); // TODO RAII
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed");
#else
    // TODO memcpy
    NO_OPENCL_SUPPORT_ERROR;
#endif
}

#if defined(HAVE_DIRECTX) && defined(HAVE_OPENCL)
clCreateFromDX9MediaSurfaceKHR_fn clCreateFromDX9MediaSurfaceKHR = NULL;
clEnqueueAcquireDX9MediaSurfacesKHR_fn clEnqueueAcquireDX9MediaSurfacesKHR = NULL;
clEnqueueReleaseDX9MediaSurfacesKHR_fn clEnqueueReleaseDX9MediaSurfacesKHR = NULL;

static void __OpenCLinitializeD3D9()
{
    using namespace cv::ocl;
    static cl_platform_id initializedPlatform = NULL;
    cl_platform_id platform = (cl_platform_id)Platform::getDefault().ptr();
    if (initializedPlatform != platform)
    {
        clCreateFromDX9MediaSurfaceKHR = (clCreateFromDX9MediaSurfaceKHR_fn)
                clGetExtensionFunctionAddressForPlatform(platform, "clCreateFromDX9MediaSurfaceKHR");
        clEnqueueAcquireDX9MediaSurfacesKHR = (clEnqueueAcquireDX9MediaSurfacesKHR_fn)
                clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueAcquireDX9MediaSurfacesKHR");
        clEnqueueReleaseDX9MediaSurfacesKHR = (clEnqueueReleaseDX9MediaSurfacesKHR_fn)
                clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueReleaseDX9MediaSurfacesKHR");
        initializedPlatform = platform;
    }
    if (!clCreateFromDX9MediaSurfaceKHR || !clEnqueueAcquireDX9MediaSurfacesKHR || !clEnqueueReleaseDX9MediaSurfacesKHR)
    {
        CV_Error(cv::Error::OpenCLInitError, "OpenCL: Can't find functions for D3D9");
    }
}
#endif // defined(HAVE_DIRECTX) && defined(HAVE_OPENCL)

void convertToDirect3DSurface9(InputArray src, IDirect3DSurface9* pDirect3DSurface9, void* surfaceSharedHandle)
{
    CV_UNUSED(src); CV_UNUSED(pDirect3DSurface9); CV_UNUSED(surfaceSharedHandle);
#if !defined(HAVE_DIRECTX)
    NO_DIRECTX_SUPPORT_ERROR;
#elif defined(HAVE_OPENCL)
    __OpenCLinitializeD3D9();

    D3DSURFACE_DESC desc;
    if (FAILED(pDirect3DSurface9->GetDesc(&desc)))
    {
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: Can't get D3D surface description");
    }

    int srcType = src.type();
    int surfaceType = getTypeFromD3DFORMAT(desc.Format);
    CV_Assert(surfaceType == srcType);

    Size srcSize = src.size();
    CV_Assert(srcSize.width == (int)desc.Width && srcSize.height == (int)desc.Height);

    using namespace cv::ocl;
    Context& ctx = Context::getDefault();
    cl_context context = (cl_context)ctx.ptr();

    UMat u = src.getUMat();

    // TODO Add support for roi
    CV_Assert(u.offset == 0);
    CV_Assert(u.isContinuous());

    cl_int status = 0;
    cl_dx9_surface_info_khr surfaceInfo = {pDirect3DSurface9, (HANDLE)surfaceSharedHandle};
    cl_mem clImage = clCreateFromDX9MediaSurfaceKHR(context, CL_MEM_WRITE_ONLY,
            ocl::g_isDirect3DDevice9Ex ? CL_ADAPTER_D3D9EX_KHR : CL_ADAPTER_D3D9_KHR,
            &surfaceInfo, 0, &status);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromDX9MediaSurfaceKHR failed");

    cl_mem clBuffer = (cl_mem)u.handle(ACCESS_READ);

    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
    status = clEnqueueAcquireDX9MediaSurfacesKHR(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireDX9MediaSurfacesKHR failed");
    size_t offset = 0; // TODO
    size_t dst_origin[3] = {0, 0, 0};
    size_t region[3] = {(size_t)u.cols, (size_t)u.rows, 1};
    status = clEnqueueCopyBufferToImage(q, clBuffer, clImage, offset, dst_origin, region, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueCopyBufferToImage failed");
    status = clEnqueueReleaseDX9MediaSurfacesKHR(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseDX9MediaSurfacesKHR failed");

    status = clFinish(q); // TODO Use events
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clFinish failed");

    status = clReleaseMemObject(clImage); // TODO RAII
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed");
#else
    // TODO pDirect3DSurface9->LockRect() + memcpy + Unlock()
    NO_OPENCL_SUPPORT_ERROR;
#endif
}

void convertFromDirect3DSurface9(IDirect3DSurface9* pDirect3DSurface9, OutputArray dst, void* surfaceSharedHandle)
{
    CV_UNUSED(pDirect3DSurface9); CV_UNUSED(dst); CV_UNUSED(surfaceSharedHandle);
#if !defined(HAVE_DIRECTX)
    NO_DIRECTX_SUPPORT_ERROR;
#elif defined(HAVE_OPENCL)
    __OpenCLinitializeD3D9();

    D3DSURFACE_DESC desc;
    if (FAILED(pDirect3DSurface9->GetDesc(&desc)))
    {
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: Can't get D3D surface description");
    }

    int surfaceType = getTypeFromD3DFORMAT(desc.Format);
    CV_Assert(surfaceType >= 0);

    using namespace cv::ocl;
    Context& ctx = Context::getDefault();
    cl_context context = (cl_context)ctx.ptr();

    // TODO Need to specify ACCESS_WRITE here somehow to prevent useless data copying!
    dst.create(Size(desc.Width, desc.Height), surfaceType);
    UMat u = dst.getUMat();

    // TODO Add support for roi
    CV_Assert(u.offset == 0);
    CV_Assert(u.isContinuous());

    cl_int status = 0;
    cl_dx9_surface_info_khr surfaceInfo = {pDirect3DSurface9, (HANDLE)surfaceSharedHandle};
    cl_mem clImage = clCreateFromDX9MediaSurfaceKHR(context, CL_MEM_READ_ONLY,
            ocl::g_isDirect3DDevice9Ex ? CL_ADAPTER_D3D9EX_KHR : CL_ADAPTER_D3D9_KHR,
            &surfaceInfo, 0, &status);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromDX9MediaSurfaceKHR failed");

    cl_mem clBuffer = (cl_mem)u.handle(ACCESS_WRITE);

    cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
    status = clEnqueueAcquireDX9MediaSurfacesKHR(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireDX9MediaSurfacesKHR failed");
    size_t offset = 0; // TODO
    size_t src_origin[3] = {0, 0, 0};
    size_t region[3] = {(size_t)u.cols, (size_t)u.rows, 1};
    status = clEnqueueCopyImageToBuffer(q, clImage, clBuffer, src_origin, region, offset, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueCopyImageToBuffer failed");
    status = clEnqueueReleaseDX9MediaSurfacesKHR(q, 1, &clImage, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseDX9MediaSurfacesKHR failed");

    status = clFinish(q); // TODO Use events
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clFinish failed");

    status = clReleaseMemObject(clImage); // TODO RAII
    if (status != CL_SUCCESS)
        CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed");
#else
    // TODO pDirect3DSurface9->LockRect() + memcpy + Unlock()
    NO_OPENCL_SUPPORT_ERROR;
#endif
}

} } // namespace cv::directx
