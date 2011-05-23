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

using namespace cv;
using namespace cv::gpu;

#if !defined (HAVE_CUDA)

void cv::gpu::cvtColor(const GpuMat&, GpuMat&, int, int) { throw_nogpu(); }
void cv::gpu::cvtColor(const GpuMat&, GpuMat&, int, int, const Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu {  namespace color  
{
    void RGB2RGB_gpu_8u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, cudaStream_t stream);
    void RGB2RGB_gpu_16u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, cudaStream_t stream);
    void RGB2RGB_gpu_32f(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, cudaStream_t stream);

    void RGB5x52RGB_gpu(const DevMem2D& src, int green_bits, const DevMem2D& dst, int dstcn, int bidx, cudaStream_t stream);
    void RGB2RGB5x5_gpu(const DevMem2D& src, int srccn, const DevMem2D& dst, int green_bits, int bidx, cudaStream_t stream);

    void Gray2RGB_gpu_8u(const DevMem2D& src, const DevMem2D& dst, int dstcn, cudaStream_t stream);
    void Gray2RGB_gpu_16u(const DevMem2D& src, const DevMem2D& dst, int dstcn, cudaStream_t stream);
    void Gray2RGB_gpu_32f(const DevMem2D& src, const DevMem2D& dst, int dstcn, cudaStream_t stream);
    void Gray2RGB5x5_gpu(const DevMem2D& src, const DevMem2D& dst, int green_bits, cudaStream_t stream);

    void RGB2Gray_gpu_8u(const DevMem2D& src, int srccn, const DevMem2D& dst, int bidx, cudaStream_t stream);
    void RGB2Gray_gpu_16u(const DevMem2D& src, int srccn, const DevMem2D& dst, int bidx, cudaStream_t stream);
    void RGB2Gray_gpu_32f(const DevMem2D& src, int srccn, const DevMem2D& dst, int bidx, cudaStream_t stream);
    void RGB5x52Gray_gpu(const DevMem2D& src, int green_bits, const DevMem2D& dst, cudaStream_t stream);

    void RGB2YCrCb_gpu_8u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, const void* coeffs, cudaStream_t stream);
    void RGB2YCrCb_gpu_16u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, const void* coeffs, cudaStream_t stream);
    void RGB2YCrCb_gpu_32f(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, const void* coeffs, cudaStream_t stream);

    void YCrCb2RGB_gpu_8u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, const void* coeffs, cudaStream_t stream);
    void YCrCb2RGB_gpu_16u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, const void* coeffs, cudaStream_t stream);
    void YCrCb2RGB_gpu_32f(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, const void* coeffs, cudaStream_t stream);

    void RGB2XYZ_gpu_8u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, const void* coeffs, cudaStream_t stream);
    void RGB2XYZ_gpu_16u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, const void* coeffs, cudaStream_t stream);
    void RGB2XYZ_gpu_32f(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, const void* coeffs, cudaStream_t stream);

    void XYZ2RGB_gpu_8u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, const void* coeffs, cudaStream_t stream);
    void XYZ2RGB_gpu_16u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, const void* coeffs, cudaStream_t stream);
    void XYZ2RGB_gpu_32f(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, const void* coeffs, cudaStream_t stream);

    void RGB2HSV_gpu_8u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, int hrange, cudaStream_t stream);
    void RGB2HSV_gpu_32f(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, int hrange, cudaStream_t stream);

    void HSV2RGB_gpu_8u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, int hrange, cudaStream_t stream);
    void HSV2RGB_gpu_32f(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, int hrange, cudaStream_t stream);

    void RGB2HLS_gpu_8u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, int hrange, cudaStream_t stream);
    void RGB2HLS_gpu_32f(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, int hrange, cudaStream_t stream);

    void HLS2RGB_gpu_8u(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, int hrange, cudaStream_t stream);
    void HLS2RGB_gpu_32f(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, int hrange, cudaStream_t stream);
}}}

namespace
{
    #undef R2Y
    #undef G2Y
    #undef B2Y
    
    enum
    {
        yuv_shift  = 14,
        xyz_shift  = 12,
        R2Y        = 4899,
        G2Y        = 9617,
        B2Y        = 1868,
        BLOCK_SIZE = 256
    };
}

namespace
{
    void cvtColor_caller(const GpuMat& src, GpuMat& dst, int code, int dcn, const cudaStream_t& stream) 
    {
        Size sz = src.size();
        int scn = src.channels(), depth = src.depth(), bidx;
        
        CV_Assert(depth == CV_8U || depth == CV_16U || depth == CV_32F);

        switch (code)
        {
            case CV_BGR2BGRA: case CV_RGB2BGRA: case CV_BGRA2BGR:
            case CV_RGBA2BGR: case CV_RGB2BGR: case CV_BGRA2RGBA:                
                {
                    typedef void (*func_t)(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, cudaStream_t stream);
                    static const func_t funcs[] = {color::RGB2RGB_gpu_8u, 0, color::RGB2RGB_gpu_16u, 0, 0, color::RGB2RGB_gpu_32f};

                    CV_Assert(scn == 3 || scn == 4);

                    dcn = code == CV_BGR2BGRA || code == CV_RGB2BGRA || code == CV_BGRA2RGBA ? 4 : 3;
                    bidx = code == CV_BGR2BGRA || code == CV_BGRA2BGR ? 0 : 2;
                    
                    dst.create(sz, CV_MAKETYPE(depth, dcn));

                    funcs[depth](src, scn, dst, dcn, bidx, stream);
                    break;
                }
                
            case CV_BGR2BGR565: case CV_BGR2BGR555: case CV_RGB2BGR565: case CV_RGB2BGR555:
            case CV_BGRA2BGR565: case CV_BGRA2BGR555: case CV_RGBA2BGR565: case CV_RGBA2BGR555:
                {
                    CV_Assert((scn == 3 || scn == 4) && depth == CV_8U);

                    int green_bits = code == CV_BGR2BGR565 || code == CV_RGB2BGR565 
                        || code == CV_BGRA2BGR565 || code == CV_RGBA2BGR565 ? 6 : 5;
                    bidx = code == CV_BGR2BGR565 || code == CV_BGR2BGR555 
                        || code == CV_BGRA2BGR565 || code == CV_BGRA2BGR555 ? 0 : 2;

                    dst.create(sz, CV_8UC2);

                    color::RGB2RGB5x5_gpu(src, scn, dst, green_bits, bidx, stream);
                    break;
                }
            
            case CV_BGR5652BGR: case CV_BGR5552BGR: case CV_BGR5652RGB: case CV_BGR5552RGB:
            case CV_BGR5652BGRA: case CV_BGR5552BGRA: case CV_BGR5652RGBA: case CV_BGR5552RGBA:
                {
                    if (dcn <= 0) dcn = 3;

                    CV_Assert((dcn == 3 || dcn == 4) && scn == 2 && depth == CV_8U);

                    int green_bits = code == CV_BGR5652BGR || code == CV_BGR5652RGB 
                        || code == CV_BGR5652BGRA || code == CV_BGR5652RGBA ? 6 : 5;
                    bidx = code == CV_BGR5652BGR || code == CV_BGR5552BGR 
                        || code == CV_BGR5652BGRA || code == CV_BGR5552BGRA ? 0 : 2;

                    dst.create(sz, CV_MAKETYPE(depth, dcn));

                    color::RGB5x52RGB_gpu(src, green_bits, dst, dcn, bidx, stream);
                    break;
                }
                        
            case CV_BGR2GRAY: case CV_BGRA2GRAY: case CV_RGB2GRAY: case CV_RGBA2GRAY:
                {
                    typedef void (*func_t)(const DevMem2D& src, int srccn, const DevMem2D& dst, int bidx, cudaStream_t stream);
                    static const func_t funcs[] = {color::RGB2Gray_gpu_8u, 0, color::RGB2Gray_gpu_16u, 0, 0, color::RGB2Gray_gpu_32f};

                    CV_Assert(scn == 3 || scn == 4);
                    
                    bidx = code == CV_BGR2GRAY || code == CV_BGRA2GRAY ? 0 : 2;

                    dst.create(sz, CV_MAKETYPE(depth, 1));

                    funcs[depth](src, scn, dst, bidx, stream);
                    break;
                }
            
            case CV_BGR5652GRAY: case CV_BGR5552GRAY:
                {
                    CV_Assert(scn == 2 && depth == CV_8U);

                    int green_bits = code == CV_BGR5652GRAY ? 6 : 5;

                    dst.create(sz, CV_8UC1);

                    color::RGB5x52Gray_gpu(src, green_bits, dst, stream);
                    break;
                }
            
            case CV_GRAY2BGR: case CV_GRAY2BGRA:
                {
                    typedef void (*func_t)(const DevMem2D& src, const DevMem2D& dst, int dstcn, cudaStream_t stream);
                    static const func_t funcs[] = {color::Gray2RGB_gpu_8u, 0, color::Gray2RGB_gpu_16u, 0, 0, color::Gray2RGB_gpu_32f};

                    if (dcn <= 0) dcn = 3;

                    CV_Assert(scn == 1 && (dcn == 3 || dcn == 4));

                    dst.create(sz, CV_MAKETYPE(depth, dcn));

                    funcs[depth](src, dst, dcn, stream);
                    break;
                }
                
            case CV_GRAY2BGR565: case CV_GRAY2BGR555:
                {
                    CV_Assert(scn == 1 && depth == CV_8U);

                    int green_bits =  code == CV_GRAY2BGR565 ? 6 : 5;

                    dst.create(sz, CV_8UC2);
                    
                    color::Gray2RGB5x5_gpu(src, dst, green_bits, stream);
                    break;
                }

            case CV_BGR2YCrCb: case CV_RGB2YCrCb:
            case CV_BGR2YUV: case CV_RGB2YUV:
                {
                    typedef void (*func_t)(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, 
                        const void* coeffs, cudaStream_t stream);
                    static const func_t funcs[] = {color::RGB2YCrCb_gpu_8u, 0, color::RGB2YCrCb_gpu_16u, 0, 0, color::RGB2YCrCb_gpu_32f};

                    if (dcn <= 0) dcn = 3;
                    CV_Assert((scn == 3 || scn == 4) && (dcn == 3 || dcn == 4));

                    bidx = code == CV_BGR2YCrCb || code == CV_RGB2YUV ? 0 : 2;

                    static const float yuv_f[] = { 0.114f, 0.587f, 0.299f, 0.492f, 0.877f };
                    static const int yuv_i[] = { B2Y, G2Y, R2Y, 8061, 14369 };

                    static const float YCrCb_f[] = {0.299f, 0.587f, 0.114f, 0.713f, 0.564f};
                    static const int YCrCb_i[] = {R2Y, G2Y, B2Y, 11682, 9241};

                    float coeffs_f[5];
                    int coeffs_i[5];
                    ::memcpy(coeffs_f, code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? YCrCb_f : yuv_f, sizeof(yuv_f));
                    ::memcpy(coeffs_i, code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? YCrCb_i : yuv_i, sizeof(yuv_i));

                    if (bidx == 0) 
                    {
                        std::swap(coeffs_f[0], coeffs_f[2]);
                        std::swap(coeffs_i[0], coeffs_i[2]);
                    }
                        
                    dst.create(sz, CV_MAKETYPE(depth, dcn));

                    const void* coeffs = depth == CV_32F ? (void*)coeffs_f : (void*)coeffs_i;

                    funcs[depth](src, scn, dst, dcn, bidx, coeffs, stream);
                    break;
                }
                
            case CV_YCrCb2BGR: case CV_YCrCb2RGB:
            case CV_YUV2BGR: case CV_YUV2RGB:
                {
                    typedef void (*func_t)(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, 
                        const void* coeffs, cudaStream_t stream);
                    static const func_t funcs[] = {color::YCrCb2RGB_gpu_8u, 0, color::YCrCb2RGB_gpu_16u, 0, 0, color::YCrCb2RGB_gpu_32f};

                    if (dcn <= 0) dcn = 3;

                    CV_Assert((scn == 3 || scn == 4) && (dcn == 3 || dcn == 4));

                    bidx = code == CV_YCrCb2BGR || code == CV_YUV2RGB ? 0 : 2;

                    static const float yuv_f[] = { 2.032f, -0.395f, -0.581f, 1.140f };
                    static const int yuv_i[] = { 33292, -6472, -9519, 18678 }; 

                    static const float YCrCb_f[] = {1.403f, -0.714f, -0.344f, 1.773f};
                    static const int YCrCb_i[] = {22987, -11698, -5636, 29049};

                    const float* coeffs_f = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? YCrCb_f : yuv_f;
                    const int* coeffs_i = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? YCrCb_i : yuv_i;
                    
                    dst.create(sz, CV_MAKETYPE(depth, dcn));
                    
                    const void* coeffs = depth == CV_32F ? (void*)coeffs_f : (void*)coeffs_i;

                    funcs[depth](src, scn, dst, dcn, bidx, coeffs, stream);
                    break;
                }
            
            case CV_BGR2XYZ: case CV_RGB2XYZ:
                {
                    typedef void (*func_t)(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, 
                        const void* coeffs, cudaStream_t stream);
                    static const func_t funcs[] = {color::RGB2XYZ_gpu_8u, 0, color::RGB2XYZ_gpu_16u, 0, 0, color::RGB2XYZ_gpu_32f};

                    if (dcn <= 0) dcn = 3;

                    CV_Assert((scn == 3 || scn == 4) && (dcn == 3 || dcn == 4));

                    bidx = code == CV_BGR2XYZ ? 0 : 2;

                    static const float RGB2XYZ_D65f[] =
                    {
                        0.412453f, 0.357580f, 0.180423f,
                        0.212671f, 0.715160f, 0.072169f,
                        0.019334f, 0.119193f, 0.950227f
                    };
                    static const int RGB2XYZ_D65i[] =
                    {
                        1689,    1465,    739,
                        871,     2929,    296,
                        79,      488,     3892
                    };

                    float coeffs_f[9];
                    int coeffs_i[9];
                    ::memcpy(coeffs_f, RGB2XYZ_D65f, sizeof(RGB2XYZ_D65f));
                    ::memcpy(coeffs_i, RGB2XYZ_D65i, sizeof(RGB2XYZ_D65i));

                    if (bidx == 0) 
                    {
                        std::swap(coeffs_f[0], coeffs_f[2]);
                        std::swap(coeffs_f[3], coeffs_f[5]);
                        std::swap(coeffs_f[6], coeffs_f[8]);
                        
                        std::swap(coeffs_i[0], coeffs_i[2]);
                        std::swap(coeffs_i[3], coeffs_i[5]);
                        std::swap(coeffs_i[6], coeffs_i[8]);
                    }
                        
                    dst.create(sz, CV_MAKETYPE(depth, dcn));
                    
                    const void* coeffs = depth == CV_32F ? (void*)coeffs_f : (void*)coeffs_i;
                    
                    funcs[depth](src, scn, dst, dcn, coeffs, stream);
                    break;
                }
            
            case CV_XYZ2BGR: case CV_XYZ2RGB:
                {
                    typedef void (*func_t)(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, const void* coeffs, cudaStream_t stream);
                    static const func_t funcs[] = {color::XYZ2RGB_gpu_8u, 0, color::XYZ2RGB_gpu_16u, 0, 0, color::XYZ2RGB_gpu_32f};

                    if (dcn <= 0) dcn = 3;

                    CV_Assert((scn == 3 || scn == 4) && (dcn == 3 || dcn == 4));

                    bidx = code == CV_XYZ2BGR ? 0 : 2;

                    static const float XYZ2sRGB_D65f[] =
                    {
                        3.240479f, -1.53715f, -0.498535f,
                        -0.969256f, 1.875991f, 0.041556f,
                        0.055648f, -0.204043f, 1.057311f
                    };
                    static const int XYZ2sRGB_D65i[] =
                    {
                        13273,  -6296,  -2042,
                        -3970,   7684,    170,
                          228,   -836,   4331
                    };

                    float coeffs_f[9];
                    int coeffs_i[9];
                    ::memcpy(coeffs_f, XYZ2sRGB_D65f, sizeof(XYZ2sRGB_D65f));
                    ::memcpy(coeffs_i, XYZ2sRGB_D65i, sizeof(XYZ2sRGB_D65i));

                    if (bidx == 0) 
                    {
                        std::swap(coeffs_f[0], coeffs_f[6]);
                        std::swap(coeffs_f[1], coeffs_f[7]);
                        std::swap(coeffs_f[2], coeffs_f[8]);
                        
                        std::swap(coeffs_i[0], coeffs_i[6]);
                        std::swap(coeffs_i[1], coeffs_i[7]);
                        std::swap(coeffs_i[2], coeffs_i[8]);
                    }
                        
                    dst.create(sz, CV_MAKETYPE(depth, dcn));
                    
                    const void* coeffs = depth == CV_32F ? (void*)coeffs_f : (void*)coeffs_i;

                    funcs[depth](src, scn, dst, dcn, coeffs, stream);
                    break;
                }

            case CV_BGR2HSV: case CV_RGB2HSV: case CV_BGR2HSV_FULL: case CV_RGB2HSV_FULL:
            case CV_BGR2HLS: case CV_RGB2HLS: case CV_BGR2HLS_FULL: case CV_RGB2HLS_FULL:
                {
                    typedef void (*func_t)(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, 
                        int hrange, cudaStream_t stream);
                    static const func_t funcs_hsv[] = {color::RGB2HSV_gpu_8u, 0, 0, 0, 0, color::RGB2HSV_gpu_32f};
                    static const func_t funcs_hls[] = {color::RGB2HLS_gpu_8u, 0, 0, 0, 0, color::RGB2HLS_gpu_32f};

                    if (dcn <= 0) dcn = 3;

                    CV_Assert((scn == 3 || scn == 4) && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F));

                    bidx = code == CV_BGR2HSV || code == CV_BGR2HLS ||
                        code == CV_BGR2HSV_FULL || code == CV_BGR2HLS_FULL ? 0 : 2;
                    int hrange = depth == CV_32F ? 360 : code == CV_BGR2HSV || code == CV_RGB2HSV ||
                        code == CV_BGR2HLS || code == CV_RGB2HLS ? 180 : 255;
                
                    dst.create(sz, CV_MAKETYPE(depth, dcn));

                    if (code == CV_BGR2HSV || code == CV_RGB2HSV || code == CV_BGR2HSV_FULL || code == CV_RGB2HSV_FULL) 
                        funcs_hsv[depth](src, scn, dst, dcn, bidx, hrange, stream);
                    else
                        funcs_hls[depth](src, scn, dst, dcn, bidx, hrange, stream);
                    break;
                }

            case CV_HSV2BGR: case CV_HSV2RGB: case CV_HSV2BGR_FULL: case CV_HSV2RGB_FULL:
            case CV_HLS2BGR: case CV_HLS2RGB: case CV_HLS2BGR_FULL: case CV_HLS2RGB_FULL:
                {
                    typedef void (*func_t)(const DevMem2D& src, int srccn, const DevMem2D& dst, int dstcn, int bidx, 
                        int hrange, cudaStream_t stream);
                    static const func_t funcs_hsv[] = {color::HSV2RGB_gpu_8u, 0, 0, 0, 0, color::HSV2RGB_gpu_32f};
                    static const func_t funcs_hls[] = {color::HLS2RGB_gpu_8u, 0, 0, 0, 0, color::HLS2RGB_gpu_32f};

                    if (dcn <= 0) dcn = 3;

                    CV_Assert((scn == 3 || scn == 4) && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F));

                    bidx = code == CV_HSV2BGR || code == CV_HLS2BGR ||
                        code == CV_HSV2BGR_FULL || code == CV_HLS2BGR_FULL ? 0 : 2;
                    int hrange = depth == CV_32F ? 360 : code == CV_HSV2BGR || code == CV_HSV2RGB ||
                        code == CV_HLS2BGR || code == CV_HLS2RGB ? 180 : 255;
                    
                    dst.create(sz, CV_MAKETYPE(depth, dcn));

                    if (code == CV_HSV2BGR || code == CV_HSV2RGB || code == CV_HSV2BGR_FULL || code == CV_HSV2RGB_FULL)
                        funcs_hsv[depth](src, scn, dst, dcn, bidx, hrange, stream);
                    else
                        funcs_hls[depth](src, scn, dst, dcn, bidx, hrange, stream);
                    break;
                }

            default:
                CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );
        }
    }
}

void cv::gpu::cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn)
{
    cvtColor_caller(src, dst, code, dcn, 0);
}

void cv::gpu::cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn, const Stream& stream)
{
    cvtColor_caller(src, dst, code, dcn, StreamAccessor::getStream(stream));
}

#endif /* !defined (HAVE_CUDA) */
