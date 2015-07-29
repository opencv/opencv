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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
//
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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#ifdef INTEL_DEVICE
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL FP_FAST_FMAF ON
#pragma OPENCL FP_FAST_FMA ON
#endif

static
__constant
float c_YUV2RGBCoeffs_420[5] =
{
     1.163999557f,
     2.017999649f,
    -0.390999794f,
    -0.812999725f,
     1.5959997177f
};


__kernel
void YUV2RGBA_NV12_8u(
    read_only image2d_t imgY,
    read_only image2d_t imgUV,
    __global unsigned char* pRGBA,
   int rgbaStep,
   int cols,
   int rows)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols)
    {
        if (y < rows)
        {
            __global uchar* pDstRow1 = pRGBA + mad24(y, rgbaStep, mad24(x, 4, 0));
            __global uchar* pDstRow2 = pDstRow1 + rgbaStep;

            float4 Y1 = read_imagef(imgY, (int2)(x+0, y+0)) * 255.0f;
            float4 Y2 = read_imagef(imgY, (int2)(x+1, y+0)) * 255.0f;
            float4 Y3 = read_imagef(imgY, (int2)(x+0, y+1)) * 255.0f;
            float4 Y4 = read_imagef(imgY, (int2)(x+1, y+1)) * 255.0f;

            float4 UV = read_imagef(imgUV, (int2)(x/2, y/2)) * 255.0f - 128.0f;

            __constant float* coeffs = c_YUV2RGBCoeffs_420;
            float ruv = fma(coeffs[4], UV.y, 0.5f);
            float guv = fma(coeffs[3], UV.y, fma(coeffs[2], UV.x, 0.5f));
            float buv = fma(coeffs[1], UV.x, 0.5f);

            Y1 = max(0.f, Y1 - 16.f) * coeffs[0];
            pDstRow1[0+0] = convert_uchar_sat(Y1.x + ruv);
            pDstRow1[1+0] = convert_uchar_sat(Y1.x + guv);
            pDstRow1[2+0] = convert_uchar_sat(Y1.x + buv);
            pDstRow1[3+0] = 255;

            Y2 = max(0.f, Y2 - 16.f) * coeffs[0];
            pDstRow1[0+4] = convert_uchar_sat(Y2.x + ruv);
            pDstRow1[1+4] = convert_uchar_sat(Y2.x + guv);
            pDstRow1[2+4] = convert_uchar_sat(Y2.x + buv);
            pDstRow1[3+4] = 255;

            Y3 = max(0.f, Y3 - 16.f) * coeffs[0];
            pDstRow2[0+0] = convert_uchar_sat(Y3.x + ruv);
            pDstRow2[1+0] = convert_uchar_sat(Y3.x + guv);
            pDstRow2[2+0] = convert_uchar_sat(Y3.x + buv);
            pDstRow2[3+0] = 255;

            Y4 = max(0.f, Y4 - 16.f) * coeffs[0];
            pDstRow2[0+4] = convert_uchar_sat(Y4.x + ruv);
            pDstRow2[1+4] = convert_uchar_sat(Y4.x + guv);
            pDstRow2[2+4] = convert_uchar_sat(Y4.x + buv);
            pDstRow2[3+4] = 255;
        }
    }
}


static
__constant float c_RGB2YUVCoeffs_420[8] =
{
     0.256999969f,  0.50399971f,   0.09799957f,   -0.1479988098f,
    -0.2909994125f, 0.438999176f, -0.3679990768f, -0.0709991455f
};

#define scn 4
__kernel
void RGBA2YUV_NV12_8u(
    __global unsigned char* pRGBA,
    int rgbaStep,
    int cols,
    int rows,
    write_only image2d_t imgY,
    write_only image2d_t imgUV)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols)
    {
        if (y < rows)
        {
            __global const uchar* pSrcRow1 = pRGBA + mad24(y, rgbaStep, mad24(x, scn, 0));
            __global const uchar* pSrcRow2 = pSrcRow1 + rgbaStep;

            float4 src_pix1 = convert_float4(vload4(0, pSrcRow1 + 0));
            float4 src_pix2 = convert_float4(vload4(0, pSrcRow1 + scn));
            float4 src_pix3 = convert_float4(vload4(0, pSrcRow2 + 0));
            float4 src_pix4 = convert_float4(vload4(0, pSrcRow2 + scn));

            __constant float* coeffs = c_RGB2YUVCoeffs_420;

            uchar Y1 = convert_uchar_sat(fma(coeffs[0], src_pix1.x, fma(coeffs[1], src_pix1.y, fma(coeffs[2], src_pix1.z, 16.5f))));
            uchar Y2 = convert_uchar_sat(fma(coeffs[0], src_pix2.x, fma(coeffs[1], src_pix2.y, fma(coeffs[2], src_pix2.z, 16.5f))));
            uchar Y3 = convert_uchar_sat(fma(coeffs[0], src_pix3.x, fma(coeffs[1], src_pix3.y, fma(coeffs[2], src_pix3.z, 16.5f))));
            uchar Y4 = convert_uchar_sat(fma(coeffs[0], src_pix4.x, fma(coeffs[1], src_pix4.y, fma(coeffs[2], src_pix4.z, 16.5f))));

            write_imageui(imgY, (int2)(x+0, y+0), Y1);
            write_imageui(imgY, (int2)(x+1, y+0), Y2);
            write_imageui(imgY, (int2)(x+0, y+1), Y3);
            write_imageui(imgY, (int2)(x+1, y+1), Y4);

            float uf = fma(coeffs[3], src_pix1.x, fma(coeffs[4], src_pix1.y, fma(coeffs[5], src_pix1.z, 128.5f)));
            float vf = fma(coeffs[5], src_pix1.x, fma(coeffs[6], src_pix1.y, fma(coeffs[7], src_pix1.z, 128.5f)));

            uchar U = convert_uchar_sat(uf);
            uchar V = convert_uchar_sat(vf);

            write_imageui(imgUV, (int2)((x/2)+0, (y/2)), U);
            write_imageui(imgUV, (int2)((x/2)+1, (y/2)), V);
        }
    }
}