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

static const __constant float CV_8U_MAX         = 255.0f;
static const __constant float CV_8U_HALF        = 128.0f;
static const __constant float BT601_BLACK_RANGE = 16.0f;
static const __constant float CV_8U_SCALE       = 1.0f / 255.0f;
static const __constant float d1                = BT601_BLACK_RANGE / CV_8U_MAX;
static const __constant float d2                = CV_8U_HALF / CV_8U_MAX;

#define NCHANNELS 3

__kernel
void YUV2BGR_NV12_8u(
    read_only image2d_t imgY,
    read_only image2d_t imgUV,
    __global unsigned char* pBGR,
   int bgrStep,
   int cols,
   int rows)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    // each iteration computes 2*2=4 pixels
    int x2 = x*2;
    int y2 = y*2;

    if (x2 + 1 < cols) {
        if (y2 + 1 < rows) {
            __global uchar *pDstRow1 = pBGR + mad24(y2, bgrStep, mad24(x2, NCHANNELS, 0));
            __global uchar *pDstRow2 = pDstRow1 + bgrStep;

            float4 Y1 = read_imagef(imgY, (int2)(x2 + 0, y2 + 0));
            float4 Y2 = read_imagef(imgY, (int2)(x2 + 1, y2 + 0));
            float4 Y3 = read_imagef(imgY, (int2)(x2 + 0, y2 + 1));
            float4 Y4 = read_imagef(imgY, (int2)(x2 + 1, y2 + 1));
            float4 Y = (float4)(Y1.x, Y2.x, Y3.x, Y4.x);

            float4 UV = read_imagef(imgUV, (int2)(x, y)) - d2;

            __constant float *coeffs = c_YUV2RGBCoeffs_420;

            Y = max(0.f, Y - d1) * coeffs[0];

            float ruv = fma(coeffs[4], UV.y, 0.0f);
            float guv = fma(coeffs[3], UV.y, fma(coeffs[2], UV.x, 0.0f));
            float buv = fma(coeffs[1], UV.x, 0.0f);

            float4 R = (Y + ruv) * CV_8U_MAX;
            float4 G = (Y + guv) * CV_8U_MAX;
            float4 B = (Y + buv) * CV_8U_MAX;

            pDstRow1[0*NCHANNELS + 0] = convert_uchar_sat(B.x);
            pDstRow1[0*NCHANNELS + 1] = convert_uchar_sat(G.x);
            pDstRow1[0*NCHANNELS + 2] = convert_uchar_sat(R.x);

            pDstRow1[1*NCHANNELS + 0] = convert_uchar_sat(B.y);
            pDstRow1[1*NCHANNELS + 1] = convert_uchar_sat(G.y);
            pDstRow1[1*NCHANNELS + 2] = convert_uchar_sat(R.y);

            pDstRow2[0*NCHANNELS + 0] = convert_uchar_sat(B.z);
            pDstRow2[0*NCHANNELS + 1] = convert_uchar_sat(G.z);
            pDstRow2[0*NCHANNELS + 2] = convert_uchar_sat(R.z);

            pDstRow2[1*NCHANNELS + 0] = convert_uchar_sat(B.w);
            pDstRow2[1*NCHANNELS + 1] = convert_uchar_sat(G.w);
            pDstRow2[1*NCHANNELS + 2] = convert_uchar_sat(R.w);
        }
    }
}


static
__constant float c_RGB2YUVCoeffs_420[8] =
{
     0.256999969f,  0.50399971f,   0.09799957f,   -0.1479988098f,
    -0.2909994125f, 0.438999176f, -0.3679990768f, -0.0709991455f
};


__kernel
void BGR2YUV_NV12_8u(
    __global unsigned char* pBGR,
    int bgrStep,
    int cols,
    int rows,
    write_only image2d_t imgY,
    write_only image2d_t imgUV)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    // each iteration computes 2*2=4 pixels
    int x2 = x*2;
    int y2 = y*2;

    if (x2 + 1 < cols)
    {
        if (y2 + 1 < rows)
        {
            __global const uchar* pSrcRow1 = pBGR + mad24(y2, bgrStep, mad24(x2, NCHANNELS, 0));
            __global const uchar* pSrcRow2 = pSrcRow1 + bgrStep;

            float4 src_pix1 = convert_float4(vload4(0, pSrcRow1 + 0*NCHANNELS)) * CV_8U_SCALE;
            float4 src_pix2 = convert_float4(vload4(0, pSrcRow1 + 1*NCHANNELS)) * CV_8U_SCALE;
            float4 src_pix3 = convert_float4(vload4(0, pSrcRow2 + 0*NCHANNELS)) * CV_8U_SCALE;
            float4 src_pix4 = convert_float4(vload4(0, pSrcRow2 + 1*NCHANNELS)) * CV_8U_SCALE;

            __constant float* coeffs = c_RGB2YUVCoeffs_420;

            float Y1 = fma(coeffs[0], src_pix1.z, fma(coeffs[1], src_pix1.y, fma(coeffs[2], src_pix1.x, d1)));
            float Y2 = fma(coeffs[0], src_pix2.z, fma(coeffs[1], src_pix2.y, fma(coeffs[2], src_pix2.x, d1)));
            float Y3 = fma(coeffs[0], src_pix3.z, fma(coeffs[1], src_pix3.y, fma(coeffs[2], src_pix3.x, d1)));
            float Y4 = fma(coeffs[0], src_pix4.z, fma(coeffs[1], src_pix4.y, fma(coeffs[2], src_pix4.x, d1)));

            float4 UV;
            UV.x = fma(coeffs[3], src_pix1.z, fma(coeffs[4], src_pix1.y, fma(coeffs[5], src_pix1.x, d2)));
            UV.y = fma(coeffs[5], src_pix1.z, fma(coeffs[6], src_pix1.y, fma(coeffs[7], src_pix1.x, d2)));

            write_imagef(imgY, (int2)(x2+0, y2+0), Y1);
            write_imagef(imgY, (int2)(x2+1, y2+0), Y2);
            write_imagef(imgY, (int2)(x2+0, y2+1), Y3);
            write_imagef(imgY, (int2)(x2+1, y2+1), Y4);

            write_imagef(imgUV, (int2)(x, y), UV);
        }
    }
}
