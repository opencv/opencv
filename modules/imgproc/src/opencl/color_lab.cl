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
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if SRC_DEPTH == 0
    #define DATA_TYPE uchar
    #define MAX_NUM  255
    #define HALF_MAX_NUM 128
    #define COEFF_TYPE int
    #define SAT_CAST(num) convert_uchar_sat(num)
    #define DEPTH_0
#elif SRC_DEPTH == 2
    #define DATA_TYPE ushort
    #define MAX_NUM  65535
    #define HALF_MAX_NUM 32768
    #define COEFF_TYPE int
    #define SAT_CAST(num) convert_ushort_sat(num)
    #define DEPTH_2
#elif SRC_DEPTH == 5
    #define DATA_TYPE float
    #define MAX_NUM  1.0f
    #define HALF_MAX_NUM 0.5f
    #define COEFF_TYPE float
    #define SAT_CAST(num) (num)
    #define DEPTH_5
#else
    #error "invalid depth: should be 0 (CV_8U), 2 (CV_16U) or 5 (CV_32F)"
#endif

#define CV_DESCALE(x,n) (((x) + (1 << ((n)-1))) >> (n))

enum
{
    xyz_shift  = 12,
};

#define scnbytes ((int)sizeof(DATA_TYPE)*SCN)
#define dcnbytes ((int)sizeof(DATA_TYPE)*DCN)

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define DATA_TYPE_4 CAT(DATA_TYPE, 4)
#define DATA_TYPE_3 CAT(DATA_TYPE, 3)

///////////////////////////////////// RGB <-> XYZ //////////////////////////////////////

__kernel void RGB2XYZ(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset,
                      int rows, int cols, __constant COEFF_TYPE * coeffs)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1) * PIX_PER_WI_Y;

    if (dx < cols)
    {
        int src_index = mad24(dy, src_step, mad24(dx, scnbytes, src_offset));
        int dst_index = mad24(dy, dst_step, mad24(dx, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (dy < rows)
            {
                __global const DATA_TYPE * src = (__global const DATA_TYPE *)(srcptr + src_index);
                __global DATA_TYPE * dst = (__global DATA_TYPE *)(dstptr + dst_index);

                DATA_TYPE_4 src_pix = vload4(0, src);
                DATA_TYPE r = src_pix.x, g = src_pix.y, b = src_pix.z;

#ifdef DEPTH_5
                float x = fma(r, coeffs[0], fma(g, coeffs[1], b * coeffs[2]));
                float y = fma(r, coeffs[3], fma(g, coeffs[4], b * coeffs[5]));
                float z = fma(r, coeffs[6], fma(g, coeffs[7], b * coeffs[8]));
#else
                int x = CV_DESCALE(mad24(r, coeffs[0], mad24(g, coeffs[1], b * coeffs[2])), xyz_shift);
                int y = CV_DESCALE(mad24(r, coeffs[3], mad24(g, coeffs[4], b * coeffs[5])), xyz_shift);
                int z = CV_DESCALE(mad24(r, coeffs[6], mad24(g, coeffs[7], b * coeffs[8])), xyz_shift);
#endif
                dst[0] = SAT_CAST(x);
                dst[1] = SAT_CAST(y);
                dst[2] = SAT_CAST(z);

                ++dy;
                dst_index += dst_step;
                src_index += src_step;
            }
        }
    }
}

__kernel void XYZ2RGB(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset,
                      int rows, int cols, __constant COEFF_TYPE * coeffs)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1) * PIX_PER_WI_Y;

    if (dx < cols)
    {
        int src_index = mad24(dy, src_step, mad24(dx, scnbytes, src_offset));
        int dst_index = mad24(dy, dst_step, mad24(dx, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (dy < rows)
            {
                __global const DATA_TYPE * src = (__global const DATA_TYPE *)(srcptr + src_index);
                __global DATA_TYPE * dst = (__global DATA_TYPE *)(dstptr + dst_index);

                DATA_TYPE_4 src_pix = vload4(0, src);
                DATA_TYPE x = src_pix.x, y = src_pix.y, z = src_pix.z;

#ifdef DEPTH_5
                float b = fma(x, coeffs[0], fma(y, coeffs[1], z * coeffs[2]));
                float g = fma(x, coeffs[3], fma(y, coeffs[4], z * coeffs[5]));
                float r = fma(x, coeffs[6], fma(y, coeffs[7], z * coeffs[8]));
#else
                int b = CV_DESCALE(mad24(x, coeffs[0], mad24(y, coeffs[1], z * coeffs[2])), xyz_shift);
                int g = CV_DESCALE(mad24(x, coeffs[3], mad24(y, coeffs[4], z * coeffs[5])), xyz_shift);
                int r = CV_DESCALE(mad24(x, coeffs[6], mad24(y, coeffs[7], z * coeffs[8])), xyz_shift);
#endif

                DATA_TYPE dst0 = SAT_CAST(b);
                DATA_TYPE dst1 = SAT_CAST(g);
                DATA_TYPE dst2 = SAT_CAST(r);
#if DCN == 3 || defined DEPTH_5
                dst[0] = dst0;
                dst[1] = dst1;
                dst[2] = dst2;
#if DCN == 4
                dst[3] = MAX_NUM;
#endif
#else
                *(__global DATA_TYPE_4 *)dst = (DATA_TYPE_4)(dst0, dst1, dst2, MAX_NUM);
#endif

                ++dy;
                dst_index += dst_step;
                src_index += src_step;
            }
        }
    }
}

/////////////////////////////////// [l|s]RGB <-> Lab ///////////////////////////

#define lab_shift xyz_shift
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)
#define GAMMA_TAB_SIZE 1024
#define GammaTabScale (float)GAMMA_TAB_SIZE

inline float splineInterpolate(float x, __global const float * tab, int n)
{
    int ix = clamp(convert_int_sat_rtn(x), 0, n-1);
    x -= ix;
    tab += ix << 2;
    return fma(fma(fma(tab[3], x, tab[2]), x, tab[1]), x, tab[0]);
}

#ifdef DEPTH_0

__kernel void BGR2Lab(__global const uchar * src, int src_step, int src_offset,
                      __global uchar * dst, int dst_step, int dst_offset, int rows, int cols,
                      __global const ushort * gammaTab, __global ushort * LabCbrtTab_b,
                      __constant int * coeffs, int Lscale, int Lshift)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const uchar* src_ptr = src + src_index;
                __global uchar* dst_ptr = dst + dst_index;
                uchar4 src_pix = vload4(0, src_ptr);

                int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
                    C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
                    C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

                int R = gammaTab[src_pix.x], G = gammaTab[src_pix.y], B = gammaTab[src_pix.z];
                int fX = LabCbrtTab_b[CV_DESCALE(mad24(R, C0, mad24(G, C1, B*C2)), lab_shift)];
                int fY = LabCbrtTab_b[CV_DESCALE(mad24(R, C3, mad24(G, C4, B*C5)), lab_shift)];
                int fZ = LabCbrtTab_b[CV_DESCALE(mad24(R, C6, mad24(G, C7, B*C8)), lab_shift)];

                int L = CV_DESCALE( Lscale*fY + Lshift, lab_shift2 );
                int a = CV_DESCALE( mad24(500, fX - fY, 128*(1 << lab_shift2)), lab_shift2 );
                int b = CV_DESCALE( mad24(200, fY - fZ, 128*(1 << lab_shift2)), lab_shift2 );

                dst_ptr[0] = SAT_CAST(L);
                dst_ptr[1] = SAT_CAST(a);
                dst_ptr[2] = SAT_CAST(b);

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
        }
    }
}

#elif defined DEPTH_5

__kernel void BGR2Lab(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __constant float * coeffs, float _1_3, float _a)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const float * src = (__global const float *)(srcptr + src_index);
                __global float * dst = (__global float *)(dstptr + dst_index);
                float4 src_pix = vload4(0, src);

                float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
                      C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
                      C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

                float R = clamp(src_pix.x, 0.0f, 1.0f);
                float G = clamp(src_pix.y, 0.0f, 1.0f);
                float B = clamp(src_pix.z, 0.0f, 1.0f);

#ifdef SRGB
                R = splineInterpolate(R * GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G * GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B * GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif

                // 7.787f = (29/3)^3/(29*4), 0.008856f = (6/29)^3, 903.3 = (29/3)^3
                float X = fma(R, C0, fma(G, C1, B*C2));
                float Y = fma(R, C3, fma(G, C4, B*C5));
                float Z = fma(R, C6, fma(G, C7, B*C8));

                float FX = X > 0.008856f ? rootn(X, 3) : fma(7.787f, X, _a);
                float FY = Y > 0.008856f ? rootn(Y, 3) : fma(7.787f, Y, _a);
                float FZ = Z > 0.008856f ? rootn(Z, 3) : fma(7.787f, Z, _a);

                float L = Y > 0.008856f ? fma(116.f, FY, -16.f) : (903.3f * Y);
                float a = 500.f * (FX - FY);
                float b = 200.f * (FY - FZ);

                dst[0] = L;
                dst[1] = a;
                dst[2] = b;

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
        }
    }
}

#endif

inline void Lab2BGR_f(const float * srcbuf, float * dstbuf,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __constant float * coeffs, float lThresh, float fThresh)
{
    float li = srcbuf[0], ai = srcbuf[1], bi = srcbuf[2];

    float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
          C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
          C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

    float y, fy;
    // 903.3 = (29/3)^3, 7.787 = (29/3)^3/(29*4)
    if (li <= lThresh)
    {
        y = li / 903.3f;
        fy = fma(7.787f, y, 16.0f / 116.0f);
    }
    else
    {
        fy = (li + 16.0f) / 116.0f;
        y = fy * fy * fy;
    }

    float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };

    #pragma unroll
    for (int j = 0; j < 2; j++)
        if (fxz[j] <= fThresh)
            fxz[j] = (fxz[j] - 16.0f / 116.0f) / 7.787f;
        else
            fxz[j] = fxz[j] * fxz[j] * fxz[j];

    float x = fxz[0], z = fxz[1];
    float ro = clamp(fma(C0, x, fma(C1, y, C2 * z)), 0.0f, 1.0f);
    float go = clamp(fma(C3, x, fma(C4, y, C5 * z)), 0.0f, 1.0f);
    float bo = clamp(fma(C6, x, fma(C7, y, C8 * z)), 0.0f, 1.0f);

#ifdef SRGB
    ro = splineInterpolate(ro * GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
    go = splineInterpolate(go * GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
    bo = splineInterpolate(bo * GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif

    dstbuf[0] = ro, dstbuf[1] = go, dstbuf[2] = bo;
}

#ifdef DEPTH_0

__kernel void Lab2BGR(__global const uchar * src, int src_step, int src_offset,
                      __global uchar * dst, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __constant float * coeffs, float lThresh, float fThresh)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const uchar* src_ptr = src + src_index;
                __global uchar * dst_ptr = dst + dst_index;
                uchar4 src_pix = vload4(0, src_ptr);

                float srcbuf[3], dstbuf[3];
                srcbuf[0] = src_pix.x*(100.f/255.f);
                srcbuf[1] = convert_float(src_pix.y - 128);
                srcbuf[2] = convert_float(src_pix.z - 128);

                Lab2BGR_f(&srcbuf[0], &dstbuf[0],
#ifdef SRGB
                    gammaTab,
#endif
                    coeffs, lThresh, fThresh);

#if DCN == 3
                dst_ptr[0] = SAT_CAST(dstbuf[0] * 255.0f);
                dst_ptr[1] = SAT_CAST(dstbuf[1] * 255.0f);
                dst_ptr[2] = SAT_CAST(dstbuf[2] * 255.0f);
#else
                *(__global uchar4 *)dst_ptr = (uchar4)(SAT_CAST(dstbuf[0] * 255.0f),
                    SAT_CAST(dstbuf[1] * 255.0f), SAT_CAST(dstbuf[2] * 255.0f), MAX_NUM);
#endif
                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
        }
    }
}

#elif defined DEPTH_5

__kernel void Lab2BGR(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __constant float * coeffs, float lThresh, float fThresh)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const float * src = (__global const float *)(srcptr + src_index);
                __global float * dst = (__global float *)(dstptr + dst_index);
                float4 src_pix = vload4(0, src);

                float srcbuf[3], dstbuf[3];
                srcbuf[0] = src_pix.x, srcbuf[1] = src_pix.y, srcbuf[2] = src_pix.z;

                Lab2BGR_f(&srcbuf[0], &dstbuf[0],
#ifdef SRGB
                    gammaTab,
#endif
                    coeffs, lThresh, fThresh);

                dst[0] = dstbuf[0], dst[1] = dstbuf[1], dst[2] = dstbuf[2];
#if DCN == 4
                dst[3] = MAX_NUM;
#endif
                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
        }
    }
}

#endif

/////////////////////////////////// [l|s]RGB <-> Luv ///////////////////////////

#define LAB_CBRT_TAB_SIZE 1024
#define LAB_CBRT_TAB_SIZE_B (256*3/2*(1<<gamma_shift))

__constant float LabCbrtTabScale = LAB_CBRT_TAB_SIZE/1.5f;

#ifdef DEPTH_5

__kernel void BGR2Luv(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __global const float * LabCbrtTab, __constant float * coeffs, float _un, float _vn)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
            if (y < rows)
            {
                __global const float * src = (__global const float *)(srcptr + src_index);
                __global float * dst = (__global float *)(dstptr + dst_index);

                float R = src[0], G = src[1], B = src[2];

                R = clamp(R, 0.f, 1.f);
                G = clamp(G, 0.f, 1.f);
                B = clamp(B, 0.f, 1.f);

#ifdef SRGB
                R = splineInterpolate(R*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif
                float X = fma(R, coeffs[0], fma(G, coeffs[1], B*coeffs[2]));
                float Y = fma(R, coeffs[3], fma(G, coeffs[4], B*coeffs[5]));
                float Z = fma(R, coeffs[6], fma(G, coeffs[7], B*coeffs[8]));

                float L = splineInterpolate(Y*LabCbrtTabScale, LabCbrtTab, LAB_CBRT_TAB_SIZE);
                L = fma(116.f, L, -16.f);

                float d = 52.0f / fmax(fma(15.0f, Y, fma(3.0f, Z, X)), FLT_EPSILON);
                float u = L*fma(X, d, -_un);
                float v = L*fma(2.25f, Y*d, -_vn);

                dst[0] = L;
                dst[1] = u;
                dst[2] = v;

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
    }
}

#elif defined DEPTH_0

__kernel void BGR2Luv(__global const uchar * src, int src_step, int src_offset,
                      __global uchar * dst, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __global const float * LabCbrtTab, __constant float * coeffs, float _un, float _vn)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        src += mad24(y, src_step, mad24(x, scnbytes, src_offset));
        dst += mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
            if (y < rows)
            {
                float scale = 1.0f / 255.0f;
                float R = src[0]*scale, G = src[1]*scale, B = src[2]*scale;

#ifdef SRGB
                R = splineInterpolate(R*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif
                float X = fma(R, coeffs[0], fma(G, coeffs[1], B*coeffs[2]));
                float Y = fma(R, coeffs[3], fma(G, coeffs[4], B*coeffs[5]));
                float Z = fma(R, coeffs[6], fma(G, coeffs[7], B*coeffs[8]));

                float L = splineInterpolate(Y*LabCbrtTabScale, LabCbrtTab, LAB_CBRT_TAB_SIZE);
                L = 116.f*L - 16.f;

                float d = (4*13) / fmax(fma(15.0f, Y, fma(3.0f, Z, X)), FLT_EPSILON);
                float u = L*(X*d - _un);
                float v = L*fma(2.25f, Y*d, -_vn);

                dst[0] = SAT_CAST(L * 2.55f);
                //0.72033 = 255/(220+134), 96.525 = 134*255/(220+134)
                dst[1] = SAT_CAST(fma(u, 0.72033898305084743f, 96.525423728813564f));
                //0.9732 = 255/(140+122), 136.259 = 140*255/(140+122)
                dst[2] = SAT_CAST(fma(v, 0.9732824427480916f, 136.259541984732824f));

                ++y;
                dst += dst_step;
                src += src_step;
            }
    }
}

#endif

#ifdef DEPTH_5

__kernel void Luv2BGR(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __constant float * coeffs, float _un, float _vn)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
            if (y < rows)
            {
                __global const float * src = (__global const float *)(srcptr + src_index);
                __global float * dst = (__global float *)(dstptr + dst_index);

                float L = src[0], u = src[1], v = src[2], X, Y, Z;
                if(L >= 8)
                {
                    Y = fma(L, 1.f/116.f, 16.f/116.f);
                    Y = Y*Y*Y;
                }
                else
                {
                    Y = L * (1.0f/903.3f); // L*(3./29.)^3
                }
                float up = 3.f*fma(L, _un, u);
                float vp = 0.25f/fma(L, _vn, v);
                vp = clamp(vp, -0.25f, 0.25f);
                X = 3.f*Y*up*vp;
                Z = Y*fma(fma(12.f*13.f, L, -up), vp, -5.f);

                float R = fma(X, coeffs[0], fma(Y, coeffs[1], Z * coeffs[2]));
                float G = fma(X, coeffs[3], fma(Y, coeffs[4], Z * coeffs[5]));
                float B = fma(X, coeffs[6], fma(Y, coeffs[7], Z * coeffs[8]));

                R = clamp(R, 0.f, 1.f);
                G = clamp(G, 0.f, 1.f);
                B = clamp(B, 0.f, 1.f);

#ifdef SRGB
                R = splineInterpolate(R*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif

                dst[0] = R;
                dst[1] = G;
                dst[2] = B;
#if DCN == 4
                dst[3] = MAX_NUM;
#endif
                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
    }
}

#elif defined DEPTH_0

__kernel void Luv2BGR(__global const uchar * src, int src_step, int src_offset,
                      __global uchar * dst, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __constant float * coeffs, float _un, float _vn)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        src += mad24(y, src_step, mad24(x, scnbytes, src_offset));
        dst += mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
            if (y < rows)
            {
                float d, X, Y, Z;
                float L = src[0]*(100.f/255.f);
                // 1.388235294117647 = (220+134)/255
                float u = fma(convert_float(src[1]), 1.388235294117647f, -134.f);
                // 1.027450980392157 = (140+122)/255
                float v = fma(convert_float(src[2]), 1.027450980392157f, - 140.f);
                if(L >= 8)
                {
                    Y = fma(L, 1.f/116.f, 16.f/116.f);
                    Y = Y*Y*Y;
                }
                else
                {
                    Y = L * (1.0f/903.3f); // L*(3./29.)^3
                }
                float up = 3.f*fma(L, _un, u);
                float vp = 0.25f/fma(L, _vn, v);
                vp = clamp(vp, -0.25f, 0.25f);
                X = 3.f*Y*up*vp;
                Z = Y*fma(fma(12.f*13.f, L, -up), vp, -5.f);

                //limit X, Y, Z to [0, 2] to fit white point
                X = clamp(X, 0.f, 2.f); Z = clamp(Z, 0.f, 2.f);

                float R = fma(X, coeffs[0], fma(Y, coeffs[1], Z * coeffs[2]));
                float G = fma(X, coeffs[3], fma(Y, coeffs[4], Z * coeffs[5]));
                float B = fma(X, coeffs[6], fma(Y, coeffs[7], Z * coeffs[8]));

                R = clamp(R, 0.f, 1.f);
                G = clamp(G, 0.f, 1.f);
                B = clamp(B, 0.f, 1.f);

#ifdef SRGB
                R = splineInterpolate(R*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif

                uchar dst0 = SAT_CAST(R * 255.0f);
                uchar dst1 = SAT_CAST(G * 255.0f);
                uchar dst2 = SAT_CAST(B * 255.0f);

#if DCN == 4
                *(__global uchar4 *)dst = (uchar4)(dst0, dst1, dst2, MAX_NUM);
#else
                dst[0] = dst0;
                dst[1] = dst1;
                dst[2] = dst2;
#endif

                ++y;
                dst += dst_step;
                src += src_step;
            }
    }
}

#endif
