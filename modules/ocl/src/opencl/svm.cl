//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2013, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Erping Pang, erping@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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
//
#if defined (DOUBLE_SUPPORT)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#define TYPE double
#else
#define TYPE float
#endif
#if defined ADDEXP
#define EXP(X) exp(X)
#else
#define EXP(X) X
#endif
#if defined ADDPOW
#define POW(X,Y) pow(fabs(X),(Y))
#else
#define POW(X,Y) X
#endif
#define FLT_MAX   3.402823466e+38F
#define MAX_VAL   (FLT_MAX*1e-3)

__kernel void svm_linear(__global float* src, int src_step, __global float* src2, int src2_step, __global TYPE* dst, int dst_step, int src_rows, int src2_cols,
                         int width, TYPE alpha, TYPE beta)
{
    const int  col = get_global_id(0);
    const int  row = get_global_id(1);

    if(row < src_rows && col < src2_cols)
    {
        int t = 0;
        TYPE temp = 0.0;
        for(t = 0; t < width - 16; t += 16)
        {
            float16 t0 = vload16(0, src + row * src_step + t);
            float16 t1 = vload16(0, src2 + col * src2_step + t);
            t0 *= t1;
            temp += t0.s0 + t0.s1 + t0.s2 + t0.s3 + t0.s4 + t0.s5 + t0.s6 + t0.s7 +
                    t0.s8 + t0.s9 + t0.sa + t0.sb + t0.sc + t0.sd + t0.se + t0.sf;
        }
        for(; t < width; t++)
        {
            temp += src[row * src_step + t] * src2[col * src2_step + t];
        }

        TYPE temp1 = (TYPE) (temp * alpha + beta);

        if( temp1 > MAX_VAL )
        {
            dst[row * dst_step + col] = MAX_VAL;
        }
        else
        {
            dst[row * dst_step + col] = temp1;
        }

    }

}
__kernel void svm_sigmod(__global float* src, int src_step, __global float* src2, int src2_step, __global TYPE* dst, int dst_step, int src_rows, int src2_cols,
                         int width, TYPE alpha, TYPE beta)
{
    const int  col = get_global_id(0);
    const int  row = get_global_id(1);

    if(row < src_rows && col < src2_cols)
    {
        int t = 0;
        TYPE temp = 0.0;
        for(t = 0; t < width - 16; t += 16)
        {
            float16 t0 = vload16(0, src + row * src_step + t);
            float16 t1 = vload16(0, src2 + col * src2_step + t);
            t0 *= t1;
            temp += t0.s0 + t0.s1 + t0.s2 + t0.s3 + t0.s4 + t0.s5 + t0.s6 + t0.s7 +
                    t0.s8 + t0.s9 + t0.sa + t0.sb + t0.sc + t0.sd + t0.se + t0.sf;
        }
        for(; t < width; t++)
        {
            temp += src[row * src_step + t] * src2[col * src2_step + t];
        }
        TYPE tp = (TYPE) (temp * alpha + beta);
        TYPE e = exp(-fabs(tp));
        TYPE temp1;
        if(tp > 0)
        {
            temp1 = (TYPE)((1. - e) / (1. + e));
        }
        else
        {
            temp1 = (TYPE)((e - 1.) / (e + 1.));
        }

        if( temp1 > MAX_VAL )
        {
            dst[row * dst_step + col] = MAX_VAL;
        }
        else
        {
            dst[row * dst_step + col] = temp1;
        }
    }

}
__kernel void svm_poly(__global float* src, int src_step, __global float* src2, int src2_step, __global TYPE* dst, int dst_step, int src_rows, int src2_cols,
                       int width, TYPE alpha, TYPE beta, TYPE degree)
{
    const int  col = get_global_id(0);
    const int  row = get_global_id(1);

    if(row < src_rows && col < src2_cols)
    {
        int t = 0;
        TYPE temp = 0.0;
        for(t = 0; t < width - 16; t += 16)
        {
            float16 t0 = vload16(0, src + row * src_step + t);
            float16 t1 = vload16(0, src2 + col * src2_step + t);
            t0 *= t1;
            temp += t0.s0 + t0.s1 + t0.s2 + t0.s3 + t0.s4 + t0.s5 + t0.s6 + t0.s7 +
                    t0.s8 + t0.s9 + t0.sa + t0.sb + t0.sc + t0.sd + t0.se + t0.sf;
        }
        for(; t < width; t++)
        {
            temp += src[row * src_step + t] * src2[col * src2_step + t];
        }
        TYPE temp1 = (TYPE)(POW((temp * alpha + beta), degree));

        if( temp1 > MAX_VAL )
        {
            dst[row * dst_step + col] = MAX_VAL;
        }
        else
        {
            dst[row * dst_step + col] = temp1;
        }
    }

}
__kernel void svm_rbf(__global float* src, int src_step, __global float* src2, int src2_step, __global TYPE* dst, int dst_step, int src_rows, int src2_cols,
                      int width, TYPE gamma)
{
    const int  col = get_global_id(0);
    const int  row = get_global_id(1);

    if(row < src_rows && col < src2_cols)
    {
        int t = 0;
        TYPE temp = 0.0;
        for(t = 0; t < width - 16; t += 16)
        {
            float16 t0 = vload16(0, src + row * src_step + t);
            float16 t1 = vload16(0, src2 + col * src2_step + t);
            t0 = (t0 - t1) * (t0 - t1);
            temp += t0.s0 + t0.s1 + t0.s2 + t0.s3 + t0.s4 + t0.s5 + t0.s6 + t0.s7 +
                    t0.s8 + t0.s9 + t0.sa + t0.sb + t0.sc + t0.sd + t0.se + t0.sf;
        }
        for(; t < width; t++)
        {
            temp += (src[row * src_step + t] - src2[col * src2_step + t]) * (src[row * src_step + t] - src2[col * src2_step + t]);
        }
        TYPE temp1 = EXP((TYPE)(temp * gamma));

        if( temp1 > MAX_VAL )
        {
            dst[row * dst_step + col] = MAX_VAL;
        }
        else
        {
            dst[row * dst_step + col] = temp1;
        }
    }
}