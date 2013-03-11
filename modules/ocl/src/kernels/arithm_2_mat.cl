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
//    Shengen Yan,yanshengen@gmail.com
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
//M*/

/**************************************PUBLICFUNC*************************************/
#if defined (DOUBLE_SUPPORT)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#endif

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics:enable
#define CV_PI   3.1415926535897932384626433832795

char round_char(double v){
    char v1=(char)v;
    return convert_char_sat(v+(v>=0 ? 0.5 : -0.5));
}
unsigned char round_uchar(double v){
    unsigned char v1=(unsigned char)v;
    return convert_uchar_sat(v+(v>=0 ? 0.5 : -0.5));
}
short round_short(double v){
    short v1=(short)v;
    return convert_short_sat(v+(v>=0 ? 0.5 : -0.5));
}
unsigned short round_ushort(double v){
    unsigned short v1=(unsigned short)v;
    return convert_ushort_sat(v+(v>=0 ? 0.5 : -0.5));
}
int round_int(double v){
    int v1=(int)v;
    return convert_int_sat(v+(v>=0 ? 0.5 : -0.5));
}

char round2_char(double v){
    char v1=(char)v;
    if((v-v1)==0.5&&v1%2==0)
        return v1;
    else
        return convert_char_sat(v+(v>=0 ? 0.5 : -0.5));
}
unsigned char round2_uchar(double v){
    unsigned char v1=(unsigned char)v;
    if((v-v1)==0.5&&v1%2==0)
        return v1;
    else
        return convert_uchar_sat(v+(v>=0 ? 0.5 : -0.5));
}
short round2_short(double v){
    short v1=(short)v;
    if((v-v1)==0.5&&v1%2==0)
        return v1;
    else
        return convert_short_sat(v+(v>=0 ? 0.5 : -0.5));
}
unsigned short round2_ushort(double v){
    unsigned short v1=(unsigned short)v;
    if((v-v1)==0.5&&v1%2==0)
        return v1;
    else
        return convert_ushort_sat(v+(v>=0 ? 0.5 : -0.5));
}
int round2_int(double v){
    int v1=(int)v;
    if((v-v1)==0.5&&v1%2==0)
        return v1;
    else
        return convert_int_sat(v+(v>=0 ? 0.5 : -0.5));
}

/*****************************************EXP***************************************/
__kernel void arithm_op_exp_5 (int rows,int cols,int srcStep,__global float *src1Mat,
                             __global float * dstMat,int channels)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x < cols && y < rows)
    {
        size_t idx = y * ( srcStep >> 2 ) + x;
        dstMat[idx] = (float)exp((float)src1Mat[idx]);
    }
}
__kernel void arithm_op_exp_6 (int rows,int cols,int srcStep,__global double *src1Mat,
                             __global double * dstMat,int channels)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x < cols && y < rows)
    {
        size_t idx = y * ( srcStep >> 3 ) + x;
        dstMat[idx] = exp(src1Mat[idx]);
    }
}

/*****************************************LOG***************************************/
__kernel void arithm_op_log_5 (int rows,int cols,int srcStep,__global float *src1Mat,
                             __global float * dstMat,int channels)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x < cols && y < rows)
    {
        size_t idx = y * ( srcStep >> 2 ) + x;
        dstMat[idx] =(float) log((float)src1Mat[idx]);
    }
}
__kernel void arithm_op_log_6 (int rows,int cols,int srcStep,__global double *src1Mat,
                             __global double * dstMat,int channels)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x < cols && y < rows)
    {
        size_t idx = y * ( srcStep >> 3 ) + x;
        dstMat[idx] = log(src1Mat[idx]);
    }
}
