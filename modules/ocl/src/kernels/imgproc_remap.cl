
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
//    Wu Zailong, bullet@yeah.net
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other GpuMaterials provided with the distribution.
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
#pragma OPENCL EXTENSION cl_amd_printf : enable

#if defined DOUBLE_SUPPORT
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif

__kernel void remapNNSConstant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , double4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    /*
    if(x < dst_cols && y < dst_rows)
    {
        int dstIdx = y * dst_step + x + dst_offset;
        int map1Idx = y * (map1_step>>2) + x + (map1_offset>>2) - (map1_offset & 1);
        short2 map1_data = *(map1 + map1Idx);
        int srcIdx = map1_data.y*src_step+map1_data.x + src_offset;       
        uchar src_data = *(src +srcIdx);      
        uchar dst_data = src_data; 
        *(dst +dstIdx)=(map1_data.x >= map1_cols || map1_data.y >= map1_rows) ? val : dst_data;
    }
    */
    
    int gx = (x << 2) - (dst_offset&3);
    int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

    uchar4 nval =convert_uchar4(nVal);
    char val = nval.s0;

    x = x << 2;

    int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&3);

    int map1Start = y * map1_step + (x << 2) + map1_offset - ((dst_offset & 3) << 2);
    short8 map1_data;

     map1_data.s01 = *((__global short2 *)((__global char*)map1 + map1Start));
     map1_data.s23 = *((__global short2 *)((__global char*)map1 + map1Start + 4));
     map1_data.s45 = *((__global short2 *)((__global char*)map1 + map1Start + 8));
     map1_data.s67 = *((__global short2 *)((__global char*)map1 + map1Start + 12));
    
    int4 srcIdx ;
    srcIdx.s0 = map1_data.s1 * src_step + map1_data.s0 + src_offset;
    srcIdx.s1 = map1_data.s3 * src_step + map1_data.s2 + src_offset;
    srcIdx.s2 = map1_data.s5 * src_step + map1_data.s4 + src_offset;
    srcIdx.s3 = map1_data.s7 * src_step + map1_data.s6 + src_offset;
    
        //uchar4 src_data = *(src + srcIdx);
    uchar4 src_data;
    src_data.s0 = *(src + srcIdx.s0);
    src_data.s1 = *(src + srcIdx.s1);
    src_data.s2 = *(src + srcIdx.s2);
    src_data.s3 = *(src + srcIdx.s3);

    uchar4 dst_data;
    dst_data.s0 = (map1_data.s0 >= src_cols || map1_data.s1 >= src_rows)? val : src_data.s0;
    dst_data.s1 = (map1_data.s2 >= src_cols || map1_data.s3 >= src_rows)? val : src_data.s1;
    dst_data.s2 = (map1_data.s4 >= src_cols || map1_data.s5 >= src_rows)? val : src_data.s2;
    dst_data.s3 = (map1_data.s6 >= src_cols || map1_data.s7 >= src_rows)? val : src_data.s3;
    
    __global uchar4* d = (__global uchar4 *)(dst + dstStart);

    uchar4 dVal = *d;      
    int4 con = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
    dst_data = (convert_uchar4(con) != 0) ? dst_data : dVal;

    *d = dst_data;
}

__kernel void remapNNSConstant_C2_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , double4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int gx = (x << 3) - (dst_offset&7);
    int8 Gx = (int8)(gx, gx+1, gx+2, gx+3, gx+4, gx+5, gx+6, gx+7);

    uchar4 nval =convert_uchar4(nVal);
    uchar2 val = nval.s01;//testing

    x = x << 3;

    int dstStart = (y * dst_step + x + dst_offset) - (dst_offset&7);

    int map1Start = y * map1_step + (x << 1) + map1_offset - (((dst_offset>>1) & 3) << 2);
    short8 map1_data;

     map1_data.s01 = *((__global short2 *)((__global char*)map1 + map1Start));
     map1_data.s23 = *((__global short2 *)((__global char*)map1 + map1Start + 4));
     map1_data.s45 = *((__global short2 *)((__global char*)map1 + map1Start + 8));
     map1_data.s67 = *((__global short2 *)((__global char*)map1 + map1Start + 12));
    
    int4 srcIdx ;
    srcIdx.s0 = map1_data.s1 * src_step + (map1_data.s0 << 1) + src_offset;
    srcIdx.s1 = map1_data.s3 * src_step + (map1_data.s2 << 1) + src_offset;
    srcIdx.s2 = map1_data.s5 * src_step + (map1_data.s4 << 1) + src_offset;
    srcIdx.s3 = map1_data.s7 * src_step + (map1_data.s6 << 1) + src_offset;
    
        //uchar4 src_data = *(src + srcIdx);
    uchar8 src_data;
    src_data.s01 = *((__global uchar2 *)((__global char*)src + srcIdx.s0));
    src_data.s23 = *((__global uchar2 *)((__global char*)src + srcIdx.s1));
    src_data.s45 = *((__global uchar2 *)((__global char*)src + srcIdx.s2));
    src_data.s67 = *((__global uchar2 *)((__global char*)src + srcIdx.s3));

    uchar8 dst_data;
    dst_data.s01 = (map1_data.s0 >= src_cols || map1_data.s1 >= src_rows) ? val : (convert_uchar2(src_data.s01));
    dst_data.s23 = (map1_data.s2 >= src_cols || map1_data.s3 >= src_rows) ? val : (convert_uchar2(src_data.s23));
    dst_data.s45 = (map1_data.s4 >= src_cols || map1_data.s5 >= src_rows) ? val : (convert_uchar2(src_data.s45));
    dst_data.s67 = (map1_data.s6 >= src_cols || map1_data.s7 >= src_rows) ? val : (convert_uchar2(src_data.s67));
    __global uchar8* d = (__global uchar8 *)(dst + dstStart);

    uchar8 dVal = *d;      
    int8 con = (Gx >= 0 && Gx < (dst_cols << 1) && y >= 0 && y < dst_rows);
    dst_data = (convert_uchar8(con) != 0) ? dst_data : dVal;
    *d = dst_data;
}
__kernel void remapNNSConstant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , double4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int gx = (x << 4) - (dst_offset&15);
    int16 Gx = (int16)(gx, gx+1, gx+2, gx+3, gx+4, gx+5, gx+6, gx+7, gx+8, gx+9, gx+10, gx+11, gx+12, gx+13, gx+14, gx+15);

    uchar4 nval =convert_uchar4(nVal);

    x = x << 4;

    int dstStart = (y * dst_step + x + dst_offset) - (dst_offset&15);

    int map1Start = y * map1_step + x + map1_offset - (((dst_offset>>2) & 3) << 2);
    short8 map1_data;

     map1_data.s01 = *((__global short2 *)((__global char*)map1 + map1Start));
     map1_data.s23 = *((__global short2 *)((__global char*)map1 + map1Start + 4));
     map1_data.s45 = *((__global short2 *)((__global char*)map1 + map1Start + 8));
     map1_data.s67 = *((__global short2 *)((__global char*)map1 + map1Start + 12));
    
    int4 srcIdx ;
    srcIdx.s0 = map1_data.s1 * src_step + (map1_data.s0 << 2) + src_offset;
    srcIdx.s1 = map1_data.s3 * src_step + (map1_data.s2 << 2) + src_offset;
    srcIdx.s2 = map1_data.s5 * src_step + (map1_data.s4 << 2) + src_offset;
    srcIdx.s3 = map1_data.s7 * src_step + (map1_data.s6 << 2) + src_offset;
    
  //  uchar16 src_data;
    uchar4 src_a, src_b, src_c, src_d;
    src_a = *((__global uchar4 *)((__global char*)src + srcIdx.s0));
    src_b = *((__global uchar4 *)((__global char*)src + srcIdx.s1));
    src_c = *((__global uchar4 *)((__global char*)src + srcIdx.s2));
    src_d = *((__global uchar4 *)((__global char*)src + srcIdx.s3));
  //  src_data = (uchar16)(src_a, src_b, src_c, src_d);
    uchar16 dst_data;
    uchar4 dst_a, dst_b, dst_c, dst_d;
    dst_a = (map1_data.s0 >= src_cols || map1_data.s1 >= src_rows) ? nval : src_a;
    dst_b = (map1_data.s2 >= src_cols || map1_data.s3 >= src_rows) ? nval : src_b;
    dst_c = (map1_data.s4 >= src_cols || map1_data.s5 >= src_rows) ? nval : src_c;
    dst_d = (map1_data.s6 >= src_cols || map1_data.s7 >= src_rows) ? nval : src_d;
    dst_data = (uchar16)(dst_a, dst_b, dst_c, dst_d);
    __global uchar16* d = (__global uchar16 *)(dst + dstStart);

    uchar16 dVal = *d;      
    int16 con = (Gx >= 0 && Gx < (dst_cols << 2) && y >= 0 && y < dst_rows);
    dst_data = (convert_uchar16(con) != 0) ? dst_data : dVal;

    *d = dst_data;
}

__kernel void remapNNFConstant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , double4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    int gx = (x << 2) - (dst_offset&3);
    int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

    uchar4 nval =convert_uchar4_sat_rte(nVal);
    char val = nval.s0;

    x = x << 2;

    int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&3);

    int map1Start = y * map1_step + (x << 3) + map1_offset - ((dst_offset & 3) << 3);
    float8 map1_data;

    map1_data = *((__global float8 *)((__global char*)map1 + map1Start));
 /*   map1_data.s01 = *((__global float2 *)((__global char*)map1 + map1Start));
    map1_data.s23 = *((__global float2 *)((__global char*)map1 + map1Start + 8));
    map1_data.s45 = *((__global float2 *)((__global char*)map1 + map1Start + 16));
    map1_data.s67 = *((__global float2 *)((__global char*)map1 + map1Start + 24));
*/
    int8 map1_dataZ;

    map1_dataZ = convert_int8_sat_rte(map1_data);

    int4 srcIdx ;
    srcIdx.s0 = map1_dataZ.s1 * src_step + map1_dataZ.s0 + src_offset;
    srcIdx.s1 = map1_dataZ.s3 * src_step + map1_dataZ.s2 + src_offset;
    srcIdx.s2 = map1_dataZ.s5 * src_step + map1_dataZ.s4 + src_offset;
    srcIdx.s3 = map1_dataZ.s7 * src_step + map1_dataZ.s6 + src_offset;
    
        //uchar4 src_data = *(src + srcIdx);
    uchar4 src_data;
    src_data.s0 = *(src + srcIdx.s0);
    src_data.s1 = *(src + srcIdx.s1);
    src_data.s2 = *(src + srcIdx.s2);
    src_data.s3 = *(src + srcIdx.s3);

    uchar4 dst_data;
    dst_data.s0 = (map1_dataZ.s0 >= src_cols || map1_dataZ.s1 >= src_rows)? val : src_data.s0;
    dst_data.s1 = (map1_dataZ.s2 >= src_cols || map1_dataZ.s3 >= src_rows)? val : src_data.s1;
    dst_data.s2 = (map1_dataZ.s4 >= src_cols || map1_dataZ.s5 >= src_rows)? val : src_data.s2;
    dst_data.s3 = (map1_dataZ.s6 >= src_cols || map1_dataZ.s7 >= src_rows)? val : src_data.s3;
    
    __global uchar4* d = (__global uchar4 *)(dst + dstStart);

    uchar4 dVal = *d;      
    int4 con = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
    dst_data = (convert_uchar4(con) != 0) ? dst_data : dVal;

    *d = dst_data;
}


__kernel void remapLNFConstant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , double4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    int gx = (x << 2) - (dst_offset&3);
    int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

    uchar4 nval =convert_uchar4(nVal);
    uchar val = nval.s0;
  
    x = x << 2;

    int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&3);

    int map1Start = y * map1_step + (x << 3) + map1_offset - ((dst_offset & 3) << 3);
    float8 map1_data;

    map1_data = *((__global float8 *)((__global char*)map1 + map1Start));
    int8 map1_dataD = convert_int8(map1_data);
    float8 temp = map1_data - convert_float8(map1_dataD);

    float4 u = temp.even;
    float4 v = temp.odd;
    float4 ud = 1.0 - u;
    float4 vd = 1.0 - v;
    //float8 map1_dataU = map1_dataD + 1;

    int4 map1_dataDx = map1_dataD.even;
    int4 map1_dataDy = map1_dataD.odd;
    int4 map1_dataDx1 = map1_dataDx + 1;
    int4 map1_dataDy1 = map1_dataDy + 1;

    int4 src_StartU = map1_dataDy * src_step + map1_dataDx + src_offset;
    int4 src_StartD = src_StartU + src_step;
    int4 src_StartU1 = src_StartU + 1;
    int4 src_StartD1 = src_StartD + 1;

    uchar4 a, b, c, d;
    a.x = *(src_StartU.x + src);
    a.y = *(src_StartU.y + src);
    a.z = *(src_StartU.z + src);
    a.w = *(src_StartU.w + src);
    
    b.x = *(src_StartU1.x + src);
    b.y = *(src_StartU1.y + src);
    b.z = *(src_StartU1.z + src);
    b.w = *(src_StartU1.w + src);

    c.x = *(src_StartD.x + src);
    c.y = *(src_StartD.y + src);
    c.z = *(src_StartD.z + src);
    c.w = *(src_StartD.w + src);

    d.x = *(src_StartD1.x + src);
    d.y = *(src_StartD1.y + src);
    d.z = *(src_StartD1.z + src);
    d.w = *(src_StartD1.w + src);
    int4 ac =(map1_dataDx >= src_cols || map1_dataDy >= src_rows || map1_dataDy< 0 || map1_dataDy < 0);
    int4 bc =(map1_dataDx1 >= src_cols || map1_dataDy >= src_rows || map1_dataDx1 < 0 || map1_dataDy < 0);
    int4 cc =(map1_dataDx >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDx < 0);
    int4 dc =(map1_dataDx1 >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDy1 < 0);
    a = (convert_uchar4(ac) == 0)? a : val;
    b = (convert_uchar4(bc) == 0)? b : val;
    c = (convert_uchar4(cc) == 0)? c : val;
    d = (convert_uchar4(dc) == 0)? d : val;

    uchar4 dst_data = convert_uchar4_sat_rte((convert_float4(a))* ud * vd +(convert_float4(b))* u * vd + (convert_float4(c))* ud * v + (convert_float4(d)) * u * v );
    
    __global uchar4* D = (__global uchar4 *)(dst + dstStart);

    uchar4 dVal = *D;      
    int4 con = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
    dst_data = (convert_uchar4(con) != 0) ? dst_data : dVal;

    *D = dst_data;
}


__kernel void remapLNFConstant_C2_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , double4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    int gx = (x << 3) - (dst_offset&7);
    int8 Gx = (int8)(gx, gx+1, gx+2, gx+3, gx+4, gx+5, gx+6, gx+7);

    uchar4 nval =convert_uchar4(nVal);
    uchar8 val = (uchar8)(nval.s01, nval.s01, nval.s01, nval.s01);
  
    x = x << 3;

    int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&7);

    int map1Start = y * map1_step + (x << 2) + map1_offset - (((dst_offset>>1) & 3) << 3);
    float8 map1_data;

    map1_data = *((__global float8 *)((__global char*)map1 + map1Start));
    int8 map1_dataD = convert_int8(map1_data);
    float8 temp = map1_data - convert_float8(map1_dataD);

    float4 U = temp.even;
    float4 V = temp.odd;
    float4 UD = 1.0 - U;
    float4 VD = 1.0 - V;

    float8 u, v, ud, vd;
    u = (float8)(U.x, U.x, U.y, U.y, U.z, U.z, U.w, U.w);
    v = (float8)(V.x, V.x, V.y, V.y, V.z, V.z, V.w, V.w);
    ud = (float8)(UD.x, UD.x, UD.y, UD.y, UD.z, UD.z, UD.w, UD.w);
    vd = (float8)(VD.x, VD.x, VD.y, VD.y, VD.z, VD.z, VD.w, VD.w);

    //float8 map1_dataU = map1_dataD + 1;

    int4 map1_dataDx = map1_dataD.even;
    int4 map1_dataDy = map1_dataD.odd;
    int4 map1_dataDx1 = map1_dataDx + 1;
    int4 map1_dataDy1 = map1_dataDy + 1;

    int4 src_StartU = map1_dataDy * src_step + (map1_dataDx << 1) + src_offset;
    int4 src_StartD = src_StartU + src_step;
    int4 src_StartU1 = src_StartU + 2;
    int4 src_StartD1 = src_StartD + 2;

    uchar8 a, b, c, d;
    a.s01 = *((__global uchar2 *)((__global char*)src + src_StartU.x));
    a.s23 = *((__global uchar2 *)((__global char*)src + src_StartU.y));
    a.s45 = *((__global uchar2 *)((__global char*)src + src_StartU.z));
    a.s67 = *((__global uchar2 *)((__global char*)src + src_StartU.w));

    b.s01 = *((__global uchar2 *)((__global char*)src + src_StartU1.x));
    b.s23 = *((__global uchar2 *)((__global char*)src + src_StartU1.y));
    b.s45 = *((__global uchar2 *)((__global char*)src + src_StartU1.z));
    b.s67 = *((__global uchar2 *)((__global char*)src + src_StartU1.w));

    c.s01 = *((__global uchar2 *)((__global char*)src + src_StartD.x));
    c.s23 = *((__global uchar2 *)((__global char*)src + src_StartD.y));
    c.s45 = *((__global uchar2 *)((__global char*)src + src_StartD.z));
    c.s67 = *((__global uchar2 *)((__global char*)src + src_StartD.w));

    d.s01 = *((__global uchar2 *)((__global char*)src + src_StartD1.x));
    d.s23 = *((__global uchar2 *)((__global char*)src + src_StartD1.y));
    d.s45 = *((__global uchar2 *)((__global char*)src + src_StartD1.z));
    d.s67 = *((__global uchar2 *)((__global char*)src + src_StartD1.w));

    int4 ac =(map1_dataDx >= src_cols || map1_dataDy >= src_rows || map1_dataDy< 0 || map1_dataDy < 0);
    int4 bc =(map1_dataDx1 >= src_cols || map1_dataDy >= src_rows || map1_dataDx1 < 0 || map1_dataDy < 0);
    int4 cc =(map1_dataDx >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDx < 0);
    int4 dc =(map1_dataDx1 >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDy1 < 0);

 /*   a.even = (convert_uchar4(ac) == 0)? a.even : val.even;
    a.odd = (convert_uchar4(ac) == 0)? a.odd : val.odd;
    b.even = (convert_uchar4(bc) == 0)? b.even : val.even;
    b.odd = (convert_uchar4(bc) == 0)? b.odd : val.odd;
    c.even = (convert_uchar4(cc) == 0)? c.even : val.even;
    c.odd = (convert_uchar4(cc) == 0)? c.odd : val.odd;
    d.even = (convert_uchar4(dc) == 0)? d.even : val.even;
    d.odd = (convert_uchar4(dc) == 0)? d.odd : val.odd;
*/
    int8 aC = (int8)(ac.x, ac.x, ac.y, ac.y, ac.z, ac.z, ac.w, ac.w);
    int8 bC = (int8)(bc.x, bc.x, bc.y, bc.y, bc.z, bc.z, bc.w, bc.w);
    int8 cC = (int8)(cc.x, cc.x, cc.y, cc.y, cc.z, cc.z, cc.w, cc.w);
    int8 dC = (int8)(dc.x, dc.x, dc.y, dc.y, dc.z, dc.z, dc.w, dc.w);

    a = (convert_uchar8(aC) == 0)? a : val;
    b = (convert_uchar8(bC) == 0)? b : val;
    c = (convert_uchar8(cC) == 0)? c : val;
    d = (convert_uchar8(dC) == 0)? d : val;
    uchar8 dst_data = convert_uchar8_sat_rte((convert_float8(a))* ud * vd +(convert_float8(b))* u * vd + (convert_float8(c))* ud * v + (convert_float8(d)) * u * v );
    
    __global uchar8* D = (__global uchar8 *)(dst + dstStart);

    uchar8 dVal = *D;      
    int8 con = (Gx >= 0 && Gx < (dst_cols << 1) && y >= 0 && y < dst_rows);
    dst_data = (convert_uchar8(con) != 0) ? dst_data : dVal;

    *D = dst_data;
}

/*
__kernel void remapLNFConstant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , double4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    int gx = (x << 4) - (dst_offset&15);
    int16 Gx = (int16)(gx, gx+1, gx+2, gx+3, gx+4, gx+5, gx+6, gx+7, gx+8, gx+9, gx+10, gx+11, gx+12, gx+13, gx+14, gx+15);

    uchar4 nval =convert_uchar4(nVal);
    uchar16 val = (uchar16)(nval.s01, nval.s01, nval.s01, nval.s01, nval.s01, nval.s01, nval.s01, nval.s01, nval.s01, nval.s01, nval.s01, nval.s01, nval.s01, nval.s01, nval.s01, nval.s01);
  
    x = x << 4;

    int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&15);

    int map1Start = y * map1_step + (x << 1) + map1_offset - (((dst_offset>>2) & 3) << 3);
    float8 map1_data;

    map1_data = *((__global float8 *)((__global char*)map1 + map1Start));
    int8 map1_dataD = convert_int8(map1_data);
    float8 temp = map1_data - convert_float8(map1_dataD);

    float4 U = temp.even;
    float4 V = temp.odd;
    float4 UD = 1.0 - U;
    float4 VD = 1.0 - V;

    float16 u, v, ud, vd;
    u = (float16)(U.x, U.x, U.x, U.x, U.y, U.y, U.y, U.y, U.z, U.z, U.z, U.z, U.w, U.w, U.w, U.w);
    v = (float16)(V.x, V.x, V.x, V.x, V.y, V.y, V.y, V.y, V.z, V.z, V.z, V.z, V.w, V.w, V.w, V.w);
    ud = (float16)(UD.x, UD.x, UD.x, UD.x, UD.y, UD.y, UD.y, UD.y, UD.z, UD.z, UD.z, UD.z, UD.w, UD.w, UD.w, UD.w);
    vd = (float16)(VD.x, VD.x, VD.y, VD.y, VD.z, VD.z, VD.w, VD.w);

    //float8 map1_dataU = map1_dataD + 1;

    int4 map1_dataDx = map1_dataD.even;
    int4 map1_dataDy = map1_dataD.odd;
    int4 map1_dataDx1 = map1_dataDx + 1;
    int4 map1_dataDy1 = map1_dataDy + 1;

    int4 src_StartU = map1_dataDy * src_step + (map1_dataDx << 1) + src_offset;
    int4 src_StartD = src_StartU + src_step;
    int4 src_StartU1 = src_StartU + 2;
    int4 src_StartD1 = src_StartD + 2;

    uchar8 a, b, c, d;
    a.s01 = *((__global uchar2 *)((__global char*)src + src_StartU.x));
    a.s23 = *((__global uchar2 *)((__global char*)src + src_StartU.y));
    a.s45 = *((__global uchar2 *)((__global char*)src + src_StartU.z));
    a.s67 = *((__global uchar2 *)((__global char*)src + src_StartU.w));

    b.s01 = *((__global uchar2 *)((__global char*)src + src_StartU1.x));
    b.s23 = *((__global uchar2 *)((__global char*)src + src_StartU1.y));
    b.s45 = *((__global uchar2 *)((__global char*)src + src_StartU1.z));
    b.s67 = *((__global uchar2 *)((__global char*)src + src_StartU1.w));

    c.s01 = *((__global uchar2 *)((__global char*)src + src_StartD.x));
    c.s23 = *((__global uchar2 *)((__global char*)src + src_StartD.y));
    c.s45 = *((__global uchar2 *)((__global char*)src + src_StartD.z));
    c.s67 = *((__global uchar2 *)((__global char*)src + src_StartD.w));

    d.s01 = *((__global uchar2 *)((__global char*)src + src_StartD1.x));
    d.s23 = *((__global uchar2 *)((__global char*)src + src_StartD1.y));
    d.s45 = *((__global uchar2 *)((__global char*)src + src_StartD1.z));
    d.s67 = *((__global uchar2 *)((__global char*)src + src_StartD1.w));

    int4 ac =(map1_dataDx >= src_cols || map1_dataDy >= src_rows || map1_dataDy< 0 || map1_dataDy < 0);
    int4 bc =(map1_dataDx1 >= src_cols || map1_dataDy >= src_rows || map1_dataDx1 < 0 || map1_dataDy < 0);
    int4 cc =(map1_dataDx >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDx < 0);
    int4 dc =(map1_dataDx1 >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDy1 < 0);

    int8 aC = (int8)(ac.x, ac.x, ac.y, ac.y, ac.z, ac.z, ac.w, ac.w);
    int8 bC = (int8)(bc.x, bc.x, bc.y, bc.y, bc.z, bc.z, bc.w, bc.w);
    int8 cC = (int8)(cc.x, cc.x, cc.y, cc.y, cc.z, cc.z, cc.w, cc.w);
    int8 dC = (int8)(dc.x, dc.x, dc.y, dc.y, dc.z, dc.z, dc.w, dc.w);

    a = (convert_uchar8(aC) == 0)? a : val;
    b = (convert_uchar8(bC) == 0)? b : val;
    c = (convert_uchar8(cC) == 0)? c : val;
    d = (convert_uchar8(dC) == 0)? d : val;
    uchar8 dst_data = convert_uchar8_sat_rte((convert_float8(a))* ud * vd +(convert_float8(b))* u * vd + (convert_float8(c))* ud * v + (convert_float8(d)) * u * v );
    
    __global uchar8* D = (__global uchar8 *)(dst + dstStart);

    uchar8 dVal = *D;      
    int8 con = (Gx >= 0 && Gx < (dst_cols << 1) && y >= 0 && y < dst_rows);
    dst_data = (convert_uchar8(con) != 0) ? dst_data : dVal;

    *D = dst_data;
    
}
*/

