
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
//#pragma OPENCL EXTENSION cl_amd_printf : enable

#if defined DOUBLE_SUPPORT
#pragma OPENCL EXTENSION cl_khr_fp64:enable
typedef double4 F4 ;
#else 
typedef float4 F4;
#endif


/////////////////////////////////////////////////////////
///////////////////////using buffer//////////////////////
/////////////////////////////////////////////////////////
__kernel void remapNNSConstant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {
        x = x << 2;
        int gx = x - (dst_offset&3);
        int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

        uchar4 nval =convert_uchar4(nVal);
        uchar val = nval.s0;

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&3);

        int map1Start = y * map1_step + (x << 2) + map1_offset - ((dst_offset & 3) << 2);
        short8 map1_data;

        map1_data = *((__global short8 *)((__global char*)map1 + map1Start));
        int4 srcIdx = convert_int4(map1_data.odd) * src_step + convert_int4(map1_data.even) + src_offset;
    
        uchar4 src_data;

        src_data.s0 = *(src + srcIdx.s0);
        src_data.s1 = *(src + srcIdx.s1);
        src_data.s2 = *(src + srcIdx.s2);
        src_data.s3 = *(src + srcIdx.s3);
        uchar4 dst_data;
        dst_data = convert_uchar4((convert_int4(map1_data.even) >= (int4)(src_cols) || convert_int4(map1_data.odd) >= (int4)(src_rows)))? (uchar4)(val) : src_data;
 
        __global uchar4* d = (__global uchar4 *)(dst + dstStart);

        uchar4 dVal = *d;      

        int4 con = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
        dst_data = (convert_uchar4(con) != convert_uchar4((int4)(0))) ? dst_data : dVal;

        *d = dst_data;

    }

}
__kernel void remapNNFConstant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {
        x = x << 2;
        int gx = x - (dst_offset&3);
        int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

        uchar4 nval =convert_uchar4(nVal);
        uchar val = nval.s0;

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&3);

        int map1Start = y * map1_step + (x << 3) + map1_offset - ((dst_offset & 3) << 3);
        float8 map1_data;

        map1_data = *((__global float8 *)((__global char*)map1 + map1Start));
        int8 map1_dataZ = convert_int8_sat_rte(map1_data);
        int4 srcIdx = map1_dataZ.odd * src_step + map1_dataZ.even + src_offset;
    
        uchar4 src_data;

        src_data.s0 = *(src + srcIdx.s0);
        src_data.s1 = *(src + srcIdx.s1);
        src_data.s2 = *(src + srcIdx.s2);
        src_data.s3 = *(src + srcIdx.s3);
        uchar4 dst_data;
        dst_data = convert_uchar4(map1_dataZ.even >= (int4)(src_cols) || map1_dataZ.odd >= (int4)(src_rows)) ? (uchar4)(val) : src_data;
        __global uchar4* d = (__global uchar4 *)(dst + dstStart);

        uchar4 dVal = *d;      

        int4 con = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
  
        dst_data = (convert_uchar4(con) != convert_uchar4((int4)(0))) ? dst_data : dVal;
        *d = dst_data;

    }
}

__kernel void remapNNF1Constant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1,  __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {
        x = x << 2;
        int gx = x - (dst_offset&3);
        int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

        uchar4 nval =convert_uchar4(nVal);
        uchar val = nval.s0;

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&3);

        int map1Start = y * map1_step + (x << 2) + map1_offset - ((dst_offset & 3) << 2);
        float4 map1_data;
        float4 map2_data;

        map1_data = *((__global float4 *)((__global char*)map1 + map1Start));
        map2_data = *((__global float4 *)((__global char*)map2 + map1Start));
        float8 map_data = (float8)(map1_data.s0, map2_data.s0, map1_data.s1, map2_data.s1, map1_data.s2, map2_data.s2, map1_data.s3, map2_data.s3);
        int8 map_dataZ = convert_int8_sat_rte(map_data);
        int4 srcIdx = map_dataZ.odd * src_step + map_dataZ.even + src_offset;
    
        uchar4 src_data;

        src_data.s0 = *(src + srcIdx.s0);
        src_data.s1 = *(src + srcIdx.s1);
        src_data.s2 = *(src + srcIdx.s2);
        src_data.s3 = *(src + srcIdx.s3);
        uchar4 dst_data;
        dst_data = convert_uchar4(map_dataZ.even >= (int4)(src_cols) || map_dataZ.odd >= (int4)(src_rows)) ? (uchar4)(val) : src_data;
        __global uchar4* d = (__global uchar4 *)(dst + dstStart);

        uchar4 dVal = *d;      

        int4 con = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
  
        dst_data = (convert_uchar4(con) != convert_uchar4((int4)(0))) ? dst_data : dVal;
        *d = dst_data;
    }
}


__kernel void remapNNSConstant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    {
        x = x << 4;
        int gx = x - (dst_offset&15);
        int16 Gx = (int16)(gx, gx+1, gx+2, gx+3, gx+4, gx+5, gx+6, gx+7, gx+8, gx+9, gx+10, gx+11, gx+12, gx+13, gx+14, gx+15);
        uchar4 nval =convert_uchar4_sat_rte(nVal);

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&15);
        int map1Start = y * map1_step + x + map1_offset - (dst_offset&15 );
        short8 map1_data;

        map1_data = *((__global short8 *)((__global char*)map1 + map1Start));
        int4 srcIdx = convert_int4(map1_data.odd) * src_step + (convert_int4(map1_data.even) <<((int4)(2))) + src_offset;
        uchar4 src_a, src_b, src_c, src_d;
        src_a = *((__global uchar4 *)((__global char*)src + srcIdx.s0));
        src_b = *((__global uchar4 *)((__global char*)src + srcIdx.s1));
        src_c = *((__global uchar4 *)((__global char*)src + srcIdx.s2));
        src_d = *((__global uchar4 *)((__global char*)src + srcIdx.s3));

        uchar16 dst_data;
        uchar4 dst_a, dst_b, dst_c, dst_d;
        dst_a = (map1_data.s0 >= src_cols || map1_data.s1 >= src_rows)? nval : src_a;
        dst_b = (map1_data.s2 >= src_cols || map1_data.s3 >= src_rows)? nval : src_b;
        dst_c = (map1_data.s4 >= src_cols || map1_data.s5 >= src_rows)? nval : src_c;
        dst_d = (map1_data.s6 >= src_cols || map1_data.s7 >= src_rows)? nval : src_d;

        dst_data = (uchar16)(dst_a, dst_b, dst_c, dst_d);
        __global uchar16* d = (__global uchar16 *)(dst + dstStart);

        uchar16 dVal = *d;      

        int16 con = (Gx >= 0 && Gx < (dst_cols<<2) && y >= 0 && y < dst_rows);
        dst_data = (convert_uchar16(con) != ((uchar16)(0))) ? dst_data : dVal;

        *d = dst_data;
    }

}
__kernel void remapNNFConstant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    {
        x = x << 4;
        int gx = x - (dst_offset&15);
        int16 Gx = (int16)(gx, gx+1, gx+2, gx+3, gx+4, gx+5, gx+6, gx+7, gx+8, gx+9, gx+10, gx+11, gx+12, gx+13, gx+14, gx+15);

        uchar4 nval =convert_uchar4(nVal);

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&15);

        int map1Start = y * map1_step +(x << 1) + map1_offset - ((dst_offset&15) << 1);
        float8 map1_data;

        map1_data = *((__global float8 *)((__global char*)map1 + map1Start));
        int8 map1_dataZ = convert_int8_sat_rte(map1_data);

        int4 srcIdx = map1_dataZ.odd * src_step + (map1_dataZ.even <<((int4)(2))) + src_offset;
        uchar4 src_a, src_b, src_c, src_d;
        src_a = *((__global uchar4 *)((__global char*)src + srcIdx.s0));
        src_b = *((__global uchar4 *)((__global char*)src + srcIdx.s1));
        src_c = *((__global uchar4 *)((__global char*)src + srcIdx.s2));
        src_d = *((__global uchar4 *)((__global char*)src + srcIdx.s3));

        uchar16 dst_data;
        uchar4 dst_a, dst_b, dst_c, dst_d;
        dst_a = (map1_dataZ.s0 >= src_cols || map1_dataZ.s1 >= src_rows)? nval : src_a;
        dst_b = (map1_dataZ.s2 >= src_cols || map1_dataZ.s3 >= src_rows)? nval : src_b;
        dst_c = (map1_dataZ.s4 >= src_cols || map1_dataZ.s5 >= src_rows)? nval : src_c;
        dst_d = (map1_dataZ.s6 >= src_cols || map1_dataZ.s7 >= src_rows)? nval : src_d;

        dst_data = (uchar16)(dst_a, dst_b, dst_c, dst_d);
        __global uchar16* d = (__global uchar16 *)(dst + dstStart);

        uchar16 dVal = *d;      

        int16 con = (Gx >= 0 && Gx < (dst_cols<<2) && y >= 0 && y < dst_rows);
        dst_data = (convert_uchar16(con) != ((uchar16)(0))) ? dst_data : dVal;

        *d = dst_data;

    }

}

__kernel void remapNNF1Constant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1,  __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    {
        x = x << 4;
        int gx = x - (dst_offset&15);
        int16 Gx = (int16)(gx, gx+1, gx+2, gx+3, gx+4, gx+5, gx+6, gx+7, gx+8, gx+9, gx+10, gx+11, gx+12, gx+13, gx+14, gx+15);

        uchar4 nval =convert_uchar4(nVal);

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&15);

        int map1Start = y * map1_step + x + map1_offset - (dst_offset&15);
        float4 map1_data;
        float4 map2_data;

        map1_data = *((__global float4 *)((__global char*)map1 + map1Start));
        map2_data = *((__global float4 *)((__global char*)map2 + map1Start));
        float8 map_data = (float8)(map1_data.s0, map2_data.s0, map1_data.s1, map2_data.s1, map1_data.s2, map2_data.s2, map1_data.s3, map2_data.s3);
        int8 map1_dataZ = convert_int8_sat_rte(map_data);

        int4 srcIdx = map1_dataZ.odd * src_step + (map1_dataZ.even <<((int4)(2))) + src_offset;
        uchar4 src_a, src_b, src_c, src_d;
        src_a = *((__global uchar4 *)((__global char*)src + srcIdx.s0));
        src_b = *((__global uchar4 *)((__global char*)src + srcIdx.s1));
        src_c = *((__global uchar4 *)((__global char*)src + srcIdx.s2));
        src_d = *((__global uchar4 *)((__global char*)src + srcIdx.s3));

        uchar16 dst_data;
        uchar4 dst_a, dst_b, dst_c, dst_d;
        dst_a = (map1_dataZ.s0 >= src_cols || map1_dataZ.s1 >= src_rows)? nval : src_a;
        dst_b = (map1_dataZ.s2 >= src_cols || map1_dataZ.s3 >= src_rows)? nval : src_b;
        dst_c = (map1_dataZ.s4 >= src_cols || map1_dataZ.s5 >= src_rows)? nval : src_c;
        dst_d = (map1_dataZ.s6 >= src_cols || map1_dataZ.s7 >= src_rows)? nval : src_d;

        dst_data = (uchar16)(dst_a, dst_b, dst_c, dst_d);
        __global uchar16* d = (__global uchar16 *)(dst + dstStart);

        uchar16 dVal = *d;      

        int16 con = (Gx >= 0 && Gx < (dst_cols<<2) && y >= 0 && y < dst_rows);
        dst_data = (convert_uchar16(con) != ((uchar16)(0))) ? dst_data : dVal;

        *d = dst_data;

    }

}


__kernel void remapNNSConstant_C1_D5(__global float* dst, __global float const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows ,int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {     
        x = x << 4;

        int gx = x - (dst_offset&15);
        int4 Gx = (int4)(gx, gx+4, gx+8, gx+12);

        float4 nval =convert_float4(nVal);
        float val = nval.s0;

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&15);

        int map1Start = y * map1_step + x + map1_offset - (dst_offset&15);
        short8 map1_data;

        map1_data = *((__global short8 *)((__global char*)map1 + map1Start));
    
        int4 srcIdx = convert_int4(map1_data.odd) * src_step + (convert_int4(map1_data.even) <<((int4)(2))) + src_offset;
    
        float4 src_data;
        src_data.s0 = *((__global float *)((__global char*)src + srcIdx.s0));
        src_data.s1 = *((__global float *)((__global char*)src + srcIdx.s1));
        src_data.s2 = *((__global float *)((__global char*)src + srcIdx.s2));
        src_data.s3 = *((__global float *)((__global char*)src + srcIdx.s3));
        float4 dst_data;
        
        dst_data.s0 = (map1_data.s0 >= src_cols || map1_data.s1 >= src_rows)? val : src_data.s0;
        dst_data.s1 = (map1_data.s2 >= src_cols || map1_data.s3 >= src_rows)? val : src_data.s1;
        dst_data.s2 = (map1_data.s4 >= src_cols || map1_data.s5 >= src_rows)? val : src_data.s2;
        dst_data.s3 = (map1_data.s6 >= src_cols || map1_data.s7 >= src_rows)? val : src_data.s3;
        
  
        __global float4* d = (__global float4 *)((__global uchar*)dst + dstStart);

        float4 dVal = *d;      

        int4 con = (Gx >= 0 && Gx < (dst_cols<<2) && y >= 0 && y < dst_rows);
        dst_data = (convert_float4(con) != (float4)(0)) ? dst_data : dVal;

        *d = dst_data;

    }

}
__kernel void remapNNFConstant_C1_D5(__global float* dst, __global float const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows ,int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {
        x = x << 4;

        int gx = x - (dst_offset&15);
        int4 Gx = (int4)(gx, gx+4, gx+8, gx+12);

        float4 nval =convert_float4(nVal);
        float val = nval.s0;

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&15);

        int map1Start = y * map1_step + (x << 1) + map1_offset - ((dst_offset&15) << 1);
        float8 map1_data;

        map1_data = *((__global float8 *)((__global char*)map1 + map1Start));
        int8 map1_dataZ = convert_int8_sat_rte(map1_data);

        int4 srcIdx = convert_int4(map1_dataZ.odd) * src_step + convert_int4(map1_dataZ.even <<(int4)(2)) + src_offset;
    
        float4 src_data;
        src_data.s0 = *((__global float *)((__global char*)src + srcIdx.s0));
        src_data.s1 = *((__global float *)((__global char*)src + srcIdx.s1));
        src_data.s2 = *((__global float *)((__global char*)src + srcIdx.s2));
        src_data.s3 = *((__global float *)((__global char*)src + srcIdx.s3));
        float4 dst_data;
        
        dst_data.s0 = (map1_dataZ.s0 >= src_cols || map1_dataZ.s1 >= src_rows)? val : src_data.s0;
        dst_data.s1 = (map1_dataZ.s2 >= src_cols || map1_dataZ.s3 >= src_rows)? val : src_data.s1;
        dst_data.s2 = (map1_dataZ.s4 >= src_cols || map1_dataZ.s5 >= src_rows)? val : src_data.s2;
        dst_data.s3 = (map1_dataZ.s6 >= src_cols || map1_dataZ.s7 >= src_rows)? val : src_data.s3;
        
 
        __global float4* d = (__global float4 *)((__global uchar*)dst + dstStart);

        float4 dVal = *d;      

        int4 con = (Gx >= 0 && Gx < (dst_cols<<2) && y >= 0 && y < dst_rows);
        dst_data = (convert_float4(con) != (float4)(0)) ? dst_data : dVal;

        *d = dst_data;

    }

}

__kernel void remapNNF1Constant_C1_D5(__global float* dst, __global float const * restrict  src,
        __global float * map1, __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows ,int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {
        x = x << 4;

        int gx = x - (dst_offset&15);
        int4 Gx = (int4)(gx, gx+4, gx+8, gx+12);

        float4 nval =convert_float4(nVal);
        float val = nval.s0;

        int dstStart = y * dst_step + x  + dst_offset - (dst_offset&15);

        int map1Start = y * map1_step + x + map1_offset - (dst_offset&15);
        float4 map1_data;
        float4 map2_data;

        map1_data = *((__global float4 *)((__global char*)map1 + map1Start));
        map2_data = *((__global float4 *)((__global char*)map2 + map1Start));
        float8 map_data = (float8)(map1_data.s0, map2_data.s0, map1_data.s1, map2_data.s1, map1_data.s2, map2_data.s2, map1_data.s3, map2_data.s3);
        int8 map1_dataZ = convert_int8_sat_rte(map_data);

        int4 srcIdx = convert_int4(map1_dataZ.odd) * src_step + convert_int4(map1_dataZ.even <<(int4)(2)) + src_offset;
    
        float4 src_data;
        src_data.s0 = *((__global float *)((__global char*)src + srcIdx.s0));
        src_data.s1 = *((__global float *)((__global char*)src + srcIdx.s1));
        src_data.s2 = *((__global float *)((__global char*)src + srcIdx.s2));
        src_data.s3 = *((__global float *)((__global char*)src + srcIdx.s3));
        float4 dst_data;
        
        dst_data.s0 = (map1_dataZ.s0 >= src_cols || map1_dataZ.s1 >= src_rows)? val : src_data.s0;
        dst_data.s1 = (map1_dataZ.s2 >= src_cols || map1_dataZ.s3 >= src_rows)? val : src_data.s1;
        dst_data.s2 = (map1_dataZ.s4 >= src_cols || map1_dataZ.s5 >= src_rows)? val : src_data.s2;
        dst_data.s3 = (map1_dataZ.s6 >= src_cols || map1_dataZ.s7 >= src_rows)? val : src_data.s3;
        
 
        __global float4* d = (__global float4 *)((__global uchar*)dst + dstStart);

        float4 dVal = *d;      

        int4 con = (Gx >= 0 && Gx < (dst_cols<<2) && y >= 0 && y < dst_rows);
        dst_data = (convert_float4(con) != (float4)(0)) ? dst_data : dVal;

        *d = dst_data;

    }

}

__kernel void remapNNSConstant_C4_D5(__global float * dst, __global float const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    {
      int dstIdx = y * dst_step + (x << 4) + dst_offset  ;
      int mapIdx = y * map1_step + (x << 2) + map1_offset ;
      short2 map1_data = *((__global short2 *)((__global char*)map1 + mapIdx));

      int srcIdx = map1_data.y * src_step + (map1_data.x << 4) + src_offset;
      float4 nval = convert_float4(nVal);
      float4 src_data = *((__global float4 *)((__global uchar *)src + srcIdx));
       *((__global float4 *)((__global uchar*)dst + dstIdx)) = (map1_data.x >= src_cols || map1_data.y >= src_rows) ? nval : src_data;
    }
}
__kernel void remapNNFConstant_C4_D5(__global float * dst, __global float const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    {
      int dstIdx = y * dst_step + (x << 4) + dst_offset  ;
      int mapIdx = y * map1_step + (x << 3) + map1_offset ;
      float2 map1_data = *((__global float2 *)((__global char*)map1 + mapIdx));
      int2 map1_dataZ = convert_int2_sat_rte(map1_data);
      int srcIdx = map1_dataZ.y * src_step + (map1_dataZ.x << 4) + src_offset;
      float4 nval = convert_float4(nVal);
      float4 src_data = *((__global float4 *)((__global uchar *)src + srcIdx));
       *((__global float4 *)((__global uchar*)dst + dstIdx)) = (map1_dataZ.x >= src_cols || map1_dataZ.y >= src_rows) ? nval : src_data;
    }
}

__kernel void remapNNF1Constant_C4_D5(__global float * dst, __global float const * restrict  src,
        __global float * map1,  __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    {
      int dstIdx = y * dst_step + (x << 4) + dst_offset  ;
      int mapIdx = y * map1_step + (x << 2) + map1_offset ;
      float map1_data = *((__global float *)((__global char*)map1 + mapIdx));
      float map2_data = *((__global float *)((__global char*)map2 + mapIdx));
      float2 map_data = (float2)(map1_data, map2_data);
      int2 map1_dataZ = convert_int2_sat_rte(map_data);
      int srcIdx = map1_dataZ.y * src_step + (map1_dataZ.x << 4) + src_offset;
      float4 nval = convert_float4(nVal);
      float4 src_data = *((__global float4 *)((__global uchar *)src + srcIdx));
       *((__global float4 *)((__global uchar*)dst + dstIdx)) = (map1_dataZ.x >= src_cols || map1_dataZ.y >= src_rows) ? nval : src_data;
    }
}



__kernel void remapLNFConstant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < threadCols && y < dst_rows)
    {
      x = x << 2; 
      int gx = x - (dst_offset&3);
      int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

      uchar4 nval =convert_uchar4(nVal);
      uchar val = nval.s0;
  

      int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&3);

      int map1Start = y * map1_step + (x << 3) + map1_offset - ((dst_offset & 3) << 3);
      float8 map1_data;

      map1_data = *((__global float8 *)((__global char*)map1 + map1Start));
      int8 map1_dataD = convert_int8(map1_data);
      float8 temp = map1_data - convert_float8(map1_dataD);

      float4 u = temp.even;
      float4 v = temp.odd;
      float4 ud = (float4)(1.0) - u;
      float4 vd = (float4)(1.0) - v;
      //float8 map1_dataU = map1_dataD + 1;

      int4 map1_dataDx = map1_dataD.even;
      int4 map1_dataDy = map1_dataD.odd;
      int4 map1_dataDx1 = map1_dataDx + (int4)(1);
      int4 map1_dataDy1 = map1_dataDy + (int4)(1);

      int4 src_StartU = map1_dataDy * src_step + map1_dataDx + src_offset;
      int4 src_StartD = src_StartU + src_step;
     /* 
      //not using the vload
      int4 src_StartU1 = src_StartU + (int4)(1);
      int4 src_StartD1 = src_StartD + (int4)(1);

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
      */
      uchar2 aU, aD, bU, bD, cU, cD, dU, dD;

      aU = vload2(0, src + src_StartU.s0);
      bU = vload2(0, src + src_StartU.s1);
      cU = vload2(0, src + src_StartU.s2);
      dU = vload2(0, src + src_StartU.s3);
      aD = vload2(0, src + src_StartD.s0);
      bD = vload2(0, src + src_StartD.s1);
      cD = vload2(0, src + src_StartD.s2);
      dD = vload2(0, src + src_StartD.s3);

      uchar4 a, b, c, d;
      a = (uchar4)(aU.x, bU.x, cU.x, dU.x);
      b = (uchar4)(aU.y, bU.y, cU.y, dU.y);
      c = (uchar4)(aD.x, bD.x, cD.x, dD.x);
      d = (uchar4)(aD.y, bD.y, cD.y, dD.y);
      
      int4 ac =(map1_dataDx >= src_cols || map1_dataDy >= src_rows || map1_dataDy< 0 || map1_dataDy < 0);
      int4 bc =(map1_dataDx1 >= src_cols || map1_dataDy >= src_rows || map1_dataDx1 < 0 || map1_dataDy < 0);
      int4 cc =(map1_dataDx >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDx < 0);
      int4 dc =(map1_dataDx1 >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDy1 < 0);
      a = (convert_uchar4(ac) == (uchar4)(0))? a : val;
      b = (convert_uchar4(bc) == (uchar4)(0))? b : val;
      c = (convert_uchar4(cc) == (uchar4)(0))? c : val;
      d = (convert_uchar4(dc) == (uchar4)(0))? d : val;

      uchar4 dst_data = convert_uchar4_sat_rte((convert_float4(a))* ud * vd +(convert_float4(b))* u * vd + (convert_float4(c))* ud * v + (convert_float4(d)) * u * v );
    
      __global uchar4* D = (__global uchar4 *)(dst + dstStart);

      uchar4 dVal = *D;      
      int4 con = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
      dst_data = (convert_uchar4(con) != (uchar4)(0)) ? dst_data : dVal;

      *D = dst_data;
    }
}

__kernel void remapLNF1Constant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1,  __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < threadCols && y < dst_rows)
    {
      x = x << 2; 
      int gx = x - (dst_offset&3);
      int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

      uchar4 nval =convert_uchar4(nVal);
      uchar val = nval.s0;
  

      int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&3);

      int map1Start = y * map1_step + (x << 2) + map1_offset - ((dst_offset & 3) << 2);
      float4 map1_data;
      float4 map2_data;

      map1_data = *((__global float4 *)((__global char*)map1 + map1Start));
      map2_data = *((__global float4 *)((__global char*)map2 + map1Start));
      float8 map_data = (float8)(map1_data.s0, map2_data.s0, map1_data.s1, map2_data.s1, map1_data.s2, map2_data.s2, map1_data.s3, map2_data.s3);
      int8 map1_dataD = convert_int8(map_data);
      float8 temp = map_data - convert_float8(map1_dataD);

      float4 u = temp.even;
      float4 v = temp.odd;
      float4 ud = (float4)(1.0) - u;
      float4 vd = (float4)(1.0) - v;
      //float8 map1_dataU = map1_dataD + 1;

      int4 map1_dataDx = map1_dataD.even;
      int4 map1_dataDy = map1_dataD.odd;
      int4 map1_dataDx1 = map1_dataDx + (int4)(1);
      int4 map1_dataDy1 = map1_dataDy + (int4)(1);

      int4 src_StartU = map1_dataDy * src_step + map1_dataDx + src_offset;
      int4 src_StartD = src_StartU + src_step;
     /* 
      //not using the vload
      int4 src_StartU1 = src_StartU + (int4)(1);
      int4 src_StartD1 = src_StartD + (int4)(1);

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
      */
      uchar2 aU, aD, bU, bD, cU, cD, dU, dD;

      aU = vload2(0, src + src_StartU.s0);
      bU = vload2(0, src + src_StartU.s1);
      cU = vload2(0, src + src_StartU.s2);
      dU = vload2(0, src + src_StartU.s3);
      aD = vload2(0, src + src_StartD.s0);
      bD = vload2(0, src + src_StartD.s1);
      cD = vload2(0, src + src_StartD.s2);
      dD = vload2(0, src + src_StartD.s3);

      uchar4 a, b, c, d;
      a = (uchar4)(aU.x, bU.x, cU.x, dU.x);
      b = (uchar4)(aU.y, bU.y, cU.y, dU.y);
      c = (uchar4)(aD.x, bD.x, cD.x, dD.x);
      d = (uchar4)(aD.y, bD.y, cD.y, dD.y);
      
      int4 ac =(map1_dataDx >= src_cols || map1_dataDy >= src_rows || map1_dataDy< 0 || map1_dataDy < 0);
      int4 bc =(map1_dataDx1 >= src_cols || map1_dataDy >= src_rows || map1_dataDx1 < 0 || map1_dataDy < 0);
      int4 cc =(map1_dataDx >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDx < 0);
      int4 dc =(map1_dataDx1 >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDy1 < 0);
      a = (convert_uchar4(ac) == (uchar4)(0))? a : val;
      b = (convert_uchar4(bc) == (uchar4)(0))? b : val;
      c = (convert_uchar4(cc) == (uchar4)(0))? c : val;
      d = (convert_uchar4(dc) == (uchar4)(0))? d : val;

      uchar4 dst_data = convert_uchar4_sat_rte((convert_float4(a))* ud * vd +(convert_float4(b))* u * vd + (convert_float4(c))* ud * v + (convert_float4(d)) * u * v );
    
      __global uchar4* D = (__global uchar4 *)(dst + dstStart);

      uchar4 dVal = *D;      
      int4 con = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
      dst_data = (convert_uchar4(con) != (uchar4)(0)) ? dst_data : dVal;

      *D = dst_data;
    }
}


__kernel void remapLNSConstant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {
        x = x << 2;
        int gx = x - (dst_offset&3);
        int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

        uchar4 nval =convert_uchar4(nVal);
        uchar val = nval.s0;

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&3);

        int map1Start = y * map1_step + (x << 2) + map1_offset - ((dst_offset & 3) << 2);
        short8 map1_data;

        map1_data = *((__global short8 *)((__global char*)map1 + map1Start));
        int4 srcIdx = convert_int4(map1_data.odd) * src_step + convert_int4(map1_data.even) + src_offset;
    
        uchar4 src_data;

        src_data.s0 = *(src + srcIdx.s0);
        src_data.s1 = *(src + srcIdx.s1);
        src_data.s2 = *(src + srcIdx.s2);
        src_data.s3 = *(src + srcIdx.s3);
        uchar4 dst_data;
        dst_data = convert_uchar4((convert_int4(map1_data.even) >= (int4)(src_cols) || convert_int4(map1_data.odd) >= (int4)(src_rows)))? (uchar4)(val) : src_data;
 
        __global uchar4* d = (__global uchar4 *)(dst + dstStart);

        uchar4 dVal = *d;      

        int4 con = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
        dst_data = (convert_uchar4(con) != (uchar4)(0)) ? dst_data : dVal;

        *d = dst_data;

    }

}



__kernel void remapLNFConstant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < threadCols && y < dst_rows)
    {
      x = x << 4; 
      int gx = x - (dst_offset&15);
      int16 Gx = (int16)(gx, gx+1, gx+2, gx+3, gx+4, gx+5, gx+6, gx+7, gx+8, gx+9, gx+10, gx+11, gx+12, gx+13, gx+14, gx+15);

      uchar4 nval =convert_uchar4(nVal);

      int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&15);

      int map1Start = y * map1_step + (x << 1) + map1_offset - ((dst_offset & 15) << 1);
      float8 map1_data;

      map1_data = *((__global float8 *)((__global char*)map1 + map1Start));
      int8 map1_dataD = convert_int8(map1_data);
      float8 temp = map1_data - convert_float8(map1_dataD);

      float4 u = temp.even;
      float4 v = temp.odd;
      float4 ud = (float4)(1.0) - u;
      float4 vd = (float4)(1.0) - v;
      
      //float8 map1_dataU = map1_dataD + 1;

      int4 map1_dataDx = map1_dataD.even;
      int4 map1_dataDy = map1_dataD.odd;
      int4 map1_dataDx1 = map1_dataDx + (int4)(1);
      int4 map1_dataDy1 = map1_dataDy + (int4)(1);

      int4 src_StartU = map1_dataDy * src_step + (convert_int4(map1_dataDx) << (int4)(2)) + src_offset;
      int4 src_StartD = src_StartU + src_step;

      uchar8 aU, bU, cU, dU, aD, bD, cD, dD;
      aU = vload8(0, src + src_StartU.s0);
      bU = vload8(0, src + src_StartU.s1);
      cU = vload8(0, src + src_StartU.s2);
      dU = vload8(0, src + src_StartU.s3);
      aD = vload8(0, src + src_StartD.s0);
      bD = vload8(0, src + src_StartD.s1);
      cD = vload8(0, src + src_StartD.s2);
      dD = vload8(0, src + src_StartD.s3);
      uchar16 a, b, c, d;
      a = (uchar16)(aU.s0123, bU.s0123, cU.s0123, dU.s0123);
      b = (uchar16)(aU.s4567, bU.s4567, cU.s4567, dU.s4567);
      c = (uchar16)(aD.s0123, bD.s0123, cD.s0123, dD.s0123);
      d = (uchar16)(aD.s4567, bD.s4567, cD.s4567, dD.s4567);
      int4 ac =(map1_dataDx >= src_cols || map1_dataDy >= src_rows || map1_dataDy< 0 || map1_dataDy < 0);
      int4 bc =(map1_dataDx1 >= src_cols || map1_dataDy >= src_rows || map1_dataDx1 < 0 || map1_dataDy < 0);
      int4 cc =(map1_dataDx >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDx < 0);
      int4 dc =(map1_dataDx1 >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDy1 < 0);

      int16 acc = (int16)((int4)(ac.x), (int4)(ac.y), (int4)(ac.z), (int4)(ac.w));
      int16 bcc = (int16)((int4)(bc.x), (int4)(bc.y), (int4)(bc.z), (int4)(bc.w));
      int16 ccc = (int16)((int4)(cc.x), (int4)(cc.y), (int4)(cc.z), (int4)(cc.w));
      int16 dcc = (int16)((int4)(dc.x), (int4)(dc.y), (int4)(dc.z), (int4)(dc.w));
 
      uchar16 val = (uchar16)(nval, nval, nval, nval);
      a = (convert_uchar16(acc) == (uchar16)(0))? a : val;
      b = (convert_uchar16(bcc) == (uchar16)(0))? b : val;
      c = (convert_uchar16(ccc) == (uchar16)(0))? c : val;
      d = (convert_uchar16(dcc) == (uchar16)(0))? d : val;

      float16 U = (float16)((float4)(u.x), (float4)(u.y), (float4)(u.z), (float4)(u.w));
      float16 V = (float16)((float4)(v.x), (float4)(v.y), (float4)(v.z), (float4)(v.w));
      float16 Ud = (float16)((float4)(ud.x), (float4)(ud.y), (float4)(ud.z), (float4)(ud.w));
      float16 Vd = (float16)((float4)(vd.x), (float4)(vd.y), (float4)(vd.z), (float4)(vd.w));

      uchar16 dst_data = convert_uchar16_sat_rte((convert_float16(a))* Ud * Vd +(convert_float16(b))* U * Vd + (convert_float16(c))* Ud * V + (convert_float16(d)) * U * V );
    
      __global uchar16* D = (__global uchar16 *)(dst + dstStart);

      uchar16 dVal = *D;      
      int16 con = (Gx >= 0 && Gx < (dst_cols<<2) && y >= 0 && y < dst_rows);
      dst_data = (convert_uchar16(con) != (uchar16)(0)) ? dst_data : dVal;

      *D = dst_data;
    }
}


__kernel void remapLNF1Constant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1,  __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < threadCols && y < dst_rows)
    {
      x = x << 4; 
      int gx = x - (dst_offset&15);
      int16 Gx = (int16)(gx, gx+1, gx+2, gx+3, gx+4, gx+5, gx+6, gx+7, gx+8, gx+9, gx+10, gx+11, gx+12, gx+13, gx+14, gx+15);

      uchar4 nval =convert_uchar4(nVal);

      int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&15);

      int map1Start = y * map1_step + x + map1_offset - (dst_offset & 15);
      float4 map1_data;
      float4 map2_data;

      map1_data = *((__global float4 *)((__global char*)map1 + map1Start));
      map2_data = *((__global float4 *)((__global char*)map2 + map1Start));
      float8 map_data = (float8)(map1_data.s0, map2_data.s0, map1_data.s1, map2_data.s1, map1_data.s2, map2_data.s2, map1_data.s3, map2_data.s3);
      int8 map1_dataD = convert_int8(map_data);
      float8 temp = map_data - convert_float8(map1_dataD);

      float4 u = temp.even;
      float4 v = temp.odd;
      float4 ud = (float4)(1.0) - u;
      float4 vd = (float4)(1.0) - v;
      
      //float8 map1_dataU = map1_dataD + 1;

      int4 map1_dataDx = map1_dataD.even;
      int4 map1_dataDy = map1_dataD.odd;
      int4 map1_dataDx1 = map1_dataDx + (int4)(1);
      int4 map1_dataDy1 = map1_dataDy + (int4)(1);

      int4 src_StartU = map1_dataDy * src_step + (convert_int4(map1_dataDx) << (int4)(2)) + src_offset;
      int4 src_StartD = src_StartU + src_step;

      uchar8 aU, bU, cU, dU, aD, bD, cD, dD;
      aU = vload8(0, src + src_StartU.s0);
      bU = vload8(0, src + src_StartU.s1);
      cU = vload8(0, src + src_StartU.s2);
      dU = vload8(0, src + src_StartU.s3);
      aD = vload8(0, src + src_StartD.s0);
      bD = vload8(0, src + src_StartD.s1);
      cD = vload8(0, src + src_StartD.s2);
      dD = vload8(0, src + src_StartD.s3);
      uchar16 a, b, c, d;
      a = (uchar16)(aU.s0123, bU.s0123, cU.s0123, dU.s0123);
      b = (uchar16)(aU.s4567, bU.s4567, cU.s4567, dU.s4567);
      c = (uchar16)(aD.s0123, bD.s0123, cD.s0123, dD.s0123);
      d = (uchar16)(aD.s4567, bD.s4567, cD.s4567, dD.s4567);
      int4 ac =(map1_dataDx >= src_cols || map1_dataDy >= src_rows || map1_dataDy< 0 || map1_dataDy < 0);
      int4 bc =(map1_dataDx1 >= src_cols || map1_dataDy >= src_rows || map1_dataDx1 < 0 || map1_dataDy < 0);
      int4 cc =(map1_dataDx >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDx < 0);
      int4 dc =(map1_dataDx1 >= src_cols || map1_dataDy1 >= src_rows || map1_dataDy1 < 0 || map1_dataDy1 < 0);

      int16 acc = (int16)((int4)(ac.x), (int4)(ac.y), (int4)(ac.z), (int4)(ac.w));
      int16 bcc = (int16)((int4)(bc.x), (int4)(bc.y), (int4)(bc.z), (int4)(bc.w));
      int16 ccc = (int16)((int4)(cc.x), (int4)(cc.y), (int4)(cc.z), (int4)(cc.w));
      int16 dcc = (int16)((int4)(dc.x), (int4)(dc.y), (int4)(dc.z), (int4)(dc.w));
 
      uchar16 val = (uchar16)(nval, nval, nval, nval);
      a = (convert_uchar16(acc) == (uchar16)(0))? a : val;
      b = (convert_uchar16(bcc) == (uchar16)(0))? b : val;
      c = (convert_uchar16(ccc) == (uchar16)(0))? c : val;
      d = (convert_uchar16(dcc) == (uchar16)(0))? d : val;

      float16 U = (float16)((float4)(u.x), (float4)(u.y), (float4)(u.z), (float4)(u.w));
      float16 V = (float16)((float4)(v.x), (float4)(v.y), (float4)(v.z), (float4)(v.w));
      float16 Ud = (float16)((float4)(ud.x), (float4)(ud.y), (float4)(ud.z), (float4)(ud.w));
      float16 Vd = (float16)((float4)(vd.x), (float4)(vd.y), (float4)(vd.z), (float4)(vd.w));

      uchar16 dst_data = convert_uchar16_sat_rte((convert_float16(a))* Ud * Vd +(convert_float16(b))* U * Vd + (convert_float16(c))* Ud * V + (convert_float16(d)) * U * V );
    
      __global uchar16* D = (__global uchar16 *)(dst + dstStart);

      uchar16 dVal = *D;      
      int16 con = (Gx >= 0 && Gx < (dst_cols<<2) && y >= 0 && y < dst_rows);
      dst_data = (convert_uchar16(con) != (uchar16)(0)) ? dst_data : dVal;

      *D = dst_data;
    }
}

__kernel void remapLNSConstant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    {
        x = x << 4;
        int gx = x - (dst_offset&15);
        int16 Gx = (int16)(gx, gx+1, gx+2, gx+3, gx+4, gx+5, gx+6, gx+7, gx+8, gx+9, gx+10, gx+11, gx+12, gx+13, gx+14, gx+15);
        uchar4 nval =convert_uchar4_sat_rte(nVal);

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&15);

        int map1Start = y * map1_step + x + map1_offset - (dst_offset&15 );
        short8 map1_data;

        map1_data = *((__global short8 *)((__global char*)map1 + map1Start));
        int4 srcIdx = convert_int4(map1_data.odd) * src_step + (convert_int4(map1_data.even) <<(int4)(2)) + src_offset;
        uchar4 src_a, src_b, src_c, src_d;
        src_a = *((__global uchar4 *)((__global char*)src + srcIdx.s0));
        src_b = *((__global uchar4 *)((__global char*)src + srcIdx.s1));
        src_c = *((__global uchar4 *)((__global char*)src + srcIdx.s2));
        src_d = *((__global uchar4 *)((__global char*)src + srcIdx.s3));

        uchar16 dst_data;
        uchar4 dst_a, dst_b, dst_c, dst_d;
        dst_a = (map1_data.s0 >= src_cols || map1_data.s1 >= src_rows)? nval : src_a;
        dst_b = (map1_data.s2 >= src_cols || map1_data.s3 >= src_rows)? nval : src_b;
        dst_c = (map1_data.s4 >= src_cols || map1_data.s5 >= src_rows)? nval : src_c;
        dst_d = (map1_data.s6 >= src_cols || map1_data.s7 >= src_rows)? nval : src_d;

        dst_data = (uchar16)(dst_a, dst_b, dst_c, dst_d);
        __global uchar16* d = (__global uchar16 *)(dst + dstStart);

        uchar16 dVal = *d;      

        int16 con = (Gx >= 0 && Gx < (dst_cols<<2) && y >= 0 && y < dst_rows);
        dst_data = (convert_uchar16(con) != (uchar16)(0)) ? dst_data : dVal;

        *d = dst_data;

    }

}

__kernel void remapLNFConstant_C1_D5(__global float* dst, __global float const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < threadCols && y < dst_rows)
    {
      x = x << 4; 
      int gx = x - (dst_offset&15);
      int4 Gx = (int4)(gx, gx+4, gx+8, gx+12);

      float4 nval =convert_float4(nVal);
      float4 val = (float4)(nval.s0);
  
      int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&15);
      int map1Start = y * map1_step + (x << 1) + map1_offset - ((dst_offset & 15) << 1);
      float8 map1_data;

      map1_data = *((__global float8 *)((__global char*)map1 + map1Start));
      int8 map1_dataD = convert_int8(map1_data);
      float8 temp = map1_data - convert_float8(map1_dataD);

      float4 u = temp.even;
      float4 v = temp.odd;
      float4 ud = (float4)(1.0) - u;
      float4 vd = (float4)(1.0) - v;
      //float8 map1_dataU = map1_dataD + 1;

      int4 map1_dataDx = map1_dataD.even;
      int4 map1_dataDy = map1_dataD.odd;
      int4 map1_dataDx1 = map1_dataDx + (int4)(1);
      int4 map1_dataDy1 = map1_dataDy + (int4)(1);

      int4 src_StartU = map1_dataDy * src_step + (map1_dataDx << (int4)(2)) + src_offset;
      int4 src_StartD = src_StartU + src_step;
     /* 
      //not using the vload
      int4 src_StartU1 = src_StartU + (int4)(1);
      int4 src_StartD1 = src_StartD + (int4)(1);

      float4 a, b, c, d;
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
      */
      float2 aU, aD, bU, bD, cU, cD, dU, dD;

      aU = vload2(0, (__global float *)((__global char*)src + src_StartU.s0));
      bU = vload2(0, (__global float *)((__global char*)src + src_StartU.s1));
      cU = vload2(0, (__global float *)((__global char*)src + src_StartU.s2));
      dU = vload2(0, (__global float *)((__global char*)src + src_StartU.s3));
      aD = vload2(0, (__global float *)((__global char*)src + src_StartD.s0));
      bD = vload2(0, (__global float *)((__global char*)src + src_StartD.s1));
      cD = vload2(0, (__global float *)((__global char*)src + src_StartD.s2));
      dD = vload2(0, (__global float *)((__global char*)src + src_StartD.s3));

      float4 a, b, c, d;
      a = (float4)(aU.x, bU.x, cU.x, dU.x);
      b = (float4)(aU.y, bU.y, cU.y, dU.y);
      c = (float4)(aD.x, bD.x, cD.x, dD.x);
      d = (float4)(aD.y, bD.y, cD.y, dD.y);
      
      int4 ac =(map1_dataDx >= (int4)(src_cols) || map1_dataDy >= (int4)(src_rows) || map1_dataDy < (int4)(0) || map1_dataDy < (int4)(0));
      int4 bc =(map1_dataDx1 >= (int4)(src_cols) || map1_dataDy >= (int4)(src_rows) || map1_dataDx1 < (int4)(0) || map1_dataDy < (int4)(0));
      int4 cc =(map1_dataDx >= (int4)(src_cols) || map1_dataDy1 >= (int4)(src_rows) || map1_dataDy1 < (int4)(0) || map1_dataDx < (int4)(0));
      int4 dc =(map1_dataDx1 >= (int4)(src_cols) || map1_dataDy1 >= (int4)(src_rows) || map1_dataDy1 < (int4)(0) || map1_dataDy1 < (int4)(0));
      a = (convert_float4(ac) == (float4)(0))? a : val;
      b = (convert_float4(bc) == (float4)(0))? b : val;
      c = (convert_float4(cc) == (float4)(0))? c : val;
      d = (convert_float4(dc) == (float4)(0))? d : val;

      float4 dst_data = a * ud * vd + b * u * vd + c * ud * v + d * u * v ;
    
      __global float4* D = (__global float4 *)((__global char*)dst + dstStart);

      float4 dVal = *D;      
      int4 con = (Gx >= 0 && Gx < (dst_cols << 2) && y >= 0 && y < dst_rows);
      dst_data = (convert_float4(con) != (float4)(0)) ? dst_data : dVal;

      *D = dst_data;
    }
}

__kernel void remapLNF1Constant_C1_D5(__global float* dst, __global float const * restrict  src,
        __global float * map1, __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < threadCols && y < dst_rows)
    {
      x = x << 4; 
      int gx = x - (dst_offset&15);
      int4 Gx = (int4)(gx, gx+4, gx+8, gx+12);

      float4 nval =convert_float4(nVal);
      float4 val = (float4)(nval.s0);
  
      int dstStart = y * dst_step + x  + dst_offset - (dst_offset & 15);
      int map1Start = y * map1_step + x + map1_offset - (dst_offset & 15);
      float4 map1_data;
      float4 map2_data;

      map1_data = *((__global float4 *)((__global char*)map1 + map1Start));
      map2_data = *((__global float4 *)((__global char*)map2 + map1Start));
      float8 map_data = (float8)(map1_data.s0, map2_data.s0, map1_data.s1, map2_data.s1, map1_data.s2, map2_data.s2, map1_data.s3, map2_data.s3);
      int8 map1_dataD = convert_int8(map_data);
      float8 temp = map_data - convert_float8(map1_dataD);

      float4 u = temp.even;
      float4 v = temp.odd;
      float4 ud = (float4)(1.0) - u;
      float4 vd = (float4)(1.0) - v;
      //float8 map1_dataU = map1_dataD + 1;

      int4 map1_dataDx = map1_dataD.even;
      int4 map1_dataDy = map1_dataD.odd;
      int4 map1_dataDx1 = map1_dataDx + (int4)(1);
      int4 map1_dataDy1 = map1_dataDy + (int4)(1);

      int4 src_StartU = map1_dataDy * src_step + (map1_dataDx << (int4)(2)) + src_offset;
      int4 src_StartD = src_StartU + src_step;
     /* 
      //not using the vload
      int4 src_StartU1 = src_StartU + (int4)(1);
      int4 src_StartD1 = src_StartD + (int4)(1);

      float4 a, b, c, d;
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
      */
      float2 aU, aD, bU, bD, cU, cD, dU, dD;

      aU = vload2(0, (__global float *)((__global char*)src + src_StartU.s0));
      bU = vload2(0, (__global float *)((__global char*)src + src_StartU.s1));
      cU = vload2(0, (__global float *)((__global char*)src + src_StartU.s2));
      dU = vload2(0, (__global float *)((__global char*)src + src_StartU.s3));
      aD = vload2(0, (__global float *)((__global char*)src + src_StartD.s0));
      bD = vload2(0, (__global float *)((__global char*)src + src_StartD.s1));
      cD = vload2(0, (__global float *)((__global char*)src + src_StartD.s2));
      dD = vload2(0, (__global float *)((__global char*)src + src_StartD.s3));

      float4 a, b, c, d;
      a = (float4)(aU.x, bU.x, cU.x, dU.x);
      b = (float4)(aU.y, bU.y, cU.y, dU.y);
      c = (float4)(aD.x, bD.x, cD.x, dD.x);
      d = (float4)(aD.y, bD.y, cD.y, dD.y);
      
      int4 ac =(map1_dataDx >= (int4)(src_cols) || map1_dataDy >= (int4)(src_rows) || map1_dataDy < (int4)(0) || map1_dataDy < (int4)(0));
      int4 bc =(map1_dataDx1 >= (int4)(src_cols) || map1_dataDy >= (int4)(src_rows) || map1_dataDx1 < (int4)(0) || map1_dataDy < (int4)(0));
      int4 cc =(map1_dataDx >= (int4)(src_cols) || map1_dataDy1 >= (int4)(src_rows) || map1_dataDy1 < (int4)(0) || map1_dataDx < (int4)(0));
      int4 dc =(map1_dataDx1 >= (int4)(src_cols) || map1_dataDy1 >= (int4)(src_rows) || map1_dataDy1 < (int4)(0) || map1_dataDy1 < (int4)(0));
      a = (convert_float4(ac) == (float4)(0))? a : val;
      b = (convert_float4(bc) == (float4)(0))? b : val;
      c = (convert_float4(cc) == (float4)(0))? c : val;
      d = (convert_float4(dc) == (float4)(0))? d : val;

      float4 dst_data = a * ud * vd + b * u * vd + c * ud * v + d * u * v ;
    
      __global float4* D = (__global float4 *)((__global char*)dst + dstStart);

      float4 dVal = *D;      
      int4 con = (Gx >= 0 && Gx < (dst_cols << 2) && y >= 0 && y < dst_rows);
      dst_data = (convert_float4(con) != (float4)(0)) ? dst_data : dVal;

      *D = dst_data;
    }
}

__kernel void remapLNSConstant_C1_D5(__global float* dst, __global float const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows ,int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {     
        x = x << 4;

        int gx = x - (dst_offset&15);
        int4 Gx = (int4)(gx, gx+4, gx+8, gx+12);

        float4 nval =convert_float4(nVal);
        float val = nval.s0;

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&15);

        int map1Start = y * map1_step + x + map1_offset - (dst_offset&15);
        short8 map1_data;

        map1_data = *((__global short8 *)((__global char*)map1 + map1Start));
    
        int4 srcIdx = convert_int4(map1_data.odd) * src_step + (convert_int4(map1_data.even) << (int4)(2)) + src_offset;
    
        float4 src_data;
        src_data.s0 = *((__global float *)((__global char*)src + srcIdx.s0));
        src_data.s1 = *((__global float *)((__global char*)src + srcIdx.s1));
        src_data.s2 = *((__global float *)((__global char*)src + srcIdx.s2));
        src_data.s3 = *((__global float *)((__global char*)src + srcIdx.s3));
        float4 dst_data;
        
        dst_data.s0 = (map1_data.s0 >= src_cols || map1_data.s1 >= src_rows)? val : src_data.s0;
        dst_data.s1 = (map1_data.s2 >= src_cols || map1_data.s3 >= src_rows)? val : src_data.s1;
        dst_data.s2 = (map1_data.s4 >= src_cols || map1_data.s5 >= src_rows)? val : src_data.s2;
        dst_data.s3 = (map1_data.s6 >= src_cols || map1_data.s7 >= src_rows)? val : src_data.s3;
        
  
        __global float4* d = (__global float4 *)((__global uchar*)dst + dstStart);

        float4 dVal = *d;      

        int4 con = (Gx >= 0 && Gx < (dst_cols<<2) && y >= 0 && y < dst_rows);
        dst_data = (convert_float4(con) != (float4)(0)) ? dst_data : dVal;

        *d = dst_data;

    }

}


__kernel void remapLNFConstant_C4_D5(__global float * dst, __global float const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    {
      int dstIdx = y * dst_step + (x << 4) + dst_offset  ;
      int mapIdx = y * map1_step + (x << 3) + map1_offset ;
      float2 map1_data = *((__global float2 *)((__global char*)map1 + mapIdx));

      int2 map1_dataZ = convert_int2(map1_data);

      int mX = map1_dataZ.x;
      int mY = map1_dataZ.y;
      int mX1 = map1_dataZ.x + 1;
      int mY1 = map1_dataZ.y + 1;

      float u = map1_data.x - convert_float(map1_dataZ.x);
      float v = map1_data.y - convert_float(map1_dataZ.y);
      float ud = 1.0 - u;
      float vd = 1.0 - v;

      int srcIdx = map1_dataZ.y * src_step + (map1_dataZ.x << 4) + src_offset;
      float8 src_dataU = vload8(0,(__global float *)((__global char*)src + srcIdx));
      float8 src_dataD = vload8(0,(__global float *)((__global char*)src + srcIdx + src_step));

      float4 a = src_dataU.lo;
      float4 b = src_dataU.hi;
      float4 c = src_dataD.lo;
      float4 d = src_dataD.hi;

      float4 nval = convert_float4(nVal);
      a = (mX >= src_cols || mY >= src_rows ) ? nval : a;
      b = (mX1 >= src_cols || mY >= src_rows ) ? nval : b;
      c = (mX >= src_cols || mY1 >= src_rows ) ? nval : c;
      d = (mX1 >= src_cols || mY1 >= src_rows ) ? nval : d;

      float4 dst_data = a * ud * vd + b * u * vd + c * ud * v + d * u * v; 
      *((__global float4 *)((__global uchar*)dst + dstIdx)) =  a * ud * vd + b * u * vd + c * ud * v + d * u * v ;

    }
}

__kernel void remapLNF1Constant_C4_D5(__global float * dst, __global float const * restrict  src,
        __global float * map1, __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    {
      int dstIdx = y * dst_step + (x << 4) + dst_offset  ;
      int mapIdx = y * map1_step + (x << 2) + map1_offset ;
      float map1_data = *((__global float *)((__global char*)map1 + mapIdx));
      float map2_data = *((__global float *)((__global char*)map2 + mapIdx));
      float2 map_data = (float2)(map1_data, map2_data);
      int2 map1_dataZ = convert_int2(map_data);

      int mX = map1_dataZ.x;
      int mY = map1_dataZ.y;
      int mX1 = map1_dataZ.x + 1;
      int mY1 = map1_dataZ.y + 1;

      float u = map1_data - convert_float(map1_dataZ.x);
      float v = map2_data - convert_float(map1_dataZ.y);
      float ud = 1.0 - u;
      float vd = 1.0 - v;

      int srcIdx = map1_dataZ.y * src_step + (map1_dataZ.x << 4) + src_offset;
      float8 src_dataU = vload8(0,(__global float *)((__global char*)src + srcIdx));
      float8 src_dataD = vload8(0,(__global float *)((__global char*)src + srcIdx + src_step));

      float4 a = src_dataU.lo;
      float4 b = src_dataU.hi;
      float4 c = src_dataD.lo;
      float4 d = src_dataD.hi;

      float4 nval = convert_float4(nVal);
      a = (mX >= src_cols || mY >= src_rows ) ? nval : a;
      b = (mX1 >= src_cols || mY >= src_rows ) ? nval : b;
      c = (mX >= src_cols || mY1 >= src_rows ) ? nval : c;
      d = (mX1 >= src_cols || mY1 >= src_rows ) ? nval : d;

      float4 dst_data = a * ud * vd + b * u * vd + c * ud * v + d * u * v; 
      *((__global float4 *)((__global uchar*)dst + dstIdx)) =  a * ud * vd + b * u * vd + c * ud * v + d * u * v ;

    }
}


/*
////////////////////////////////////////////////////////////////////////
///////////////////using image buffer///////////////////////////////////
////////////////////////////////////////////////////////////////////////


__kernel void remapNNSConstant_C1_D0(__global unsigned char* dst, __read_only image2d_t  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, F4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    x = x << 2;
    if(x < threadCols && y < dst_rows)
    {
      int gx = x - (dst_offset&3);
      int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

      uchar4 nval =convert_uchar4(nVal);
      char val = nval.s0;

      int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&3);

      int map1Start = y * map1_step + (x << 2) + map1_offset - ((dst_offset & 3) << 2);
      short8 map1_data;

      map1_data = *((__global short8 *)((__global char*)map1 + map1Start));
        
      const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
          CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

      int4 src_data;
      src_data.x = read_imageui(src, sampler, (int2)((int)map1_data.s0, (int)map1_data.s1)).x;
      src_data.y = read_imageui(src, sampler, (int2)((int)map1_data.s2, (int)map1_data.s3)).x;
      src_data.z = read_imageui(src, sampler, (int2)((int)map1_data.s4, (int)map1_data.s5)).x;
      src_data.w = read_imageui(src, sampler, (int2)((int)map1_data.s6, (int)map1_data.s7)).x;

      int4 bcon = (convert_int4(map1_data.even) >= (int4)(src_cols) || convert_int4(map1_data.odd) >= (int4)(src_rows));
      uchar4 dst_data = (convert_uchar4(bcon != 0)) ? (uchar4)(val) : convert_uchar4(src_data);

      __global uchar4* d = (__global uchar4 *)(dst + dstStart);
      uchar4 dVal = *d;
      int4 con = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
      dst_data = (convert_uchar4(con) != (uchar4)(0)) ? dst_data : dVal;

      *d = dst_data;   
    }
}
*/
