
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


/////////////////////////////////////////////////////////
///////////////////////using buffer//////////////////////
/////////////////////////////////////////////////////////
__kernel void remapNNSConstant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, float4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {
        x = x << 2;
        int gx = x - (dst_offset&3);
        int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

        uchar4 nval =convert_uchar4(nVal);
        uchar4 val = (uchar4)(nval.s0);

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&3);

        int map1Start = y * map1_step + (x << 2) + map1_offset - ((dst_offset & 3) << 2);
        short8 map1_data;

        map1_data = *((__global short8 *)((__global char*)map1 + map1Start));
        int4 srcIdx = convert_int4(map1_data.odd) * src_step + convert_int4(map1_data.even) + src_offset;
   
        uchar4 con = convert_uchar4(convert_int4(map1_data.even) >= (int4)(src_cols) || convert_int4(map1_data.odd) >= (int4)(src_rows) || convert_int4(map1_data.even) < (int4)(0) || convert_int4(map1_data.odd) < (int4)(0));
        uchar4 src_data = val;

        if (con.s0 == 0)
        src_data.s0 = *(src + srcIdx.s0);
        if (con.s1 == 0)
        src_data.s1 = *(src + srcIdx.s1);
        if (con.s2 == 0)
        src_data.s2 = *(src + srcIdx.s2);
        if (con.s3 == 0)
        src_data.s3 = *(src + srcIdx.s3);
        
        uchar4 dst_data;
 
        __global uchar4* d = (__global uchar4 *)(dst + dstStart);

        uchar4 dVal = *d;      

        int4 dcon = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
        dst_data = (convert_uchar4(dcon) != convert_uchar4((int4)(0))) ? src_data : dVal;

        *d = dst_data;

    }

}

__kernel void remapNNFConstant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, float4 nVal)
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
    
        uchar4 src_data = val;
        uchar4 con = convert_uchar4(map1_dataZ.even >= (int4)(src_cols) || map1_dataZ.odd >= (int4)(src_rows) || map1_dataZ.even < (int4)(0) || map1_dataZ.odd < (int4)(0)); 

        if (con.s0 == 0)
        src_data.s0 = *(src + srcIdx.s0);
        if (con.s1 == 0)
        src_data.s1 = *(src + srcIdx.s1);
        if (con.s2 == 0)
        src_data.s2 = *(src + srcIdx.s2);
        if (con.s3 == 0)
        src_data.s3 = *(src + srcIdx.s3);
        uchar4 dst_data;
       // dst_data = convert_uchar4(map1_dataZ.even >= (int4)(src_cols) || map1_dataZ.odd >= (int4)(src_rows)) ? (uchar4)(val) : src_data;
        __global uchar4* d = (__global uchar4 *)(dst + dstStart);

        uchar4 dVal = *d;      

        int4 dcon = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
  
        dst_data = (convert_uchar4(dcon) != convert_uchar4((int4)(0))) ? src_data : dVal;
        *d = dst_data;
    }
}

__kernel void remapNNF1Constant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1,  __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, float4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {
        x = x << 2;
        int gx = x - (dst_offset&3);
        int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

        uchar4 nval =convert_uchar4(nVal);
        uchar4 val = (uchar4)(nval.s0);

        int dstStart = (y * dst_step + x  + dst_offset) - (dst_offset&3);

        int map1Start = y * map1_step + (x << 2) + map1_offset - ((dst_offset & 3) << 2);
        float4 map1_data;
        float4 map2_data;

        map1_data = *((__global float4 *)((__global char*)map1 + map1Start));
        map2_data = *((__global float4 *)((__global char*)map2 + map1Start));
        float8 map_data = (float8)(map1_data.s0, map2_data.s0, map1_data.s1, map2_data.s1, map1_data.s2, map2_data.s2, map1_data.s3, map2_data.s3);
        int8 map_dataZ = convert_int8_sat_rte(map_data);
        int4 srcIdx = map_dataZ.odd * src_step + map_dataZ.even + src_offset;
     
        uchar4 src_data = val;
        uchar4 con = convert_uchar4(map_dataZ.even >= (int4)(src_cols) || map_dataZ.odd >= (int4)(src_rows)|| map_dataZ.even < (int4)(0) || map_dataZ.odd < (int4)(0)); 

        if (con.s0 == 0)
        src_data.s0 = *(src + srcIdx.s0);
        if (con.s1 == 0)
        src_data.s1 = *(src + srcIdx.s1);
        if (con.s2 == 0)
        src_data.s2 = *(src + srcIdx.s2);
        if (con.s3 == 0)
        src_data.s3 = *(src + srcIdx.s3);
        uchar4 dst_data;
    
    //    dst_data = convert_uchar4(map_dataZ.even >= (int4)(src_cols) || map_dataZ.odd >= (int4)(src_rows)) ? (uchar4)(val) : src_data;
        __global uchar4* d = (__global uchar4 *)(dst + dstStart);

        uchar4 dVal = *d;      

        int4 dcon = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
  
        dst_data = (convert_uchar4(dcon) != convert_uchar4((int4)(0))) ? src_data : dVal;
        *d = dst_data;
    }
}


__kernel void remapNNSConstant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, float4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    {
         int dstIdx = y * dst_step + (x << 2) + dst_offset;
         int mapIdx = y * map1_step + (x << 2) + map1_offset;
         short2 map1_data = *((__global short2 *)((__global char*)map1 + mapIdx));
         int srcIdx = map1_data.y * src_step + (map1_data.x << 2) + src_offset;
         uchar4 nval = convert_uchar4(nVal);
         uchar4 src_data;
         if(map1_data.x >= src_cols || map1_data.y >= src_rows || map1_data.x <0 || map1_data.y < 0 )
         src_data = nval;
         else
         src_data = *((__global uchar4 *)((__global uchar *)src + srcIdx));
         *((__global uchar4 *)((__global uchar*)dst + dstIdx)) = src_data;


    }


}

__kernel void remapNNFConstant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, float4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    {
         int dstIdx = y * dst_step + (x << 2) + dst_offset;
         int mapIdx = y * map1_step + (x << 3) + map1_offset;
         float2 map1_data = *((__global float2 *)((__global char*)map1 + mapIdx));
         int2 map1_dataZ = convert_int2_sat_rte(map1_data);
         int srcIdx = map1_dataZ.y * src_step + (map1_dataZ.x << 2) + src_offset;
         uchar4 nval = convert_uchar4(nVal);
         uchar4 src_data;
         if(map1_dataZ.x >= src_cols || map1_dataZ.y >= src_rows || map1_dataZ.x < 0 || map1_dataZ.y < 0)
         src_data = nval;
         else
         src_data = *((__global uchar4 *)((__global uchar *)src + srcIdx));
         *((__global uchar4 *)((__global uchar*)dst + dstIdx)) = src_data;


    }

}

__kernel void remapNNF1Constant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1,  __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows, int threadCols, float4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    { 
         int dstIdx = y * dst_step + (x << 2) + dst_offset;
        int mapIdx = y * map1_step + (x << 2) + map1_offset;
        float map1_data = *((__global float *)((__global char*)map1 + mapIdx));
        float map2_data = *((__global float *)((__global char*)map2 + mapIdx));
        int srcIdx = convert_int_sat_rte(map2_data) * src_step + (convert_int_sat_rte(map1_data) << 2) + src_offset;
        uchar4 nval = convert_uchar4(nVal);
        uchar4 src_data;
         if(convert_int_sat_rte(map1_data) >= src_cols || convert_int_sat_rte(map2_data) >= src_rows || convert_int_sat_rte(map1_data) < 0 || convert_int_sat_rte(map2_data) < 0)
           src_data = nval;
        else
           src_data = *((__global uchar4 *)((__global uchar *)src + srcIdx));
         *((__global uchar4 *)((__global uchar*)dst + dstIdx)) = src_data;
    }
}

__kernel void remapNNSConstant_C1_D5(__global float* dst, __global float const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows ,int threadCols, float4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {
        int dstIdx = y * dst_step + (x << 2) + dst_offset;
        int mapIdx = y * map1_step + (x << 2) + map1_offset;
        short2 map1_data = *((__global short2 *)((__global char*)map1 + mapIdx));
        int srcIdx = map1_data.y * src_step + (map1_data.x << 2) + src_offset;
        float nval = convert_float(nVal.x);
        float src_data;
        if(map1_data.x >= src_cols || map1_data.y >= src_rows|| map1_data.x < 0 || map1_data.y < 0)
           src_data = nval;
        else
           src_data = *((__global float *)((__global uchar *)src + srcIdx));
        *((__global float *)((__global uchar*)dst + dstIdx)) = src_data;

 
    }


}

__kernel void remapNNFConstant_C1_D5(__global float* dst, __global float const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows ,int threadCols, float4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {
        int dstIdx = y * dst_step + (x << 2) + dst_offset;
        int mapIdx = y * map1_step + (x << 3) + map1_offset;
        float2 map1_data = *((__global float2 *)((__global char*)map1 + mapIdx));
        int2 map1_dataZ = convert_int2_sat_rte(map1_data);
        int srcIdx = map1_dataZ.y * src_step + (map1_dataZ.x << 2) + src_offset;
        float nval = convert_float(nVal.x);
        float src_data;
        if(map1_dataZ.x >= src_cols || map1_dataZ.y >= src_rows || map1_dataZ.x < 0 || map1_dataZ.y < 0)
           src_data = nval;
        else
           src_data = *((__global float *)((__global uchar *)src + srcIdx));
        *((__global float *)((__global uchar*)dst + dstIdx)) = src_data;

 
    }

}

__kernel void remapNNF1Constant_C1_D5(__global float* dst, __global float const * restrict  src,
        __global float * map1, __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows ,int threadCols, float4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
     
    if(x < threadCols && y < dst_rows)
    {
        int dstIdx = y * dst_step + (x << 2) + dst_offset;
        int mapIdx = y * map1_step + (x << 2) + map1_offset;
        float map1_data = *((__global float *)((__global char*)map1 + mapIdx));
        float map2_data = *((__global float *)((__global char*)map2 + mapIdx));
        float2 map_data = (float2)(map1_data, map2_data);
        int2 map1_dataZ = convert_int2_sat_rte(map_data);
        int srcIdx = map1_dataZ.y * src_step + (map1_dataZ.x << 2) + src_offset;
        float nval = convert_float(nVal.x);
        float src_data;

        if(map1_dataZ.x >= src_cols || map1_dataZ.y >= src_rows || map1_dataZ.x < 0 || map1_dataZ.y < 0)
           src_data = nval;
        else
           src_data = *((__global float *)((__global uchar *)src + srcIdx));
        *((__global float *)((__global uchar*)dst + dstIdx)) = src_data;

 
    }

}

__kernel void remapNNSConstant_C4_D5(__global float * dst, __global float const * restrict  src,
        __global short * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, float4 nVal)
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
      float4 src_data;
      if (map1_data.x <0 || map1_data.x >= src_cols || map1_data.y <0 || map1_data.y >= src_rows)
          src_data = nval;
      else
          src_data = *((__global float4 *)((__global uchar *)src + srcIdx));
      *((__global float4 *)((__global uchar*)dst + dstIdx)) = src_data; 

      
    }
}

__kernel void remapNNFConstant_C4_D5(__global float * dst, __global float const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, float4 nVal)
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
      float4 src_data = nval;
      if(map1_dataZ.x >= 0 && map1_dataZ.x < src_cols && map1_dataZ.y >=0 && map1_dataZ.y < src_rows)
      src_data = *((__global float4 *)((__global uchar *)src + srcIdx));
       *((__global float4 *)((__global uchar*)dst + dstIdx)) = src_data;
    }
}

__kernel void remapNNF1Constant_C4_D5(__global float * dst, __global float const * restrict  src,
        __global float * map1,  __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, float4 nVal)
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
      float4 src_data = nval;
      if(map1_dataZ.x >= 0 && map1_dataZ.x < src_cols && map1_dataZ.y >= 0 && map1_dataZ.y < src_rows)
      src_data = *((__global float4 *)((__global uchar *)src + srcIdx));
       *((__global float4 *)((__global uchar*)dst + dstIdx)) = src_data;
    }
}



__kernel void remapLNFConstant_C1_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, float4 nVal)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < threadCols && y < dst_rows)
    {
      x = x << 2; 
      int gx = x - (dst_offset&3);
      int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

      uchar4 nval =convert_uchar4(nVal);
      uchar4 val = (uchar4)(nval.s0);
  

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
      uchar4 a = val, b = val, c = val, d =val;

      if (map1_dataDx.s0 < src_cols && map1_dataDx.s0 >= 0 && map1_dataDy.s0 < src_rows && map1_dataDy.s0 >= 0)
          a.s0 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s0 * src_step + map1_dataDx.s0 + src_offset));
      if (map1_dataDx.s1 < src_cols && map1_dataDx.s1 >= 0 && map1_dataDy.s1 < src_rows && map1_dataDy.s1 >= 0)
          a.s1 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s1 * src_step + map1_dataDx.s1 + src_offset));
      if (map1_dataDx.s2 < src_cols && map1_dataDx.s2 >= 0 && map1_dataDy.s2 < src_rows && map1_dataDy.s2 >= 0)
          a.s2 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s2 * src_step + map1_dataDx.s2 + src_offset));
      if (map1_dataDx.s3 < src_cols && map1_dataDx.s3 >= 0 && map1_dataDy.s3 < src_rows && map1_dataDy.s3 >= 0)
          a.s3 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s3 * src_step + map1_dataDx.s3 + src_offset));

      if (map1_dataDx1.s0 < src_cols && map1_dataDx1.s0 >= 0 && map1_dataDy.s0 < src_rows && map1_dataDy.s0 >= 0)
          b.s0 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s0 * src_step + map1_dataDx1.s0 + src_offset));
      if (map1_dataDx1.s1 < src_cols && map1_dataDx1.s1 >= 0 && map1_dataDy.s1 < src_rows && map1_dataDy.s1 >= 0)
          b.s1 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s1 * src_step + map1_dataDx1.s1 + src_offset));
      if (map1_dataDx1.s2 < src_cols && map1_dataDx1.s2 >= 0 && map1_dataDy.s2 < src_rows && map1_dataDy.s2 >= 0)
          b.s2 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s2 * src_step + map1_dataDx1.s2 + src_offset));
      if (map1_dataDx1.s3 < src_cols && map1_dataDx1.s3 >= 0 && map1_dataDy.s3 < src_rows && map1_dataDy.s3 >= 0)
          b.s3 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s3 * src_step + map1_dataDx1.s3 + src_offset));

      if (map1_dataDx.s0 < src_cols && map1_dataDx.s0 >= 0 && map1_dataDy1.s0 < src_rows && map1_dataDy1.s0 >= 0)
          c.s0 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s0 * src_step + map1_dataDx.s0 + src_offset));
      if (map1_dataDx.s1 < src_cols && map1_dataDx.s1 >= 0 && map1_dataDy1.s1 < src_rows && map1_dataDy1.s1 >= 0)
          c.s1 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s1 * src_step + map1_dataDx.s1 + src_offset));
      if (map1_dataDx.s2 < src_cols && map1_dataDx.s2 >= 0 && map1_dataDy1.s2 < src_rows && map1_dataDy1.s2 >= 0)
          c.s2 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s2 * src_step + map1_dataDx.s2 + src_offset));
      if (map1_dataDx.s3 < src_cols && map1_dataDx.s3 >= 0 && map1_dataDy1.s3 < src_rows && map1_dataDy1.s3 >= 0)
          c.s3 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s3 * src_step + map1_dataDx.s3 + src_offset));

      if (map1_dataDx1.s0 < src_cols && map1_dataDx1.s0 >= 0 && map1_dataDy1.s0 < src_rows && map1_dataDy1.s0 >= 0)
          d.s0 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s0 * src_step + map1_dataDx1.s0 + src_offset));
      if (map1_dataDx1.s1 < src_cols && map1_dataDx1.s1 >= 0 && map1_dataDy1.s1 < src_rows && map1_dataDy1.s1 >= 0)
          d.s1 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s1 * src_step + map1_dataDx1.s1 + src_offset));
      if (map1_dataDx1.s2 < src_cols && map1_dataDx1.s2 >= 0 && map1_dataDy1.s2 < src_rows && map1_dataDy1.s2 >= 0)
          d.s2 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s2 * src_step + map1_dataDx1.s2 + src_offset));
      if (map1_dataDx1.s3 < src_cols && map1_dataDx1.s3 >= 0 && map1_dataDy1.s3 < src_rows && map1_dataDy1.s3 >= 0)
          d.s3 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s3 * src_step + map1_dataDx1.s3 + src_offset));
 
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
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, float4 nVal)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < threadCols && y < dst_rows)
    {
      x = x << 2; 
      int gx = x - (dst_offset&3);
      int4 Gx = (int4)(gx, gx+1, gx+2, gx+3);

      uchar4 nval =convert_uchar4(nVal);
      uchar4 val = (uchar4)(nval.s0);
  

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

      uchar4 a = val, b = val, c = val, d =val;
      if (map1_dataDx.s0 < src_cols && map1_dataDx.s0 >= 0 && map1_dataDy.s0 < src_rows && map1_dataDy.s0 >= 0)
          a.s0 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s0 * src_step + map1_dataDx.s0 + src_offset));
      if (map1_dataDx.s1 < src_cols && map1_dataDx.s1 >= 0 && map1_dataDy.s1 < src_rows && map1_dataDy.s1 >= 0)
          a.s1 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s1 * src_step + map1_dataDx.s1 + src_offset));
      if (map1_dataDx.s2 < src_cols && map1_dataDx.s2 >= 0 && map1_dataDy.s2 < src_rows && map1_dataDy.s2 >= 0)
          a.s2 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s2 * src_step + map1_dataDx.s2 + src_offset));
      if (map1_dataDx.s3 < src_cols && map1_dataDx.s3 >= 0 && map1_dataDy.s3 < src_rows && map1_dataDy.s3 >= 0)
          a.s3 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s3 * src_step + map1_dataDx.s3 + src_offset));

      if (map1_dataDx1.s0 < src_cols && map1_dataDx1.s0 >= 0 && map1_dataDy.s0 < src_rows && map1_dataDy.s0 >= 0)
          b.s0 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s0 * src_step + map1_dataDx1.s0 + src_offset));
      if (map1_dataDx1.s1 < src_cols && map1_dataDx1.s1 >= 0 && map1_dataDy.s1 < src_rows && map1_dataDy.s1 >= 0)
          b.s1 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s1 * src_step + map1_dataDx1.s1 + src_offset));
      if (map1_dataDx1.s2 < src_cols && map1_dataDx1.s2 >= 0 && map1_dataDy.s2 < src_rows && map1_dataDy.s2 >= 0)
          b.s2 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s2 * src_step + map1_dataDx1.s2 + src_offset));
      if (map1_dataDx1.s3 < src_cols && map1_dataDx1.s3 >= 0 && map1_dataDy.s3 < src_rows && map1_dataDy.s3 >= 0)
          b.s3 = *((__global uchar*)((__global uchar *)src + map1_dataDy.s3 * src_step + map1_dataDx1.s3 + src_offset));

      if (map1_dataDx.s0 < src_cols && map1_dataDx.s0 >= 0 && map1_dataDy1.s0 < src_rows && map1_dataDy1.s0 >= 0)
          c.s0 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s0 * src_step + map1_dataDx.s0 + src_offset));
      if (map1_dataDx.s1 < src_cols && map1_dataDx.s1 >= 0 && map1_dataDy1.s1 < src_rows && map1_dataDy1.s1 >= 0)
          c.s1 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s1 * src_step + map1_dataDx.s1 + src_offset));
      if (map1_dataDx.s2 < src_cols && map1_dataDx.s2 >= 0 && map1_dataDy1.s2 < src_rows && map1_dataDy1.s2 >= 0)
          c.s2 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s2 * src_step + map1_dataDx.s2 + src_offset));
      if (map1_dataDx.s3 < src_cols && map1_dataDx.s3 >= 0 && map1_dataDy1.s3 < src_rows && map1_dataDy1.s3 >= 0)
          c.s3 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s3 * src_step + map1_dataDx.s3 + src_offset));

      if (map1_dataDx1.s0 < src_cols && map1_dataDx1.s0 >= 0 && map1_dataDy1.s0 < src_rows && map1_dataDy1.s0 >= 0)
          d.s0 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s0 * src_step + map1_dataDx1.s0 + src_offset));
      if (map1_dataDx1.s1 < src_cols && map1_dataDx1.s1 >= 0 && map1_dataDy1.s1 < src_rows && map1_dataDy1.s1 >= 0)
          d.s1 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s1 * src_step + map1_dataDx1.s1 + src_offset));
      if (map1_dataDx1.s2 < src_cols && map1_dataDx1.s2 >= 0 && map1_dataDy1.s2 < src_rows && map1_dataDy1.s2 >= 0)
          d.s2 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s2 * src_step + map1_dataDx1.s2 + src_offset));
      if (map1_dataDx1.s3 < src_cols && map1_dataDx1.s3 >= 0 && map1_dataDy1.s3 < src_rows && map1_dataDy1.s3 >= 0)
          d.s3 = *((__global uchar*)((__global uchar *)src + map1_dataDy1.s3 * src_step + map1_dataDx1.s3 + src_offset));
 

      uchar4 dst_data = convert_uchar4_sat_rte((convert_float4(a))* ud * vd +(convert_float4(b))* u * vd + (convert_float4(c))* ud * v + (convert_float4(d)) * u * v );
    
      __global uchar4* D = (__global uchar4 *)(dst + dstStart);

      uchar4 dVal = *D;      
      int4 con = (Gx >= 0 && Gx < dst_cols && y >= 0 && y < dst_rows);
      dst_data = (convert_uchar4(con) != (uchar4)(0)) ? dst_data : dVal;

      *D = dst_data;
    }
}



__kernel void remapLNFConstant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, float4 nVal)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < threadCols && y < dst_rows)
    {
        int dstIdx = y * dst_step + (x << 2) + dst_offset;
        int mapIdx = y * map1_step + (x << 3) + map1_offset;
        float2 map_data = *((__global float2 *)((__global char*)map1 + mapIdx));
        int2 map_dataA = convert_int2(map_data);
        float2 u = map_data - convert_float2(map_dataA);
        int2 map_dataB = (int2)(map_dataA.x + 1, map_dataA.y);
        int2 map_dataC = (int2)(map_dataA.x, map_dataA.y + 1);
        int2 map_dataD = (int2)(map_dataA.x + 1, map_dataA.y +1);
        uchar4 nval = convert_uchar4(nVal);
        uchar4 a, b, c , d;
        if(map_dataA.x < 0 || map_dataA.x >= src_cols || map_dataA.y >= src_rows || map_dataA.y < 0)
        a = nval;
        else
        a = *((__global uchar4 *)((__global uchar *)src + map_dataA.y * src_step + (map_dataA.x<<2) + src_offset ));
        if(map_dataB.x < 0 || map_dataB.x >= src_cols || map_dataB.y >= src_rows || map_dataB.y < 0)
        b = nval;
        else
        b = *((__global uchar4 *)((__global uchar *)src + map_dataB.y * src_step + (map_dataB.x<<2) + src_offset ));

        if(map_dataC.x < 0 || map_dataC.x >= src_cols || map_dataC.y >= src_rows || map_dataC.y < 0)
        c = nval;
        else
        c = *((__global uchar4 *)((__global uchar *)src + map_dataC.y * src_step + (map_dataC.x<<2) + src_offset ));

        if(map_dataD.x < 0 || map_dataD.x >= src_cols || map_dataD.y >= src_rows || map_dataD.y < 0)
        d = nval;
        else
        d = *((__global uchar4 *)((__global uchar *)src + map_dataD.y * src_step + (map_dataD.x<<2) + src_offset ));
        float4 dst_data = convert_float4(a)*((float4)(1.0-u.x)*((float4)(1.0-u.y))) + convert_float4(b)*((float4)(u.x))*((float4)(1.0-u.y)) + convert_float4(c)*((float4)(1.0-u.x))*((float4)(u.y)) + convert_float4(d)*((float4)(u.x))*((float4)(u.y));
        *((__global uchar4 *)((__global uchar*)dst + dstIdx)) = convert_uchar4_sat_rte(dst_data);


    }

}
__kernel void remapLNF1Constant_C4_D0(__global unsigned char* dst, __global unsigned char const * restrict  src,
        __global float * map1,  __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, float4 nVal)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x < threadCols && y < dst_rows)
    {
        int dstIdx = y * dst_step + (x << 2) + dst_offset;
        int mapIdx = y * map1_step + (x << 2) + map1_offset;
        float map1_data = *((__global float *)((__global char*)map1 + mapIdx));
        float map2_data = *((__global float *)((__global char*)map2 + mapIdx));
        float2 map_data = (float2)(map1_data, map2_data);
        int2 map_dataA = convert_int2(map_data);
        float2 u = map_data - convert_float2(map_dataA);
        int2 map_dataB = (int2)(map_dataA.x + 1, map_dataA.y);
        int2 map_dataC = (int2)(map_dataA.x, map_dataA.y + 1);
        int2 map_dataD = (int2)(map_dataA.x + 1, map_dataA.y +1);
        uchar4 nval = convert_uchar4(nVal);
        uchar4 a, b, c , d;
        if(map_dataA.x < 0 || map_dataA.x >= src_cols || map_dataA.y >= src_rows || map_dataA.y < 0)
        a = nval;
        else
        a = *((__global uchar4 *)((__global uchar *)src + map_dataA.y * src_step + (map_dataA.x<<2) + src_offset ));
        if(map_dataB.x < 0 || map_dataB.x >= src_cols || map_dataB.y >= src_rows || map_dataB.y < 0)
        b = nval;
        else
        b = *((__global uchar4 *)((__global uchar *)src + map_dataB.y * src_step + (map_dataB.x<<2) + src_offset ));

        if(map_dataC.x < 0 || map_dataC.x >= src_cols || map_dataC.y >= src_rows || map_dataC.y < 0)
        c = nval;
        else
        c = *((__global uchar4 *)((__global uchar *)src + map_dataC.y * src_step + (map_dataC.x<<2) + src_offset ));

        if(map_dataD.x < 0 || map_dataD.x >= src_cols || map_dataD.y >= src_rows || map_dataD.y < 0)
        d = nval;
        else
        d = *((__global uchar4 *)((__global uchar *)src + map_dataD.y * src_step + (map_dataD.x<<2) + src_offset ));
        float4 dst_data = convert_float4(a)*((float4)(1.0-u.x)*((float4)(1.0-u.y))) + convert_float4(b)*((float4)(u.x))*((float4)(1.0-u.y)) + convert_float4(c)*((float4)(1.0-u.x))*((float4)(u.y)) + convert_float4(d)*((float4)(u.x))*((float4)(u.y));
        *((__global uchar4 *)((__global uchar*)dst + dstIdx)) = convert_uchar4_sat_rte(dst_data);



    }
}



__kernel void remapLNFConstant_C1_D5(__global float* dst, __global float const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, float4 nVal)
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

      float4 a = val, b = val, c = val, d = val;
      if (map1_dataDx.s0 < src_cols && map1_dataDx.s0 >= 0 && map1_dataDy.s0 < src_rows && map1_dataDy.s0 >= 0)
          a.s0 = *((__global float*)((__global uchar *)src + map1_dataDy.s0 * src_step + (map1_dataDx.s0 << 2) + src_offset));
      if (map1_dataDx.s1 < src_cols && map1_dataDx.s1 >= 0 && map1_dataDy.s1 < src_rows && map1_dataDy.s1 >= 0)
          a.s1 = *((__global float*)((__global uchar *)src + map1_dataDy.s1 * src_step + (map1_dataDx.s1 << 2) + src_offset));
      if (map1_dataDx.s2 < src_cols && map1_dataDx.s2 >= 0 && map1_dataDy.s2 < src_rows && map1_dataDy.s2 >= 0)
          a.s2 = *((__global float*)((__global uchar *)src + map1_dataDy.s2 * src_step + (map1_dataDx.s2 << 2) + src_offset));
      if (map1_dataDx.s3 < src_cols && map1_dataDx.s3 >= 0 && map1_dataDy.s3 < src_rows && map1_dataDy.s3 >= 0)
          a.s3 = *((__global float*)((__global uchar *)src + map1_dataDy.s3 * src_step + (map1_dataDx.s3 << 2) + src_offset));

      if (map1_dataDx1.s0 < src_cols && map1_dataDx1.s0 >= 0 && map1_dataDy.s0 < src_rows && map1_dataDy.s0 >= 0)
          b.s0 = *((__global float*)((__global uchar *)src + map1_dataDy.s0 * src_step + (map1_dataDx1.s0 << 2) + src_offset));
      if (map1_dataDx1.s1 < src_cols && map1_dataDx1.s1 >= 0 && map1_dataDy.s1 < src_rows && map1_dataDy.s1 >= 0)
          b.s1 = *((__global float*)((__global uchar *)src + map1_dataDy.s1 * src_step + (map1_dataDx1.s1 << 2) + src_offset));
      if (map1_dataDx1.s2 < src_cols && map1_dataDx1.s2 >= 0 && map1_dataDy.s2 < src_rows && map1_dataDy.s2 >= 0)
          b.s2 = *((__global float*)((__global uchar *)src + map1_dataDy.s2 * src_step + (map1_dataDx1.s2 << 2) + src_offset));
      if (map1_dataDx1.s3 < src_cols && map1_dataDx1.s3 >= 0 && map1_dataDy.s3 < src_rows && map1_dataDy.s3 >= 0)
          b.s3 = *((__global float*)((__global uchar *)src + map1_dataDy.s3 * src_step + (map1_dataDx1.s3 << 2) + src_offset));

      if (map1_dataDx.s0 < src_cols && map1_dataDx.s0 >= 0 && map1_dataDy1.s0 < src_rows && map1_dataDy1.s0 >= 0)
          c.s0 = *((__global float*)((__global uchar *)src + map1_dataDy1.s0 * src_step + (map1_dataDx.s0 << 2) + src_offset));
      if (map1_dataDx.s1 < src_cols && map1_dataDx.s1 >= 0 && map1_dataDy1.s1 < src_rows && map1_dataDy1.s1 >= 0)
          c.s1 = *((__global float*)((__global uchar *)src + map1_dataDy1.s1 * src_step + (map1_dataDx.s1 << 2) + src_offset));
      if (map1_dataDx.s2 < src_cols && map1_dataDx.s2 >= 0 && map1_dataDy1.s2 < src_rows && map1_dataDy1.s2 >= 0)
          c.s2 = *((__global float*)((__global uchar *)src + map1_dataDy1.s2 * src_step + (map1_dataDx.s2 << 2) + src_offset));
      if (map1_dataDx.s3 < src_cols && map1_dataDx.s3 >= 0 && map1_dataDy1.s3 < src_rows && map1_dataDy1.s3 >= 0)
          c.s3 = *((__global float*)((__global uchar *)src + map1_dataDy1.s3 * src_step + (map1_dataDx.s3 << 2) + src_offset));

      if (map1_dataDx1.s0 < src_cols && map1_dataDx1.s0 >= 0 && map1_dataDy1.s0 < src_rows && map1_dataDy1.s0 >= 0)
          d.s0 = *((__global float*)((__global uchar *)src + map1_dataDy1.s0 * src_step + (map1_dataDx1.s0 << 2) + src_offset));
      if (map1_dataDx1.s1 < src_cols && map1_dataDx1.s1 >= 0 && map1_dataDy1.s1 < src_rows && map1_dataDy1.s1 >= 0)
          d.s1 = *((__global float*)((__global uchar *)src + map1_dataDy1.s1 * src_step + (map1_dataDx1.s1 << 2) + src_offset));
      if (map1_dataDx1.s2 < src_cols && map1_dataDx1.s2 >= 0 && map1_dataDy1.s2 < src_rows && map1_dataDy1.s2 >= 0)
          d.s2 = *((__global float*)((__global uchar *)src + map1_dataDy1.s2 * src_step + (map1_dataDx1.s2 << 2) + src_offset));
      if (map1_dataDx1.s3 < src_cols && map1_dataDx1.s3 >= 0 && map1_dataDy1.s3 < src_rows && map1_dataDy1.s3 >= 0)
          d.s3 = *((__global float*)((__global uchar *)src + map1_dataDy1.s3 * src_step + (map1_dataDx1.s3 << 2) + src_offset));
    
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
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, float4 nVal)
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

      float4 a = val, b = val, c = val, d = val;
      if (map1_dataDx.s0 < src_cols && map1_dataDx.s0 >= 0 && map1_dataDy.s0 < src_rows && map1_dataDy.s0 >= 0)
          a.s0 = *((__global float*)((__global uchar *)src + map1_dataDy.s0 * src_step + (map1_dataDx.s0 << 2) + src_offset));
      if (map1_dataDx.s1 < src_cols && map1_dataDx.s1 >= 0 && map1_dataDy.s1 < src_rows && map1_dataDy.s1 >= 0)
          a.s1 = *((__global float*)((__global uchar *)src + map1_dataDy.s1 * src_step + (map1_dataDx.s1 << 2) + src_offset));
      if (map1_dataDx.s2 < src_cols && map1_dataDx.s2 >= 0 && map1_dataDy.s2 < src_rows && map1_dataDy.s2 >= 0)
          a.s2 = *((__global float*)((__global uchar *)src + map1_dataDy.s2 * src_step + (map1_dataDx.s2 << 2) + src_offset));
      if (map1_dataDx.s3 < src_cols && map1_dataDx.s3 >= 0 && map1_dataDy.s3 < src_rows && map1_dataDy.s3 >= 0)
          a.s3 = *((__global float*)((__global uchar *)src + map1_dataDy.s3 * src_step + (map1_dataDx.s3 << 2) + src_offset));

      if (map1_dataDx1.s0 < src_cols && map1_dataDx1.s0 >= 0 && map1_dataDy.s0 < src_rows && map1_dataDy.s0 >= 0)
          b.s0 = *((__global float*)((__global uchar *)src + map1_dataDy.s0 * src_step + (map1_dataDx1.s0 << 2) + src_offset));
      if (map1_dataDx1.s1 < src_cols && map1_dataDx1.s1 >= 0 && map1_dataDy.s1 < src_rows && map1_dataDy.s1 >= 0)
          b.s1 = *((__global float*)((__global uchar *)src + map1_dataDy.s1 * src_step + (map1_dataDx1.s1 << 2) + src_offset));
      if (map1_dataDx1.s2 < src_cols && map1_dataDx1.s2 >= 0 && map1_dataDy.s2 < src_rows && map1_dataDy.s2 >= 0)
          b.s2 = *((__global float*)((__global uchar *)src + map1_dataDy.s2 * src_step + (map1_dataDx1.s2 << 2) + src_offset));
      if (map1_dataDx1.s3 < src_cols && map1_dataDx1.s3 >= 0 && map1_dataDy.s3 < src_rows && map1_dataDy.s3 >= 0)
          b.s3 = *((__global float*)((__global uchar *)src + map1_dataDy.s3 * src_step + (map1_dataDx1.s3 << 2) + src_offset));

      if (map1_dataDx.s0 < src_cols && map1_dataDx.s0 >= 0 && map1_dataDy1.s0 < src_rows && map1_dataDy1.s0 >= 0)
          c.s0 = *((__global float*)((__global uchar *)src + map1_dataDy1.s0 * src_step + (map1_dataDx.s0 << 2) + src_offset));
      if (map1_dataDx.s1 < src_cols && map1_dataDx.s1 >= 0 && map1_dataDy1.s1 < src_rows && map1_dataDy1.s1 >= 0)
          c.s1 = *((__global float*)((__global uchar *)src + map1_dataDy1.s1 * src_step + (map1_dataDx.s1 << 2) + src_offset));
      if (map1_dataDx.s2 < src_cols && map1_dataDx.s2 >= 0 && map1_dataDy1.s2 < src_rows && map1_dataDy1.s2 >= 0)
          c.s2 = *((__global float*)((__global uchar *)src + map1_dataDy1.s2 * src_step + (map1_dataDx.s2 << 2) + src_offset));
      if (map1_dataDx.s3 < src_cols && map1_dataDx.s3 >= 0 && map1_dataDy1.s3 < src_rows && map1_dataDy1.s3 >= 0)
          c.s3 = *((__global float*)((__global uchar *)src + map1_dataDy1.s3 * src_step + (map1_dataDx.s3 << 2) + src_offset));

      if (map1_dataDx1.s0 < src_cols && map1_dataDx1.s0 >= 0 && map1_dataDy1.s0 < src_rows && map1_dataDy1.s0 >= 0)
          d.s0 = *((__global float*)((__global uchar *)src + map1_dataDy1.s0 * src_step + (map1_dataDx1.s0 << 2) + src_offset));
      if (map1_dataDx1.s1 < src_cols && map1_dataDx1.s1 >= 0 && map1_dataDy1.s1 < src_rows && map1_dataDy1.s1 >= 0)
          d.s1 = *((__global float*)((__global uchar *)src + map1_dataDy1.s1 * src_step + (map1_dataDx1.s1 << 2) + src_offset));
      if (map1_dataDx1.s2 < src_cols && map1_dataDx1.s2 >= 0 && map1_dataDy1.s2 < src_rows && map1_dataDy1.s2 >= 0)
          d.s2 = *((__global float*)((__global uchar *)src + map1_dataDy1.s2 * src_step + (map1_dataDx1.s2 << 2) + src_offset));
      if (map1_dataDx1.s3 < src_cols && map1_dataDx1.s3 >= 0 && map1_dataDy1.s3 < src_rows && map1_dataDy1.s3 >= 0)
          d.s3 = *((__global float*)((__global uchar *)src + map1_dataDy1.s3 * src_step + (map1_dataDx1.s3 << 2) + src_offset));
 
      
      float4 dst_data = a * ud * vd + b * u * vd + c * ud * v + d * u * v ;
    
      __global float4* D = (__global float4 *)((__global char*)dst + dstStart);

      float4 dVal = *D;      
      int4 con = (Gx >= 0 && Gx < (dst_cols << 2) && y >= 0 && y < dst_rows);
      dst_data = (convert_float4(con) != (float4)(0)) ? dst_data : dVal;

      *D = dst_data;
    }
}



__kernel void remapLNFConstant_C4_D5(__global float * dst, __global float const * restrict  src,
        __global float * map1, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, float4 nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < threadCols && y < dst_rows)
    {
      int dstIdx = y * dst_step + (x << 4) + dst_offset  ;
      int mapIdx = y * map1_step + (x << 3) + map1_offset ;
      float2 map_data = *((__global float2 *)((__global char*)map1 + mapIdx));
      int2 map_dataA = convert_int2(map_data);
      float2 u = map_data - convert_float2(map_dataA);
      int2 map_dataB = (int2)(map_dataA.x + 1, map_dataA.y);
      int2 map_dataC = (int2)(map_dataA.x, map_dataA.y + 1);
      int2 map_dataD = (int2)(map_dataA.x + 1, map_dataA.y +1);
      float4 nval = convert_float4(nVal);
      float4 a, b, c , d;
      if(map_dataA.x < 0 || map_dataA.x >= src_cols || map_dataA.y >= src_rows || map_dataA.y < 0)
      a = nval;
      else
      a = *((__global float4 *)((__global uchar *)src + map_dataA.y * src_step + (map_dataA.x<<4) + src_offset ));
      if(map_dataB.x < 0 || map_dataB.x >= src_cols || map_dataB.y >= src_rows || map_dataB.y < 0)
      b = nval;
      else
      b = *((__global float4 *)((__global uchar *)src + map_dataB.y * src_step + (map_dataB.x<<4) + src_offset ));

      if(map_dataC.x < 0 || map_dataC.x >= src_cols || map_dataC.y >= src_rows || map_dataC.y < 0)
      c = nval;
      else
      c = *((__global float4 *)((__global uchar *)src + map_dataC.y * src_step + (map_dataC.x<<4) + src_offset ));

      if(map_dataD.x < 0 || map_dataD.x >= src_cols || map_dataD.y >= src_rows || map_dataD.y < 0)
      d = nval;
      else
      d = *((__global float4 *)((__global uchar *)src + map_dataD.y * src_step + (map_dataD.x<<4) + src_offset ));

      float4 dst_data = a * ((float4)(1.0-u.x)) * ((float4)(1.0-u.y)) + b *((float4)(u.x)) * ((float4)(1.0-u.y)) + c * ((float4)(1.0-u.x)) *((float4)(u.y)) + d *((float4)(u.x)) *((float4)(u.y)); 
      *((__global float4 *)((__global uchar*)dst + dstIdx)) =  dst_data ;

    }
}

__kernel void remapLNF1Constant_C4_D5(__global float * dst, __global float const * restrict  src,
        __global float * map1, __global float * map2, int dst_offset, int src_offset, int map1_offset, int dst_step, int src_step,
        int map1_step, int src_cols, int src_rows, int dst_cols, int dst_rows, int map1_cols, int map1_rows , int threadCols, float4 nVal)
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
      int2 map_dataA = convert_int2(map_data);
      float2 u = map_data - convert_float2(map_dataA);
      int2 map_dataB = (int2)(map_dataA.x + 1, map_dataA.y);
      int2 map_dataC = (int2)(map_dataA.x, map_dataA.y + 1);
      int2 map_dataD = (int2)(map_dataA.x + 1, map_dataA.y +1);
      float4 nval = convert_float4(nVal);
      float4 a, b, c , d;
      if(map_dataA.x < 0 || map_dataA.x >= src_cols || map_dataA.y >= src_rows || map_dataA.y < 0)
      a = nval;
      else
      a = *((__global float4 *)((__global uchar *)src + map_dataA.y * src_step + (map_dataA.x<<4) + src_offset ));
      if(map_dataB.x < 0 || map_dataB.x >= src_cols || map_dataB.y >= src_rows || map_dataB.y < 0)
      b = nval;
      else
      b = *((__global float4 *)((__global uchar *)src + map_dataB.y * src_step + (map_dataB.x<<4) + src_offset ));

      if(map_dataC.x < 0 || map_dataC.x >= src_cols || map_dataC.y >= src_rows || map_dataC.y < 0)
      c = nval;
      else
      c = *((__global float4 *)((__global uchar *)src + map_dataC.y * src_step + (map_dataC.x<<4) + src_offset ));

      if(map_dataD.x < 0 || map_dataD.x >= src_cols || map_dataD.y >= src_rows || map_dataD.y < 0)
      d = nval;
      else
      d = *((__global float4 *)((__global uchar *)src + map_dataD.y * src_step + (map_dataD.x<<4) + src_offset ));

      float4 dst_data = a * ((float4)(1.0-u.x)) * ((float4)(1.0-u.y)) + b *((float4)(u.x)) * ((float4)(1.0-u.y)) + c * ((float4)(1.0-u.x)) *((float4)(u.y)) + d *((float4)(u.x)) *((float4)(u.y)); 
      *((__global float4 *)((__global uchar*)dst + dstIdx)) =  dst_data ;


    }
}



