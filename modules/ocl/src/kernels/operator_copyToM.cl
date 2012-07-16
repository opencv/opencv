//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
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
//

__kernel void copy_to_with_mask_C1_D0(
		__global const uchar* restrict srcMat,
		__global uchar* dstMat,
		__global const uchar* restrict maskMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		int maskStep,
		int maskoffset)
{
		int x=get_global_id(0)<<2;
		int y=get_global_id(1);

    int dst_addr_start = mad24((uint)y, (uint)dstStep_in_pixel, (uint)dstoffset_in_pixel); 
		int dst_addr_end = mad24((uint)y, (uint)dstStep_in_pixel, (uint)cols+dstoffset_in_pixel);
		int dstidx = mad24((uint)y, (uint)dstStep_in_pixel, (uint)x+ dstoffset_in_pixel) & (int)0xfffffffc;

    int vector_off = dstoffset_in_pixel & 3; 

		int srcidx = mad24((uint)y, (uint)srcStep_in_pixel, (uint)x + srcoffset_in_pixel - vector_off);		

    int mask_addr_start = mad24((uint)y, (uint)maskStep, (uint)maskoffset);
		int mask_addr_end = mad24((uint)y, (uint)maskStep, (uint)cols+maskoffset);
		int maskidx = mad24((uint)y, (uint)maskStep, (uint)x + maskoffset - vector_off);

		if ( (x < cols + dstoffset_in_pixel) & (y < rows) )
		{
        uchar4 src_data  = vload4(0, srcMat + srcidx);
        uchar4 mask_data = vload4(0, maskMat + maskidx);
        uchar4 dst_data  = *((__global uchar4 *)(dstMat + dstidx));
        uchar4 tmp_data;

        mask_data.x = ((maskidx + 0 >= mask_addr_start) && (maskidx + 0 < mask_addr_end)) ? mask_data.x : 0;
        mask_data.y = ((maskidx + 1 >= mask_addr_start) && (maskidx + 1 < mask_addr_end)) ? mask_data.y : 0;
        mask_data.z = ((maskidx + 2 >= mask_addr_start) && (maskidx + 2 < mask_addr_end)) ? mask_data.z : 0;
        mask_data.w = ((maskidx + 3 >= mask_addr_start) && (maskidx + 3 < mask_addr_end)) ? mask_data.w : 0;
			
        tmp_data.x = ((dstidx + 0 >= dst_addr_start) && (dstidx + 0 < dst_addr_end) && (mask_data.x)) 
                     ? src_data.x : dst_data.x;
        tmp_data.y = ((dstidx + 1 >= dst_addr_start) && (dstidx + 1 < dst_addr_end) && (mask_data.y)) 
                     ? src_data.y : dst_data.y;
        tmp_data.z = ((dstidx + 2 >= dst_addr_start) && (dstidx + 2 < dst_addr_end) && (mask_data.z)) 
                     ? src_data.z : dst_data.z;
        tmp_data.w = ((dstidx + 3 >= dst_addr_start) && (dstidx + 3 < dst_addr_end) && (mask_data.w)) 
                     ? src_data.w : dst_data.w;

        (*(__global uchar4*)(dstMat+dstidx)) = tmp_data;
		}
}

__kernel void copy_to_with_mask_C4_D0(
		__global const uchar4* restrict srcMat,
		__global uchar4* dstMat,
		__global const uchar* restrict maskMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		int maskStep,
		int maskoffset)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		int maskidx = mad24(y,maskStep,x+ maskoffset);
		uchar mask = maskMat[maskidx];		
		if ( (x < cols) & (y < rows) & mask)
		{
			dstMat[dstidx] = srcMat[srcidx];
		}
}
__kernel void copy_to_with_mask_C1_D4(
		__global const int* restrict srcMat,
		__global int* dstMat,
		__global const uchar* restrict maskMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		int maskStep,
		int maskoffset)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		int maskidx = mad24(y,maskStep,x+ maskoffset);
		uchar mask = maskMat[maskidx];		
		if ( (x < cols) & (y < rows) & mask)
		{
			dstMat[dstidx] = srcMat[srcidx];
		}
}
__kernel void copy_to_with_mask_C4_D4(
		__global const int4* restrict srcMat,
		__global int4* dstMat,
		__global const uchar* restrict maskMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		int maskStep,
		int maskoffset)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		int maskidx = mad24(y,maskStep,x+ maskoffset);
		uchar mask = maskMat[maskidx];		
		if ( (x < cols) & (y < rows) & mask)
		{
			dstMat[dstidx] = srcMat[srcidx];
		}
}
__kernel void copy_to_with_mask_C1_D5(
		__global const float* restrict srcMat,
		__global float* dstMat,
		__global const uchar* restrict maskMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		int maskStep,
		int maskoffset)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		int maskidx = mad24(y,maskStep,x+ maskoffset);
		uchar mask = maskMat[maskidx];		
		if ( (x < cols) & (y < rows) & mask)
		{
			dstMat[dstidx] = srcMat[srcidx];
		}
}
__kernel void copy_to_with_mask_C4_D5(
		__global const float4* restrict srcMat,
		__global float4* dstMat,
		__global const uchar* restrict maskMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		int maskStep,
		int maskoffset)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		int maskidx = mad24(y,maskStep,x+ maskoffset);
		uchar mask = maskMat[maskidx];		
		if ( (x < cols) & (y < rows) & mask)
		{
			dstMat[dstidx] = srcMat[srcidx];
		}
}
