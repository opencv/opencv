//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
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
//


/*
__kernel void set_to_with_mask_C1_D0(
		float4 scalar,
		__global uchar* dstMat,
		int cols,
		int rows,
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 		
        __global const uchar * maskMat,
		int maskStep,
		int maskoffset)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		int maskidx = mad24(y,maskStep,x+ maskoffset);
		uchar mask = maskMat[maskidx];		
		if ( (x < cols) & (y < rows) & mask)
		{
			dstMat[dstidx] = convert_uchar_sat(scalar.x);
		}

}
*/
//#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void set_to_with_mask_C1_D0(
		uchar scalar,
		__global uchar* dstMat,
		int cols,
		int rows,
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 		
        __global const uchar * restrict maskMat,
		int maskStep,
		int maskoffset)
{
		int x=get_global_id(0)<<2;
		int y=get_global_id(1);
		int dst_addr_start = mad24(y,dstStep_in_pixel,dstoffset_in_pixel);
		int dst_addr_end = mad24(y,dstStep_in_pixel,cols+dstoffset_in_pixel);
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel & (int)0xfffffffc);
		int mask_addr_start = mad24(y,maskStep,maskoffset);
		int mask_addr_end = mad24(y,maskStep,cols+maskoffset);
		int maskidx = mad24(y,maskStep,x+ maskoffset & (int)0xfffffffc);
	
		int off_mask = (maskoffset & 3) - (dstoffset_in_pixel & 3) +3;	
		
		if ( (x < cols) & (y < rows) )
		{
			uchar4 temp_dst = *(__global uchar4*)(dstMat+dstidx);
			uchar4 temp_mask1 = *(__global uchar4*)(maskMat+maskidx-4);
			uchar4 temp_mask = *(__global uchar4*)(maskMat+maskidx);
			uchar4 temp_mask2 = *(__global uchar4*)(maskMat+maskidx+4);		
			temp_mask1.x = (maskidx-4 >=mask_addr_start)&(maskidx-4 < mask_addr_end) ? temp_mask1.x : 0;
			temp_mask1.y = (maskidx-3 >=mask_addr_start)&(maskidx-3 < mask_addr_end) ? temp_mask1.y : 0;
			temp_mask1.z = (maskidx-2 >=mask_addr_start)&(maskidx-2 < mask_addr_end) ? temp_mask1.z : 0;
			temp_mask1.w = (maskidx-1 >=mask_addr_start)&(maskidx-1 < mask_addr_end) ? temp_mask1.w : 0;			
			temp_mask.x = (maskidx >=mask_addr_start)&(maskidx < mask_addr_end) ? temp_mask.x : 0;
			temp_mask.y = (maskidx+1 >=mask_addr_start)&(maskidx+1 < mask_addr_end) ? temp_mask.y : 0;
			temp_mask.z = (maskidx+2 >=mask_addr_start)&(maskidx+2 < mask_addr_end) ? temp_mask.z : 0;
			temp_mask.w = (maskidx+3 >=mask_addr_start)&(maskidx+3 < mask_addr_end) ? temp_mask.w : 0;	
			temp_mask2.x = (maskidx+4 >=mask_addr_start)&(maskidx+4 < mask_addr_end) ? temp_mask2.x : 0;
			temp_mask2.y = (maskidx+5 >=mask_addr_start)&(maskidx+5 < mask_addr_end) ? temp_mask2.y : 0;
			temp_mask2.z = (maskidx+6 >=mask_addr_start)&(maskidx+6 < mask_addr_end) ? temp_mask2.z : 0;
			temp_mask2.w = (maskidx+7 >=mask_addr_start)&(maskidx+7 < mask_addr_end) ? temp_mask2.w : 0;	
			uchar trans_mask[10] = {temp_mask1.y,temp_mask1.z,temp_mask1.w,temp_mask.x,temp_mask.y,temp_mask.z,temp_mask.w,temp_mask2.x,temp_mask2.y,temp_mask2.z};				
			temp_dst.x = (dstidx>=dst_addr_start)&(dstidx<dst_addr_end)& trans_mask[off_mask] ? scalar : temp_dst.x;
			temp_dst.y = (dstidx+1>=dst_addr_start)&(dstidx+1<dst_addr_end)& trans_mask[off_mask+1] ? scalar : temp_dst.y;
			temp_dst.z = (dstidx+2>=dst_addr_start)&(dstidx+2<dst_addr_end)& trans_mask[off_mask+2] ? scalar : temp_dst.z;
			temp_dst.w = (dstidx+3>=dst_addr_start)&(dstidx+3<dst_addr_end)& trans_mask[off_mask+3] ? scalar : temp_dst.w;
			*(__global uchar4*)(dstMat+dstidx) = temp_dst;
		}
}
__kernel void set_to_with_mask(
		GENTYPE scalar,
		__global GENTYPE * dstMat,
		int cols,
		int rows,
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 		
        __global const uchar * restrict maskMat,
		int maskStep,
		int maskoffset)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		int maskidx = mad24(y,maskStep,x+ maskoffset);
		uchar mask = maskMat[maskidx];		
		if ( (x < cols) & (y < rows) & mask)
		{
			dstMat[dstidx] = scalar;	
		}

}

