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
#if defined (DOUBLE_SUPPORT)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
*/

__kernel void set_to_without_mask_C1_D0(float4 scalar,__global uchar * dstMat,
        int cols,int rows,int dstStep_in_pixel,int offset_in_pixel)
{
		int x=get_global_id(0)<<2;
		int y=get_global_id(1);
		int addr_start = mad24(y,dstStep_in_pixel,offset_in_pixel);
		int addr_end = mad24(y,dstStep_in_pixel,cols+offset_in_pixel);
		int idx = mad24(y,dstStep_in_pixel,(int)(x+ offset_in_pixel & (int)0xfffffffc));
		uchar4 out;
		out.x = out.y = out.z = out.w = convert_uchar_sat(scalar.x);
		if ( (idx>=addr_start)&(idx+3 < addr_end) & (y < rows))
		{
			*(__global uchar4*)(dstMat+idx) = out;
		}
		else if(y < rows)
		{
			uchar4 temp = *(__global uchar4*)(dstMat+idx);
			temp.x = (idx>=addr_start)&(idx < addr_end)? out.x : temp.x;
			temp.y = (idx+1>=addr_start)&(idx+1 < addr_end)? out.y : temp.y;
			temp.z = (idx+2>=addr_start)&(idx+2 < addr_end)? out.z : temp.z;
			temp.w = (idx+3>=addr_start)&(idx+3 < addr_end)? out.w : temp.w;
			*(__global uchar4*)(dstMat+idx) = temp;
		}
}

__kernel void set_to_without_mask_C4_D0(float4 scalar,__global uchar4 * dstMat,
        int cols,int rows,int dstStep_in_pixel,int offset_in_pixel)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		if ( (x < cols) & (y < rows))
		{
		    int idx = mad24(y,dstStep_in_pixel,x+ offset_in_pixel);
			dstMat[idx] = convert_uchar4_sat(scalar);
		}
}
__kernel void set_to_without_mask_C1_D4(float4 scalar,__global int * dstMat,
        int cols,int rows,int dstStep_in_pixel,int offset_in_pixel)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		if ( (x < cols) & (y < rows))
		{
		    int idx = mad24(y, dstStep_in_pixel, x+offset_in_pixel);
			dstMat[idx] = convert_int_sat(scalar.x);
		}
}
__kernel void set_to_without_mask_C4_D4(float4 scalar,__global int4 * dstMat,
        int cols,int rows,int dstStep_in_pixel,int offset_in_pixel)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		if ( (x < cols) & (y < rows))
		{
		    int idx = mad24(y,dstStep_in_pixel,x+ offset_in_pixel);
			dstMat[idx] = convert_int4_sat(scalar);
		}
}

__kernel void set_to_without_mask_C1_D5(float4 scalar,__global float * dstMat,
        int cols,int rows,int dstStep_in_pixel,int offset_in_pixel)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		if ( (x < cols) & (y < rows))
		{
		    int idx = mad24(y,dstStep_in_pixel,x+ offset_in_pixel);
			dstMat[idx] = scalar.x;
		}
}
__kernel void set_to_without_mask_C4_D5(float4 scalar,__global float4 * dstMat,
        int cols,int rows,int dstStep_in_pixel,int offset_in_pixel)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		if ( (x < cols) & (y < rows))
		{
		    int idx = mad24(y,dstStep_in_pixel,x+ offset_in_pixel);
			dstMat[idx] = scalar;
		}
}

