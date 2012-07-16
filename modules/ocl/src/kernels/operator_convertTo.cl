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
#define F float

__kernel void convert_to_S4_C1_D0(
		__global const int* restrict srcMat,
		__global uchar* dstMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		F alpha,
		F beta)
{
		int x=get_global_id(0)<<2;
		int y=get_global_id(1);
		//int src_addr_start = mad24(y,srcStep_in_pixel,srcoffset_in_pixel);
		//int src_addr_end = mad24(y,srcStep_in_pixel,cols+srcoffset_in_pixel);
		int off_src = (dstoffset_in_pixel & 3);	
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel - off_src);		
		int dst_addr_start = mad24(y,dstStep_in_pixel,dstoffset_in_pixel);
		int dst_addr_end = mad24(y,dstStep_in_pixel,cols+dstoffset_in_pixel);
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel & (int)0xfffffffc);

		if ( (x < cols + off_src) & (y < rows) )
		{
			float4 temp_src = convert_float4(vload4(0,srcMat+srcidx));				
			uchar4 temp_dst = *(__global uchar4*)(dstMat+dstidx);		
			//int trans_src[10] = {temp_src1.y,temp_src1.z,temp_src1.w,temp_src.x,temp_src.y,temp_src.z,temp_src.w,temp_src2.x,temp_src2.y,temp_src2.z};		
			temp_dst.x = (dstidx>=dst_addr_start)&(dstidx<dst_addr_end) ? convert_uchar_sat(temp_src.x*alpha+beta) : temp_dst.x;
			temp_dst.y = (dstidx+1>=dst_addr_start)&(dstidx+1<dst_addr_end) ? convert_uchar_sat(temp_src.y*alpha+beta) : temp_dst.y;
			temp_dst.z = (dstidx+2>=dst_addr_start)&(dstidx+2<dst_addr_end) ? convert_uchar_sat(temp_src.z*alpha+beta) : temp_dst.z;
			temp_dst.w = (dstidx+3>=dst_addr_start)&(dstidx+3<dst_addr_end) ? convert_uchar_sat(temp_src.w*alpha+beta) : temp_dst.w;
			*(__global uchar4*)(dstMat+dstidx) = temp_dst;
		}
}

__kernel void convert_to_S4_C4_D0(
		__global const int4* restrict srcMat,
		__global uchar4* dstMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		F alpha,
		F beta)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		if ( (x < cols) & (y < rows) )
		{	
			float4 temp_src = convert_float4(srcMat[srcidx]);
			dstMat[dstidx] = convert_uchar4_sat(temp_src*alpha+beta);
		}
}

__kernel void convert_to_S5_C1_D0(
		__global const float* restrict srcMat,
		__global uchar* dstMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		F alpha,
		F beta)
{
		int x=get_global_id(0)<<2;
		int y=get_global_id(1);
		//int src_addr_start = mad24(y,srcStep_in_pixel,srcoffset_in_pixel);
		//int src_addr_end = mad24(y,srcStep_in_pixel,cols+srcoffset_in_pixel);
		int off_src = (dstoffset_in_pixel & 3);	
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel - off_src);		
		int dst_addr_start = mad24(y,dstStep_in_pixel,dstoffset_in_pixel);
		int dst_addr_end = mad24(y,dstStep_in_pixel,cols+dstoffset_in_pixel);
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel & (int)0xfffffffc);

		if ( (x < cols + off_src) & (y < rows) )
		{
			float4 temp_src = vload4(0,srcMat+srcidx);				
			uchar4 temp_dst = *(__global uchar4*)(dstMat+dstidx);		
			//int trans_src[10] = {temp_src1.y,temp_src1.z,temp_src1.w,temp_src.x,temp_src.y,temp_src.z,temp_src.w,temp_src2.x,temp_src2.y,temp_src2.z};		
			temp_dst.x = (dstidx>=dst_addr_start)&(dstidx<dst_addr_end) ? convert_uchar_sat(temp_src.x*alpha+beta) : temp_dst.x;
			temp_dst.y = (dstidx+1>=dst_addr_start)&(dstidx+1<dst_addr_end) ? convert_uchar_sat(temp_src.y*alpha+beta) : temp_dst.y;
			temp_dst.z = (dstidx+2>=dst_addr_start)&(dstidx+2<dst_addr_end) ? convert_uchar_sat(temp_src.z*alpha+beta) : temp_dst.z;
			temp_dst.w = (dstidx+3>=dst_addr_start)&(dstidx+3<dst_addr_end) ? convert_uchar_sat(temp_src.w*alpha+beta) : temp_dst.w;
			*(__global uchar4*)(dstMat+dstidx) = temp_dst;
		}
}
__kernel void convert_to_S5_C4_D0(
		__global const float4* restrict srcMat,
		__global uchar4* dstMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		F alpha,
		F beta)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		if ( (x < cols) & (y < rows) )
		{	
			float4 temp_src = srcMat[srcidx];
			dstMat[dstidx] = convert_uchar4_sat(temp_src*alpha+beta);
		}
}

__kernel void convert_to_S0_C1_D4(
		__global const uchar* restrict srcMat,
		__global int* dstMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		F alpha,
		F beta)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		if ( (x < cols) & (y < rows) )
		{	
			float temp_src = convert_float(srcMat[srcidx]);
			dstMat[dstidx] = convert_int_sat(temp_src*alpha+beta);
		}
}

__kernel void convert_to_S5_C1_D4(
		__global const float* restrict srcMat,
		__global int* dstMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		F alpha,
		F beta)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		if ( (x < cols) & (y < rows) )
		{	
			float temp_src = srcMat[srcidx];
			dstMat[dstidx] = convert_int_sat(temp_src*alpha+beta);
		}
}

__kernel void convert_to_S0_C4_D4(
		__global const uchar4* restrict srcMat,
		__global int4* dstMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		F alpha,
		F beta)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		if ( (x < cols) & (y < rows) )
		{	
			float4 temp_src = convert_float4(srcMat[srcidx]);
			dstMat[dstidx] = convert_int4_sat(temp_src*alpha+beta);
		}
}

__kernel void convert_to_S5_C4_D4(
		__global const float4* restrict srcMat,
		__global int4* dstMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		F alpha,
		F beta)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		if ( (x < cols) & (y < rows) )
		{	
			float4 temp_src = srcMat[srcidx];
			dstMat[dstidx] = convert_int4_sat(temp_src*alpha+beta);
		}
}

__kernel void convert_to_S0_C1_D5(
		__global const uchar* restrict srcMat,
		__global float* dstMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		F alpha,
		F beta)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		if ( (x < cols) & (y < rows) )
		{	
			float temp_src = convert_float(srcMat[srcidx]);
			dstMat[dstidx] = temp_src*alpha+beta;
		}
}

__kernel void convert_to_S4_C1_D5(
		__global const int* restrict srcMat,
		__global float* dstMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		F alpha,
		F beta)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		if ( (x < cols) & (y < rows) )
		{	
			float temp_src = convert_float(srcMat[srcidx]);
			dstMat[dstidx] = temp_src*alpha+beta;
		}
}

__kernel void convert_to_S0_C4_D5(
		__global const uchar4* restrict srcMat,
		__global float4* dstMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		F alpha,
		F beta)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		if ( (x < cols) & (y < rows) )
		{	
			float4 temp_src = convert_float4(srcMat[srcidx]);
			dstMat[dstidx] = temp_src*alpha+beta;
		}
}

__kernel void convert_to_S4_C4_D5(
		__global const int4* restrict srcMat,
		__global float4* dstMat,
		int cols,
		int rows,
		int srcStep_in_pixel,
		int srcoffset_in_pixel, 		
		int dstStep_in_pixel,
		int dstoffset_in_pixel, 
		F alpha,
		F beta)
{
		int x=get_global_id(0);
		int y=get_global_id(1);
		int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);		
		int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
		if ( (x < cols) & (y < rows) )
		{	
			float4 temp_src = convert_float4(srcMat[srcidx]);
			dstMat[dstidx] = temp_src*alpha+beta;
		}
}
