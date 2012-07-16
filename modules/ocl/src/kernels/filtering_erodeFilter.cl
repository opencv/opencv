//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Zero Lin, zero.lin@amd.com
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

__kernel void erode_C4_D5(__global const float4 * restrict src, __global float4 *dst, int srcOffset, int dstOffset, 
					int mincols, int maxcols, int minrows, int maxrows, int cols, int rows, 
					int srcStep, int dstStep, __constant uchar * mat_kernel, int src_whole_cols, int src_whole_rows)
{
    int mX = get_global_id(0);
    int mY = get_global_id(1);
    int kX = mX - anX, kY = mY - anY;
	int end_addr = mad24(src_whole_rows-1,srcStep,src_whole_cols);
    float4 minVal = (float4)(3.4e+38);
	int k=0;
	for(int i=0;i<ksY;i++, kY++ , kX = mX - anX)
    {
        for(int j=0;j<ksX; j++, kX++)
        {
			int current_addr = mad24(kY,srcStep,kX) + srcOffset;
			current_addr = ((current_addr < end_addr) && (current_addr > 0)) ? current_addr : 0;
			float4 v = src[current_addr];
			uchar now = mat_kernel[k++];
			float4 flag = (kX >= mincols & kX <= maxcols & kY >= minrows & kY <= maxrows & now != 0) ? v : (float4)(3.4e+38);
            minVal = min(minVal , flag);
        }
    }

	if(mX < cols && mY < rows)
        dst[mY * dstStep + mX + dstOffset] = (minVal);		   
}

__kernel void erode_C1_D5(__global float4 * src, __global float *dst, int srcOffset, int dstOffset, 
					int mincols, int maxcols, int minrows, int maxrows, int cols, int rows, 
					int srcStep, int dstStep, __constant uchar * mat_kernel, int src_whole_cols, int src_whole_rows)
{
    int mX = (get_global_id(0)<<2) - (dstOffset&3);
    int mY = get_global_id(1);
    int kX = mX - anX, kY = mY - anY;
	int end_addr = mad24(src_whole_rows-1,srcStep,src_whole_cols);
    float4 minVal = (float4)(3.4e+38);
	int k=0;
	for(int i=0;i<ksY;i++, kY++ , kX = mX - anX)
    {
        for(int j=0;j<ksX;j++, kX++)
        {
			int start = mad24(kY,srcStep,kX) + srcOffset;
			start = ((start < end_addr) && (start > 0)) ? start : 0;
			int start2 = ((start + 4 < end_addr) && (start > 0)) ? start + 4 : 0;
			float8 sVal = (float8)(src[start>>2], src[start2>>2]);
			
			float sAry[8]= {sVal.s0, sVal.s1, sVal.s2, sVal.s3, sVal.s4, sVal.s5, sVal.s6, sVal.s7};
			int det = start & 3;
			float4 v=(float4)(sAry[det], sAry[det+1], sAry[det+2], sAry[det+3]);		
			uchar now = mat_kernel[k++];
			float4 flag = (kY >= minrows & kY <= maxrows & now != 0) ? v : (float4)(3.4e+38);
			flag.x = (kX >= mincols & kX <= maxcols) ? flag.x : 3.4e+38;
			flag.y = (kX+1 >= mincols & kX+1 <= maxcols) ? flag.y : 3.4e+38;
			flag.z = (kX+2 >= mincols & kX+2 <= maxcols) ? flag.z : 3.4e+38;
			flag.w = (kX+3 >= mincols & kX+3 <= maxcols) ? flag.w : 3.4e+38;
			
            minVal = min(minVal , flag);
        }
    }

	if(mY < rows && mX < cols)
	{
		__global float4* d = (__global float4*)(dst + mY * dstStep + mX + dstOffset);
		float4 dVal = *d;
		minVal.x = (mX >=0 & mX < cols) ? minVal.x : dVal.x;
		minVal.y = (mX+1 >=0 & mX+1 < cols) ? minVal.y : dVal.y;
		minVal.z = (mX+2 >=0 & mX+2 < cols) ? minVal.z : dVal.z;
		minVal.w = (mX+3 >=0 & mX+3 < cols) ? minVal.w : dVal.w;
		
        *d = (minVal);	
	}
}

__kernel void erode_C1_D0(__global const uchar4 * restrict src, __global uchar *dst, int srcOffset, int dstOffset, 
					int mincols, int maxcols, int minrows, int maxrows, int cols, int rows, 
					int srcStep, int dstStep, __constant uchar * mat_kernel, int src_whole_cols, int src_whole_rows)
{
    int mX = (get_global_id(0)<<2) - (dstOffset&3);
    int mY = get_global_id(1);
    int kX = mX - anX, kY = mY - anY;
	int end_addr = mad24(src_whole_rows-1,srcStep,src_whole_cols);
    uchar4 minVal = (uchar4)(0xff);
	int k=0;
	for(int i=0;i<ksY;i++, kY++ , kX = mX - anX)
    {
        for(int j=0;j<ksX;j++, kX++)
        {
			int start = mad24(kY,srcStep,kX) + srcOffset;
			start = ((start < end_addr) && (start > 0)) ? start : 0;
			int start2 = ((start + 4 < end_addr) && (start > 0)) ? start + 4 : 0;
			uchar8 sVal = (uchar8)(src[start>>2], src[start2>>2]);
			
			uchar sAry[8]= {sVal.s0, sVal.s1, sVal.s2, sVal.s3, sVal.s4, sVal.s5, sVal.s6, sVal.s7};
			int det = start & 3;
			uchar4 v=(uchar4)(sAry[det], sAry[det+1], sAry[det+2], sAry[det+3]);

			uchar4 flag = (kY >= minrows & kY <= maxrows & mat_kernel[k++] != 0) ? v : (uchar4)(0xff);
			flag.x = (kX >= mincols & kX <= maxcols) ? flag.x : 0xff;
			flag.y = (kX+1 >= mincols & kX+1 <= maxcols) ? flag.y : 0xff;
			flag.z = (kX+2 >= mincols & kX+2 <= maxcols) ? flag.z : 0xff;
			flag.w = (kX+3 >= mincols & kX+3 <= maxcols) ? flag.w : 0xff;			

            minVal = min(minVal , flag);
        }
    }

	if(mY < rows)
	{
		__global uchar4* d = (__global uchar4*)(dst + mY * dstStep + mX + dstOffset);
		uchar4 dVal = *d;
		
		minVal.x = (mX >=0 & mX < cols) ? minVal.x : dVal.x;
		minVal.y = (mX+1 >=0 & mX+1 < cols) ? minVal.y : dVal.y;
		minVal.z = (mX+2 >=0 & mX+2 < cols) ? minVal.z : dVal.z;
		minVal.w = (mX+3 >=0 & mX+3 < cols) ? minVal.w : dVal.w;
		
        *d = (minVal);	
	}
}

__kernel void erode_C4_D0(__global const uchar4 * restrict src, __global uchar4 *dst, int srcOffset, int dstOffset, 
					int mincols, int maxcols, int minrows, int maxrows, int cols, int rows, 
					int srcStep, int dstStep, __constant uchar * mat_kernel, int src_whole_cols, int src_whole_rows)
{
    int mX = get_global_id(0);
    int mY = get_global_id(1);
    int kX = mX - anX, kY = mY - anY;
	int end_addr = mad24(src_whole_rows-1,srcStep,src_whole_cols);
    uchar4 minVal = (uchar4)(0xff);
	int k=0;
	for(int i=0;i<ksY;i++, kY++ , kX = mX - anX)
    {
        for(int j=0;j<ksX;j++, kX++)
        {
			int current_addr = mad24(kY,srcStep,kX) + srcOffset;
			current_addr = ((current_addr < end_addr) && (current_addr > 0)) ? current_addr : 0;		
			uchar4 v = src[current_addr];
			uchar now = mat_kernel[k++];
			uchar4 flag = (kX >= mincols & kX <= maxcols & kY >= minrows & kY <= maxrows & now != 0) ? v : (uchar4)(0xff);
            minVal = min(minVal , flag);
        }
    }

	if(mX < cols && mY < rows)
        dst[mY * dstStep + mX + dstOffset] = (minVal);		   
}

