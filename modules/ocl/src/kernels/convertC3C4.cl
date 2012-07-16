//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
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

__kernel void convertC3C4_D0(__global const char4 * restrict src, __global char4 *dst, int cols, int rows, 
					int srcStep, int dstStep)
{
	int id = get_global_id(0);
	int y = id / cols;
	int x = id % cols;

	int d = y * srcStep + x * 3;
	char8 data = (char8)(src[d>>2], src[(d>>2) + 1]);
	char temp[8] = {data.s0, data.s1, data.s2, data.s3, data.s4, data.s5, data.s6, data.s7};
	
	int start = d & 3;
	char4 ndata = (char4)(temp[start], temp[start + 1], temp[start + 2], 0);
	if(y < rows)
		dst[y * dstStep + x] = ndata;
}

__kernel void convertC3C4_D1(__global const short* restrict src, __global short4 *dst, int cols, int rows, 
					int srcStep, int dstStep)
{
	int id = get_global_id(0);
	int y = id / cols;
	int x = id % cols;

	int d = (y * srcStep + x * 6)>>1;
	short4 data = *(__global short4 *)(src + ((d>>1)<<1));
	short temp[4] = {data.s0, data.s1, data.s2, data.s3};
	
	int start = d & 1;
	short4 ndata = (short4)(temp[start], temp[start + 1], temp[start + 2], 0);
	if(y < rows)
		dst[y * dstStep + x] = ndata;
}

__kernel void convertC3C4_D2(__global const int * restrict src, __global int4 *dst, int cols, int rows, 
					int srcStep, int dstStep)
{
	int id = get_global_id(0);
	int y = id / cols;
	int x = id % cols;

	int d = (y * srcStep + x * 12)>>2;
	int4 data = *(__global int4 *)(src + d);
	data.z = 0;
	
	if(y < rows)
		dst[y * dstStep + x] = data;
}

__kernel void convertC4C3_D2(__global const int4 * restrict src, __global int *dst, int cols, int rows, 
					int srcStep, int dstStep)
{
	int id = get_global_id(0);
	int y = id / cols;
	int x = id % cols;

	int4 data = src[y * srcStep + x];
	
	if(y < rows)
	{
		int d = y * dstStep + x * 3;
		dst[d] = data.x;
		dst[d + 1] = data.y;
		dst[d + 2] = data.z;
	}
}

__kernel void convertC4C3_D1(__global const short4 * restrict src, __global short *dst, int cols, int rows, 
					int srcStep, int dstStep)
{
	int id = get_global_id(0);
	int y = id / cols;
	int x = id % cols;

	short4 data = src[y * srcStep + x];
	
	if(y < rows)
	{
		int d = y * dstStep + x * 3;
		dst[d] = data.x;
		dst[d + 1] = data.y;
		dst[d + 2] = data.z;
	}
}

__kernel void convertC4C3_D0(__global const char4 * restrict src, __global char *dst, int cols, int rows, 
					int srcStep, int dstStep)
{
	int id = get_global_id(0);
	int y = id / cols;
	int x = id % cols;

	char4 data = src[y * srcStep + x];
	
	if(y < rows)
	{
		int d = y * dstStep + x * 3;
		dst[d] = data.x;
		dst[d + 1] = data.y;
		dst[d + 2] = data.z;
	}
}
