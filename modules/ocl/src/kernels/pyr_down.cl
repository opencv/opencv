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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Dachuan Zhao, dachuan@multicorewareinc.com
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

#pragma OPENCL EXTENSION cl_amd_printf : enable


uchar round_uchar_uchar(uchar v)
{ 
	return v;
}

uchar round_uchar_int(int v)
{ 
    return (uchar)((uint)v <= 255 ? v : v > 0 ? 255 : 0); 
}

uchar round_uchar_float(float v)
{ 
	if(v - convert_int_sat_rte(v) > 1e-6 || v - convert_int_sat_rte(v) < -1e-6)
	{
		if(((int)v + 1) - (v + 0.5f) < 1e-6 && ((int)v + 1) - (v + 0.5f) > -1e-6)
		{
			v = (int)v + 0.51f;
		}
	}
    int iv = convert_int_sat_rte(v);
    return round_uchar_int(iv); 
}

uchar4 round_uchar4_uchar4(uchar4 v)
{ 
	return v;
}

uchar4 round_uchar4_int4(int4 v)
{ 
	uchar4 result;
	result.x = (uchar)(v.x <= 255 ? v.x : v.x > 0 ? 255 : 0); 
	result.y = (uchar)(v.y <= 255 ? v.y : v.y > 0 ? 255 : 0); 
	result.z = (uchar)(v.z <= 255 ? v.z : v.z > 0 ? 255 : 0); 
	result.w = (uchar)(v.w <= 255 ? v.w : v.w > 0 ? 255 : 0); 
    return result; 
}

uchar4 round_uchar4_float4(float4 v)
{ 
	if(v.x - convert_int_sat_rte(v.x) > 1e-6 || v.x - convert_int_sat_rte(v.x) < -1e-6)
	{
		if(((int)(v.x) + 1) - (v.x + 0.5f) < 1e-6 && ((int)(v.x) + 1) - (v.x + 0.5f) > -1e-6)
		{
			v.x = (int)(v.x) + 0.51f;
		}
	}
	if(v.y - convert_int_sat_rte(v.y) > 1e-6 || v.y - convert_int_sat_rte(v.y) < -1e-6)
	{
		if(((int)(v.y) + 1) - (v.y + 0.5f) < 1e-6 && ((int)(v.y) + 1) - (v.y + 0.5f) > -1e-6)
		{
			v.y = (int)(v.y) + 0.51f;
		}
	}
	if(v.z - convert_int_sat_rte(v.z) > 1e-6 || v.z - convert_int_sat_rte(v.z) < -1e-6)
	{
		if(((int)(v.z) + 1) - (v.z + 0.5f) < 1e-6 && ((int)(v.z) + 1) - (v.z + 0.5f) > -1e-6)
		{
			v.z = (int)(v.z) + 0.51f;
		}
	}
	if(v.w - convert_int_sat_rte(v.w) > 1e-6 || v.w - convert_int_sat_rte(v.w) < -1e-6)
	{
		if(((int)(v.w) + 1) - (v.w + 0.5f) < 1e-6 && ((int)(v.w) + 1) - (v.w + 0.5f) > -1e-6)
		{
			v.w = (int)(v.w) + 0.51f;
		}
	}
    int4 iv = convert_int4_sat_rte(v);
    return round_uchar4_int4(iv); 
}




int idx_row_low(int y, int last_row)
{
	if(y < 0)
	{
		y = -y;
	}
    return y % (last_row + 1);
}

int idx_row_high(int y, int last_row) 
{
	int i;
	int j;
	if(last_row - y < 0)
	{
		i = (y - last_row);
	}
	else
	{
		i = (last_row - y);
	}
	if(last_row - i < 0)
	{
		j = i - last_row;
	}
	else
	{
		j = last_row - i;
	}
    return j % (last_row + 1);
}

int idx_row(int y, int last_row)
{
    return idx_row_low(idx_row_high(y, last_row), last_row);
}

int idx_col_low(int x, int last_col)
{
	if(x < 0)
	{
		x = -x;
	}
    return x % (last_col + 1);
}

int idx_col_high(int x, int last_col) 
{
	int i;
	int j;
	if(last_col - x < 0)
	{
		i = (x - last_col);
	}
	else
	{
		i = (last_col - x);
	}
	if(last_col - i < 0)
	{
		j = i - last_col;
	}
	else
	{
		j = last_col - i;
	}
    return j % (last_col + 1);
}

int idx_col(int x, int last_col)
{
    return idx_col_low(idx_col_high(x, last_col), last_col);
}

__kernel void pyrDown_C1_D0(__global uchar * srcData, int srcStep, int srcOffset, int srcRows, int srcCols, __global uchar *dst, int dstStep, int dstOffset, int dstCols)
{
    const int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int y = get_group_id(1);

    __local float smem[256 + 4];

    float sum;

    const int src_y = 2*y;
    const int last_row = srcRows - 1;
    const int last_col = srcCols - 1;

    sum = 0;

    sum = sum + 0.0625f * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y - 2, last_row) * srcStep))[idx_col(x, last_col)]);
    sum = sum + 0.25f   * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y - 1, last_row) * srcStep))[idx_col(x, last_col)]);
    sum = sum + 0.375f  * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y    , last_row) * srcStep))[idx_col(x, last_col)]);
    sum = sum + 0.25f   * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y + 1, last_row) * srcStep))[idx_col(x, last_col)]);
    sum = sum + 0.0625f * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y + 2, last_row) * srcStep))[idx_col(x, last_col)]);

    smem[2 + get_local_id(0)] = sum;

    if (get_local_id(0) < 2)
    {
        const int left_x = x - 2;

        sum = 0;

        sum = sum + 0.0625f * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y - 2, last_row) * srcStep))[idx_col(left_x, last_col)]);
		sum = sum + 0.25f   * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y - 1, last_row) * srcStep))[idx_col(left_x, last_col)]);
		sum = sum + 0.375f  * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y    , last_row) * srcStep))[idx_col(left_x, last_col)]);
		sum = sum + 0.25f   * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y + 1, last_row) * srcStep))[idx_col(left_x, last_col)]);
		sum = sum + 0.0625f * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y + 2, last_row) * srcStep))[idx_col(left_x, last_col)]);

        smem[get_local_id(0)] = sum;
    }

    if (get_local_id(0) > 253)
    {
        const int right_x = x + 2;

        sum = 0;

        sum = sum + 0.0625f * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y - 2, last_row) * srcStep))[idx_col(right_x, last_col)]);
		sum = sum + 0.25f   * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y - 1, last_row) * srcStep))[idx_col(right_x, last_col)]);
		sum = sum + 0.375f  * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y    , last_row) * srcStep))[idx_col(right_x, last_col)]);
		sum = sum + 0.25f   * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y + 1, last_row) * srcStep))[idx_col(right_x, last_col)]);
		sum = sum + 0.0625f * round_uchar_uchar(((__global uchar*)((__global char*)srcData + idx_row(src_y + 2, last_row) * srcStep))[idx_col(right_x, last_col)]);

        smem[4 + get_local_id(0)] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < 128)
    {
        const int tid2 = get_local_id(0) * 2;

        sum = 0;

        sum = sum + 0.0625f * smem[2 + tid2 - 2];
        sum = sum + 0.25f   * smem[2 + tid2 - 1];
        sum = sum + 0.375f  * smem[2 + tid2    ];
        sum = sum + 0.25f   * smem[2 + tid2 + 1];
        sum = sum + 0.0625f * smem[2 + tid2 + 2];

        const int dst_x = (get_group_id(0) * get_local_size(0) + tid2) / 2;

        if (dst_x < dstCols)
            dst[y * dstStep + dst_x] = round_uchar_float(sum);
    }
}

__kernel void pyrDown_C4_D0(__global uchar4 * srcData, int srcStep, int srcOffset, int srcRows, int srcCols, __global uchar4 *dst, int dstStep, int dstOffset, int dstCols)
{
    const int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int y = get_group_id(1);

    __local float4 smem[256 + 4];

    float4 sum;

    const int src_y = 2*y;
    const int last_row = srcRows - 1;
    const int last_col = srcCols - 1;

	float4 co1 = (float4)(0.375f, 0.375f, 0.375f, 0.375f);
	float4 co2 = (float4)(0.25f, 0.25f, 0.25f, 0.25f);
	float4 co3 = (float4)(0.0625f, 0.0625f, 0.0625f, 0.0625f);

    sum = 0;

	sum = sum + co3 * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y - 2, last_row) * srcStep / 4))[idx_col(x, last_col)]));
	sum = sum + co2   * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y - 1, last_row) * srcStep / 4))[idx_col(x, last_col)]));
	sum = sum + co1  * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y    , last_row) * srcStep / 4))[idx_col(x, last_col)]));
	sum = sum + co2   * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y + 1, last_row) * srcStep / 4))[idx_col(x, last_col)]));
	sum = sum + co3 * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y + 2, last_row) * srcStep / 4))[idx_col(x, last_col)]));

	smem[2 + get_local_id(0)] = sum;

	if (get_local_id(0) < 2)
	{
		const int left_x = x - 2;

		sum = 0;

		sum = sum + co3 * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y - 2, last_row) * srcStep / 4))[idx_col(left_x, last_col)]));
		sum = sum + co2   * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y - 1, last_row) * srcStep / 4))[idx_col(left_x, last_col)]));
		sum = sum + co1  * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y    , last_row) * srcStep / 4))[idx_col(left_x, last_col)]));
		sum = sum + co2   * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y + 1, last_row) * srcStep / 4))[idx_col(left_x, last_col)]));
		sum = sum + co3 * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y + 2, last_row) * srcStep / 4))[idx_col(left_x, last_col)]));

		smem[get_local_id(0)] = sum;
	}

	if (get_local_id(0) > 253)
	{
		const int right_x = x + 2;

		sum = 0;

		sum = sum + co3 * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y - 2, last_row) * srcStep / 4))[idx_col(right_x, last_col)]));
		sum = sum + co2   * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y - 1, last_row) * srcStep / 4))[idx_col(right_x, last_col)]));
		sum = sum + co1  * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y    , last_row) * srcStep / 4))[idx_col(right_x, last_col)]));
		sum = sum + co2   * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y + 1, last_row) * srcStep / 4))[idx_col(right_x, last_col)]));
		sum = sum + co3 * convert_float4(round_uchar4_uchar4(((__global uchar4*)((__global char4*)srcData + idx_row(src_y + 2, last_row) * srcStep / 4))[idx_col(right_x, last_col)]));

		smem[4 + get_local_id(0)] = sum;
	}

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < 128)
    {
        const int tid2 = get_local_id(0) * 2;

        sum = 0;

        sum = sum + co3 * smem[2 + tid2 - 2];
        sum = sum + co2   * smem[2 + tid2 - 1];
        sum = sum + co1  * smem[2 + tid2    ];
        sum = sum + co2   * smem[2 + tid2 + 1];
        sum = sum + co3 * smem[2 + tid2 + 2];

        const int dst_x = (get_group_id(0) * get_local_size(0) + tid2) / 2;

        if (dst_x < dstCols)
            dst[y * dstStep / 4 + dst_x] = round_uchar4_float4(sum);
    }
}

__kernel void pyrDown_C1_D5(__global float * srcData, int srcStep, int srcOffset, int srcRows, int srcCols, __global float *dst, int dstStep, int dstOffset, int dstCols)
{
    const int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int y = get_group_id(1);

    __local float smem[256 + 4];

    float sum;

    const int src_y = 2*y;
    const int last_row = srcRows - 1;
    const int last_col = srcCols - 1;

    sum = 0;

    sum = sum + 0.0625f * ((__global float*)((__global char*)srcData + idx_row(src_y - 2, last_row) * srcStep))[idx_col(x, last_col)];
    sum = sum + 0.25f   * ((__global float*)((__global char*)srcData + idx_row(src_y - 1, last_row) * srcStep))[idx_col(x, last_col)];
    sum = sum + 0.375f  * ((__global float*)((__global char*)srcData + idx_row(src_y    , last_row) * srcStep))[idx_col(x, last_col)];
    sum = sum + 0.25f   * ((__global float*)((__global char*)srcData + idx_row(src_y + 1, last_row) * srcStep))[idx_col(x, last_col)];
    sum = sum + 0.0625f * ((__global float*)((__global char*)srcData + idx_row(src_y + 2, last_row) * srcStep))[idx_col(x, last_col)];

    smem[2 + get_local_id(0)] = sum;

    if (get_local_id(0) < 2)
    {
        const int left_x = x - 2;

        sum = 0;

        sum = sum + 0.0625f * ((__global float*)((__global char*)srcData + idx_row(src_y - 2, last_row) * srcStep))[idx_col(left_x, last_col)];
		sum = sum + 0.25f   * ((__global float*)((__global char*)srcData + idx_row(src_y - 1, last_row) * srcStep))[idx_col(left_x, last_col)];
		sum = sum + 0.375f  * ((__global float*)((__global char*)srcData + idx_row(src_y    , last_row) * srcStep))[idx_col(left_x, last_col)];
		sum = sum + 0.25f   * ((__global float*)((__global char*)srcData + idx_row(src_y + 1, last_row) * srcStep))[idx_col(left_x, last_col)];
		sum = sum + 0.0625f * ((__global float*)((__global char*)srcData + idx_row(src_y + 2, last_row) * srcStep))[idx_col(left_x, last_col)];

        smem[get_local_id(0)] = sum;
    }

    if (get_local_id(0) > 253)
    {
        const int right_x = x + 2;

        sum = 0;

        sum = sum + 0.0625f * ((__global float*)((__global char*)srcData + idx_row(src_y - 2, last_row) * srcStep))[idx_col(right_x, last_col)];
		sum = sum + 0.25f   * ((__global float*)((__global char*)srcData + idx_row(src_y - 1, last_row) * srcStep))[idx_col(right_x, last_col)];
		sum = sum + 0.375f  * ((__global float*)((__global char*)srcData + idx_row(src_y    , last_row) * srcStep))[idx_col(right_x, last_col)];
		sum = sum + 0.25f   * ((__global float*)((__global char*)srcData + idx_row(src_y + 1, last_row) * srcStep))[idx_col(right_x, last_col)];
		sum = sum + 0.0625f * ((__global float*)((__global char*)srcData + idx_row(src_y + 2, last_row) * srcStep))[idx_col(right_x, last_col)];

        smem[4 + get_local_id(0)] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < 128)
    {
        const int tid2 = get_local_id(0) * 2;

        sum = 0;

        sum = sum + 0.0625f * smem[2 + tid2 - 2];
        sum = sum + 0.25f   * smem[2 + tid2 - 1];
        sum = sum + 0.375f  * smem[2 + tid2    ];
        sum = sum + 0.25f   * smem[2 + tid2 + 1];
        sum = sum + 0.0625f * smem[2 + tid2 + 2];

        const int dst_x = (get_group_id(0) * get_local_size(0) + tid2) / 2;

        if (dst_x < dstCols)
            dst[y * dstStep / 4 + dst_x] = sum;
    }
}

__kernel void pyrDown_C4_D5(__global float4 * srcData, int srcStep, int srcOffset, int srcRows, int srcCols, __global float4 *dst, int dstStep, int dstOffset, int dstCols)
{
    const int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int y = get_group_id(1);

    __local float4 smem[256 + 4];

    float4 sum;

    const int src_y = 2*y;
    const int last_row = srcRows - 1;
    const int last_col = srcCols - 1;

	float4 co1 = (float4)(0.375f, 0.375f, 0.375f, 0.375f);
	float4 co2 = (float4)(0.25f, 0.25f, 0.25f, 0.25f);
	float4 co3 = (float4)(0.0625f, 0.0625f, 0.0625f, 0.0625f);

    sum = 0;

	sum = sum + co3 * ((__global float4*)((__global char4*)srcData + idx_row(src_y - 2, last_row) * srcStep / 4))[idx_col(x, last_col)];
	sum = sum + co2   * ((__global float4*)((__global char4*)srcData + idx_row(src_y - 1, last_row) * srcStep / 4))[idx_col(x, last_col)];
	sum = sum + co1  * ((__global float4*)((__global char4*)srcData + idx_row(src_y    , last_row) * srcStep / 4))[idx_col(x, last_col)];
	sum = sum + co2   * ((__global float4*)((__global char4*)srcData + idx_row(src_y + 1, last_row) * srcStep / 4))[idx_col(x, last_col)];
	sum = sum + co3 * ((__global float4*)((__global char4*)srcData + idx_row(src_y + 2, last_row) * srcStep / 4))[idx_col(x, last_col)];

	smem[2 + get_local_id(0)] = sum;

	if (get_local_id(0) < 2)
	{
		const int left_x = x - 2;

		sum = 0;

		sum = sum + co3 * ((__global float4*)((__global char4*)srcData + idx_row(src_y - 2, last_row) * srcStep / 4))[idx_col(left_x, last_col)];
		sum = sum + co2   * ((__global float4*)((__global char4*)srcData + idx_row(src_y - 1, last_row) * srcStep / 4))[idx_col(left_x, last_col)];
		sum = sum + co1  * ((__global float4*)((__global char4*)srcData + idx_row(src_y    , last_row) * srcStep / 4))[idx_col(left_x, last_col)];
		sum = sum + co2   * ((__global float4*)((__global char4*)srcData + idx_row(src_y + 1, last_row) * srcStep / 4))[idx_col(left_x, last_col)];
		sum = sum + co3 * ((__global float4*)((__global char4*)srcData + idx_row(src_y + 2, last_row) * srcStep / 4))[idx_col(left_x, last_col)];

		smem[get_local_id(0)] = sum;
	}

	if (get_local_id(0) > 253)
	{
		const int right_x = x + 2;

		sum = 0;

		sum = sum + co3 * ((__global float4*)((__global char4*)srcData + idx_row(src_y - 2, last_row) * srcStep / 4))[idx_col(right_x, last_col)];
		sum = sum + co2   * ((__global float4*)((__global char4*)srcData + idx_row(src_y - 1, last_row) * srcStep / 4))[idx_col(right_x, last_col)];
		sum = sum + co1  * ((__global float4*)((__global char4*)srcData + idx_row(src_y    , last_row) * srcStep / 4))[idx_col(right_x, last_col)];
		sum = sum + co2   * ((__global float4*)((__global char4*)srcData + idx_row(src_y + 1, last_row) * srcStep / 4))[idx_col(right_x, last_col)];
		sum = sum + co3 * ((__global float4*)((__global char4*)srcData + idx_row(src_y + 2, last_row) * srcStep / 4))[idx_col(right_x, last_col)];

		smem[4 + get_local_id(0)] = sum;
	}

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < 128)
    {
        const int tid2 = get_local_id(0) * 2;

        sum = 0;

        sum = sum + co3 * smem[2 + tid2 - 2];
        sum = sum + co2   * smem[2 + tid2 - 1];
        sum = sum + co1  * smem[2 + tid2    ];
        sum = sum + co2   * smem[2 + tid2 + 1];
        sum = sum + co3 * smem[2 + tid2 + 2];

        const int dst_x = (get_group_id(0) * get_local_size(0) + tid2) / 2;

        if (dst_x < dstCols)
            dst[y * dstStep / 16 + dst_x] = sum;
    }
}
