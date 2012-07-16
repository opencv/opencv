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
#define PARTITAL_HISTGRAM256_COUNT     (256) 
#define HISTOGRAM256_BIN_COUNT         (256)

#define HISTGRAM256_WORK_GROUP_SIZE     (256)
#define HISTGRAM256_LOCAL_MEM_SIZE      (HISTOGRAM256_BIN_COUNT)

__kernel __attribute__((reqd_work_group_size(256,1,1)))void calc_sub_hist_D0(__global const uchar4* src, 
							   int src_step, 
							   int src_offset,
                               __global int*   buf,
                               int data_count, 
							   int cols, 
							   int inc_x, 
							   int inc_y,
							   int dst_offset)
{
    int x  = get_global_id(0);
    int lx = get_local_id(0);
    int gx = get_group_id(0);
    int total_threads = get_global_size(0);
    src +=  src_offset;
    __local int s_hist[HISTGRAM256_LOCAL_MEM_SIZE];
    s_hist[lx] = 0;

    int pos_y = x / cols;
    int pos_x = x - mul24(pos_y, cols);
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int pos = x; pos < data_count; pos += total_threads)
    {
        int4 data = convert_int4(src[mad24(pos_y,src_step,pos_x)]);
        atomic_inc(s_hist + data.x);
        atomic_inc(s_hist + data.y);
        atomic_inc(s_hist + data.z);
        atomic_inc(s_hist + data.w);
		
        pos_x +=inc_x;
        int off = (pos_x >= cols ? -1 : 0);
        pos_x =  mad24(off,cols,pos_x);
        pos_y += inc_y - off;
		
        //pos_x = pos_x > cols ? pos_x - cols : pos_x;
        //pos_y = pos_x > cols ? pos_y + 1 : pos_y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    buf[ mad24(gx, dst_offset, lx)] = s_hist[lx];
}

__kernel void __attribute__((reqd_work_group_size(1,256,1)))calc_sub_hist2_D0( __global const uchar* src, 
				 int src_step, 
				 int src_offset,
                 __global int*   buf,
                 int left_col, 
				 int cols,
				 int rows,
				 int dst_offset)
{
	int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int gx = get_group_id(0);
	int gy = get_group_id(1);
	int gnum = get_num_groups(0);
	int output_row = mad24(gy,gnum,gx);
	//int lidx = get_local_id(0);
	int lidy = get_local_id(1);

    __local int s_hist[HISTGRAM256_LOCAL_MEM_SIZE+1];
    s_hist[lidy] = 0;
	//mem_fence(CLK_LOCAL_MEM_FENCE);


	//clamp(gidx,mask,cols-1);
	gidx = gidx >= left_col ? cols+gidx : gidx;
	//gidy = gidy >= rows?rows-1:gidy;

	int src_index = src_offset + mad24(gidy,src_step,gidx);	
	//int dst_index = dst_offset + mad24(gidy,dst_step,gidx);
	//uchar4 p,q;
	barrier(CLK_LOCAL_MEM_FENCE);
	int p = (int)src[src_index];
	p = gidy >= rows ? HISTGRAM256_LOCAL_MEM_SIZE : p;
	atomic_inc(s_hist + p);
    barrier(CLK_LOCAL_MEM_FENCE);
    buf[ mad24(output_row, dst_offset, lidy)] += s_hist[lidy];
}
__kernel __attribute__((reqd_work_group_size(256,1,1)))void merge_hist(__global int* buf,  
				__global int* hist,
				int src_step)
{
    int lx = get_local_id(0);
    int gx = get_group_id(0);

    int sum = 0;

    for(int i = lx; i < PARTITAL_HISTGRAM256_COUNT; i += HISTGRAM256_WORK_GROUP_SIZE)
        sum += buf[ mad24(i, src_step, gx)];

    __local int data[HISTGRAM256_WORK_GROUP_SIZE];
    data[lx] = sum;

    for(int stride = HISTGRAM256_WORK_GROUP_SIZE /2; stride > 0; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lx < stride)
            data[lx] += data[lx + stride];
    }

    if(lx == 0)
        hist[gx] = data[0]; 
}

__kernel __attribute__((reqd_work_group_size(256,1,1)))void calLUT(
							__global uchar * dst,
							__constant int * hist,
							float scale)
{
	int lid = get_local_id(0);
	__local int sumhist[HISTOGRAM256_BIN_COUNT];
	//__local uchar lut[HISTOGRAM256_BIN_COUNT+1];

	sumhist[lid]=hist[lid];
	barrier(CLK_LOCAL_MEM_FENCE);
	if(lid==0)
	{
		int sum = 0;
		for(int i=0;i<HISTOGRAM256_BIN_COUNT;i++)
		{
			sum+=sumhist[i];
			sumhist[i]=sum;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	dst[lid]= lid == 0 ? 0 : convert_uchar_sat(convert_float(sumhist[lid])*scale);
}
/*
///////////////////////////////equalizeHist//////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(256,1,1)))void equalizeHist(
							__global uchar * src,
							__global uchar * dst,
							__constant int * hist,
							int srcstep,
							int srcoffset,
							int dststep,
							int dstoffset,
							int width,
							int height,
							float scale,
							int inc_x,
							int inc_y)
{
	int gidx = get_global_id(0);
	int lid = get_local_id(0);
	int glb_size = get_global_size(0);
	src+=srcoffset;
	dst+=dstoffset;
	__local int sumhist[HISTOGRAM256_BIN_COUNT];
	__local uchar lut[HISTOGRAM256_BIN_COUNT+1];

	sumhist[lid]=hist[lid];
	barrier(CLK_LOCAL_MEM_FENCE);
	if(lid==0)
	{
		int sum = 0;
		for(int i=0;i<HISTOGRAM256_BIN_COUNT;i++)
		{
			sum+=sumhist[i];
			sumhist[i]=sum;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	lut[lid]= convert_uchar_sat(convert_float(sumhist[lid])*scale);
	lut[0]=0;
    int pos_y = gidx / width;
    int pos_x = gidx - mul24(pos_y, width);

    for(int pos = gidx; pos < mul24(width,height); pos += glb_size)
	{
		int inaddr = mad24(pos_y,srcstep,pos_x);
		int outaddr = mad24(pos_y,dststep,pos_x);
		dst[outaddr] = lut[src[inaddr]];
		pos_x +=inc_x;
		int off = (pos_x >= width ? -1 : 0);
		pos_x =  mad24(off,width,pos_x);
		pos_y += inc_y - off;
	}
}
*/

