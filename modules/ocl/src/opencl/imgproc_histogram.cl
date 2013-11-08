//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Xu Pang, pangxu010@163.com
//    Wenju He, wenju@multicorewareinc.com
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
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
#define PARTIAL_HISTOGRAM256_COUNT     (256)
#define HISTOGRAM256_BIN_COUNT         (256)

#define HISTOGRAM256_WORK_GROUP_SIZE     (256)
#define HISTOGRAM256_LOCAL_MEM_SIZE      (HISTOGRAM256_BIN_COUNT)

#define NBANKS (16)
#define NBANKS_BIT (4)


__kernel __attribute__((reqd_work_group_size(HISTOGRAM256_BIN_COUNT,1,1)))void calc_sub_hist_D0(
                                                                      __global const uint4* src,
                                          int src_step, int src_offset,
                                                                      __global int* globalHist,
                                                                      int dataCount,  int cols,
                                          int inc_x, int inc_y,
                                          int hist_step)
{
        __local int subhist[(HISTOGRAM256_BIN_COUNT << NBANKS_BIT)]; // NBINS*NBANKS
        int gid = get_global_id(0);
        int lid = get_local_id(0);
        int gx  = get_group_id(0);
        int gsize = get_global_size(0);
        int lsize  = get_local_size(0);
        const int shift = 8;
        const int mask = HISTOGRAM256_BIN_COUNT-1;
        int offset = (lid & (NBANKS-1));// lid % NBANKS
        uint4 data, temp1, temp2, temp3, temp4;
        src += src_offset;

        //clear LDS
        for(int i=0, idx=lid; i<(NBANKS >> 2); i++, idx += lsize)
        {
            subhist[idx] = 0;
            subhist[idx+=lsize] = 0;
            subhist[idx+=lsize] = 0;
            subhist[idx+=lsize] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //read and scatter
        int y = gid/cols;
        int x = gid - mul24(y, cols);
        for(int idx=gid; idx<dataCount; idx+=gsize)
        {
              data = src[mad24(y, src_step, x)];
              temp1 = ((data & mask) << NBANKS_BIT) + offset;
              data >>= shift;
              temp2 = ((data & mask) << NBANKS_BIT) + offset;
              data >>= shift;
              temp3 = ((data & mask) << NBANKS_BIT) + offset;
              data >>= shift;
              temp4 = ((data & mask) << NBANKS_BIT) + offset;

              atomic_inc(subhist + temp1.x);
              atomic_inc(subhist + temp1.y);
              atomic_inc(subhist + temp1.z);
              atomic_inc(subhist + temp1.w);

              atomic_inc(subhist + temp2.x);
              atomic_inc(subhist + temp2.y);
              atomic_inc(subhist + temp2.z);
              atomic_inc(subhist + temp2.w);

              atomic_inc(subhist + temp3.x);
              atomic_inc(subhist + temp3.y);
              atomic_inc(subhist + temp3.z);
              atomic_inc(subhist + temp3.w);

              atomic_inc(subhist + temp4.x);
              atomic_inc(subhist + temp4.y);
              atomic_inc(subhist + temp4.z);
              atomic_inc(subhist + temp4.w);

              x += inc_x;
              int off = ((x>=cols) ? -1 : 0);
              x = mad24(off, cols, x);
              y += inc_y - off;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //reduce local banks to single histogram per workgroup
        int bin1=0, bin2=0, bin3=0, bin4=0;
        for(int i=0; i<NBANKS; i+=4)
        {
             bin1 += subhist[(lid << NBANKS_BIT) + i];
             bin2 += subhist[(lid << NBANKS_BIT) + i+1];
             bin3 += subhist[(lid << NBANKS_BIT) + i+2];
             bin4 += subhist[(lid << NBANKS_BIT) + i+3];
        }

        globalHist[mad24(gx, hist_step, lid)] = bin1+bin2+bin3+bin4;
}

__kernel void __attribute__((reqd_work_group_size(1,HISTOGRAM256_BIN_COUNT,1)))
calc_sub_hist_border_D0(__global const uchar* src, int src_step, int src_offset,
                        __global int* globalHist, int left_col, int cols,
                        int rows, int hist_step)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
        int lidy = get_local_id(1);
        int gx = get_group_id(0);
        int gy = get_group_id(1);
        int gn = get_num_groups(0);
        int rowIndex = mad24(gy, gn, gx);
//        rowIndex &= (PARTIAL_HISTOGRAM256_COUNT - 1);

        __local int subhist[HISTOGRAM256_LOCAL_MEM_SIZE];
        subhist[lidy] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        gidx = ((gidx>=left_col) ? (gidx+cols) : gidx);
        if(gidy<rows)
        {
            int src_index = src_offset + mad24(gidy, src_step, gidx);
            int p = (int)src[src_index];
//	    p = gidy >= rows ? HISTOGRAM256_LOCAL_MEM_SIZE : p;
            atomic_inc(subhist + p);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        globalHist[mad24(rowIndex, hist_step, lidy)] += subhist[lidy];
}

__kernel __attribute__((reqd_work_group_size(256,1,1)))void merge_hist(__global int* buf,
                __global int* hist,
                int src_step)
{
    int lx = get_local_id(0);
    int gx = get_group_id(0);

    int sum = 0;

    for(int i = lx; i < PARTIAL_HISTOGRAM256_COUNT; i += HISTOGRAM256_WORK_GROUP_SIZE)
        sum += buf[ mad24(i, src_step, gx)];

    __local int data[HISTOGRAM256_WORK_GROUP_SIZE];
    data[lx] = sum;

    for(int stride = HISTOGRAM256_WORK_GROUP_SIZE /2; stride > 0; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lx < stride)
            data[lx] += data[lx + stride];
    }

    if(lx == 0)
        hist[gx] = data[0];
}

__kernel __attribute__((reqd_work_group_size(256,1,1)))
void calLUT(__global uchar * dst, __constant int * hist, int total)
{
    int lid = get_local_id(0);
    __local int sumhist[HISTOGRAM256_BIN_COUNT];
    __local float scale;

    sumhist[lid] = hist[lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0)
    {
        int sum = 0, i = 0;
        while (!sumhist[i])
            ++i;

        if (total == sumhist[i])
        {
            scale = 1;
            for (int j = 0; j < HISTOGRAM256_BIN_COUNT; ++j)
                sumhist[i] = i;
        }
        else
        {
            scale = 255.f/(total - sumhist[i]);

            for (sumhist[i++] = 0; i < HISTOGRAM256_BIN_COUNT; i++)
            {
                sum += sumhist[i];
                sumhist[i] = sum;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    dst[lid]= convert_uchar_sat_rte(convert_float(sumhist[lid])*scale);
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
