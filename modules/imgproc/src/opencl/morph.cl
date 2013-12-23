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
//    Yao Wang, bitwangyaoyao@gmail.com
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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#ifdef DEPTH_0
#ifdef ERODE
#define VAL 255
#endif
#ifdef DILATE
#define VAL 0
#endif
#endif
#ifdef DEPTH_5
#ifdef ERODE
#define VAL FLT_MAX
#endif
#ifdef DILATE
#define VAL -FLT_MAX
#endif
#endif
#ifdef DEPTH_6
#ifdef ERODE
#define VAL DBL_MAX
#endif
#ifdef DILATE
#define VAL -DBL_MAX
#endif
#endif

#ifdef ERODE
#define MORPH_OP(A,B) min((A),(B))
#endif
#ifdef DILATE
#define MORPH_OP(A,B) max((A),(B))
#endif
//BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii
#define ELEM(i,l_edge,r_edge,elem1,elem2) (i)<(l_edge) | (i) >= (r_edge) ? (elem1) : (elem2)

__kernel void morph(__global const uchar * restrict srcptr, int src_step, int src_offset,
                    __global uchar * dstptr, int dst_step, int dst_offset,
                    int src_offset_x, int src_offset_y,
                    int cols, int rows,
                    __constant uchar * mat_kernel,
                    int src_whole_cols, int src_whole_rows)
{
    int l_x = get_local_id(0);
    int l_y = get_local_id(1);
    int x = get_group_id(0)*LSIZE0;
    int y = get_group_id(1)*LSIZE1;
    int start_x = x+src_offset_x-RADIUSX;
    int end_x = x + src_offset_x+LSIZE0+RADIUSX;
    int width = end_x -(x+src_offset_x-RADIUSX)+1;
    int start_y = y+src_offset_y-RADIUSY;
    int point1 = mad24(l_y,LSIZE0,l_x);
    int point2 = point1 + LSIZE0*LSIZE1;
    int tl_x = point1 % width;
    int tl_y = point1 / width;
    int tl_x2 = point2 % width;
    int tl_y2 = point2 / width;
    int cur_x = start_x + tl_x;
    int cur_y = start_y + tl_y;
    int cur_x2 = start_x + tl_x2;
    int cur_y2 = start_y + tl_y2;
    int start_addr = mad24(cur_y,src_step, cur_x*(int)sizeof(GENTYPE));
    int start_addr2 = mad24(cur_y2,src_step, cur_x2*(int)sizeof(GENTYPE));
    GENTYPE temp0,temp1;
    __local GENTYPE LDS_DAT[2*LSIZE1*LSIZE0];

    int end_addr = mad24(src_whole_rows - 1,src_step,src_whole_cols*(int)sizeof(GENTYPE));
    //read pixels from src
    start_addr = ((start_addr < end_addr) && (start_addr > 0)) ? start_addr : 0;
    start_addr2 = ((start_addr2 < end_addr) && (start_addr2 > 0)) ? start_addr2 : 0;
    __global const GENTYPE * src;
    src = (__global const GENTYPE *)(srcptr+start_addr);
    temp0 = src[0];
    src = (__global const GENTYPE *)(srcptr+start_addr2);
    temp1 = src[0];
    //judge if read out of boundary
    temp0= ELEM(cur_x,0,src_whole_cols,(GENTYPE)VAL,temp0);
    temp0= ELEM(cur_y,0,src_whole_rows,(GENTYPE)VAL,temp0);

    temp1= ELEM(cur_x2,0,src_whole_cols,(GENTYPE)VAL,temp1);
    temp1= ELEM(cur_y2,0,src_whole_rows,(GENTYPE)VAL,temp1);

    LDS_DAT[point1] = temp0;
    LDS_DAT[point2] = temp1;
    barrier(CLK_LOCAL_MEM_FENCE);
    GENTYPE res = (GENTYPE)VAL;
    for(int i=0; i<2*RADIUSY+1; i++)
        for(int j=0; j<2*RADIUSX+1; j++)
        {
            res =
#ifndef RECTKERNEL
                mat_kernel[i*(2*RADIUSX+1)+j] ?
#endif
                MORPH_OP(res,LDS_DAT[mad24(l_y+i,width,l_x+j)])
#ifndef RECTKERNEL
                :res
#endif
                ;
        }
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    if(gidx<cols && gidy<rows)
    {
        int dst_index = mad24(gidy, dst_step, dst_offset + gidx * (int)sizeof(GENTYPE));
        __global GENTYPE * dst = (__global GENTYPE *)(dstptr + dst_index);
        dst[0] = res;
    }

}
