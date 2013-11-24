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

__kernel void set_to_without_mask_C1_D0(__global uchar * scalar,__global uchar * dstMat,
        int cols,int rows,int dstStep_in_pixel,int offset_in_pixel)
{
        int x=get_global_id(0)<<2;
        int y=get_global_id(1);
        int idx = mad24(y,dstStep_in_pixel,x+ offset_in_pixel);
        uchar4 out;
        out.x = out.y = out.z = out.w = scalar[0];

        if ( (x+3 < cols) && (y < rows)&& ((offset_in_pixel&3) == 0))
        {
            *(__global uchar4*)(dstMat+idx) = out;
        }
        else
        {
             if((x+3 < cols) && (y < rows))
             {
                dstMat[idx] = out.x;
                dstMat[idx+1] = out.y;
                dstMat[idx+2] = out.z;
                dstMat[idx+3] = out.w;
             }
             if((x+2 < cols) && (y < rows))
             {
                dstMat[idx] = out.x;
                dstMat[idx+1] = out.y;
                dstMat[idx+2] = out.z;
             }
             else if((x+1 < cols) && (y < rows))
             {
                dstMat[idx] = out.x;
                dstMat[idx+1] = out.y;
             }
             else if((x < cols) && (y < rows))
             {
                dstMat[idx] = out.x;
             }
        }
}

__kernel void set_to_without_mask(__global GENTYPE * scalar,__global GENTYPE * dstMat,
        int cols, int rows, int dstStep_in_pixel, int offset_in_pixel)
{
        int x = get_global_id(0);
        int y = get_global_id(1);
        if ( (x < cols) & (y < rows))
        {
            int idx = mad24(y, dstStep_in_pixel, x + offset_in_pixel);
            dstMat[idx] = scalar[0];
        }
}
