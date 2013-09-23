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

#if defined (DOUBLE_SUPPORT)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#endif

__kernel void copy_to_with_mask(
        __global const GENTYPE* restrict srcMat,
        __global GENTYPE* dstMat,
        __global const uchar* restrict maskMat,
        int cols,
        int rows,
        int srcStep_in_pixel,
        int srcoffset_in_pixel,
        int dstStep_in_pixel,
        int dstoffset_in_pixel,
        int maskStep,
        int maskoffset)
{
    int x=get_global_id(0);
    int y=get_global_id(1);

    if (x < cols && y < rows)
    {
        int maskidx = mad24(y,maskStep,x+ maskoffset);
        if ( maskMat[maskidx])
        {
            int srcidx = mad24(y,srcStep_in_pixel,x+ srcoffset_in_pixel);
            int dstidx = mad24(y,dstStep_in_pixel,x+ dstoffset_in_pixel);
            dstMat[dstidx] = srcMat[srcidx];
        }
    }
}
